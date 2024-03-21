# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj
)

#import clip
from clip_model import *
import os
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.cuda.amp import autocast, GradScaler
from focal_loss.focal_loss import FocalLoss



ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'W2D',
    'CLIP',
    'linear_probe_CLIP',
    'feature_based_KD',
    'SCMD',
    'Baseline',
    'fine_tune_CLIP',
]
#-----------------------------------------------------------------------------------------------------------------------------------------------------
class MovingAvg:
    def __init__(self, network, start_iter):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = start_iter
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone()) / (1. + self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        with autocast():
            loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
                        self.network.parameters(),
                        lr=self.hparams["lr"],
                        weight_decay=self.hparams['weight_decay']
                        )
        MovingAvg.__init__(self, self.network, 300)
        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        with autocast():
            loss = F.cross_entropy(self.network(all_x), all_y)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))

class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        #print('mask_f', mask_f.shape)

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        #print('mask_b', mask_b.shape)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


#-------------------------------------------------------------------------------------------------------------------------------------
# zero-shot CLIP
class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP_Featurizer(self.hparams)
        self.model = self.featurizer.clip_model

        for param in self.model.parameters():
            param.requires_grad = False

        self.Embedding = self.featurizer.n_outputs

        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)

    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}

    def predict(self, x):
        logits_per_image, _ = self.model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)

#-------------------------------------------------------------------------------------------------------------------------------------
# fine_tune (or linear probing) the vision encoder
class linear_probe_CLIP(ERM): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(linear_probe_CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP_Featurizer(self.hparams)

        # linear probing, results showing that fune-tuning will distort the learned features and lead to worse performance.
        for param in self.featurizer.clip_model.parameters():
            param.requires_grad = False

        self.return_cls = self.featurizer.return_cls
        if self.return_cls:
            out_feature_shape = self.featurizer.width
        else:
            out_feature_shape = self.featurizer.n_outputs

        self.classifier = networks.Classifier(
            out_feature_shape,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lp_lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.scaler = GradScaler()
  
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        if self.return_cls:
            _, cls_token, _, _, _= self.featurizer(x)
            return self.classifier(cls_token)
        else:
            feature, _, normed_feature = self.featurizer(x)
            return self.classifier(normed_feature)

# fine_tune (or linear probing) the vision encoder
class fine_tune_CLIP(ERM): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(fine_tune_CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP_Featurizer(self.hparams)

        # linear probing, results showing that fune-tuning will distort the learned features and lead to worse performance.
        for param in self.featurizer.clip_model.parameters():
            param.requires_grad = True

        self.return_cls = self.featurizer.return_cls
        if self.return_cls:
            out_feature_shape = self.featurizer.width
        else:
            out_feature_shape = self.featurizer.n_outputs

        self.classifier = networks.Classifier(
            out_feature_shape,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lp_lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        if self.return_cls:
            _, cls_token, _, _, _= self.featurizer(x)
            return self.classifier(cls_token)
        else:
            feature, _, normed_feature = self.featurizer(x)
            return self.classifier(normed_feature)

# #-------------------------------------------------------------------------------------------------------------------------------------
# feature-based KD
class feature_based_KD(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(feature_based_KD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using device: ', self.device)

        # construct teacher model
        self.teacher_featurizer = networks.CLIP_Featurizer(self.hparams)
        # construct student model
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(
            self.student_featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        self.return_cls = self.teacher_featurizer.return_cls
        self.method = self.hparams['method']
        # ViT or RN50
        if self.return_cls:
            out_feature_shape = self.teacher_featurizer.width
        else:
            out_feature_shape = self.teacher_featurizer.n_outputs
        # linear projector or transformer encoder-decoder
        self.is_linear = self.hparams['linear_projector']
        if self.is_linear and self.method == 'linear':
            print("linear projector")
            self.feature_projector = nn.Linear(self.student_featurizer.n_outputs, out_feature_shape)
        elif not self.is_linear and self.method == 'linear':
            print("nonlinear projector")
            self.feature_projector = networks.CLS_Projector(in_features = self.student_featurizer.n_outputs, out_features = out_feature_shape, nhead = 8, num_layers = 1)
        else:
            print("fitnet")
            self.feature_projector = networks.FitNet(in_features = self.student_featurizer.n_outputs, out_features = out_feature_shape)
            
              
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': self.feature_projector.parameters()},
            ],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.num_classes = num_classes
        self.loss_weight = hparams['lambda']
        MovingAvg.__init__(self, self.student_network, 100)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')
        
    def update(self, minibatches, unlabeled=None, step=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        # renormalize the input for CLIP
        clip_x = networks.normalize(networks.denormalize(all_x))
        if self.return_cls:
            _, teacher_feature, _, _ = self.teacher_featurizer(clip_x)
        else:
            teacher_feature, _ = self.teacher_featurizer(clip_x)

    
        student_feature = self.student_featurizer(all_x)
        if self.method == 'fitnet':
            feature_loss = self.feature_projector(student_feature, teacher_feature)
        else:
            student_feature_proj = self.feature_projector(student_feature)
            feature_loss = F.mse_loss(student_feature_proj, teacher_feature)


        ce_loss = F.cross_entropy(self.student_classifier(student_feature), all_y)

        loss = ce_loss + self.loss_weight * feature_loss
        self.student_opt.zero_grad()
        loss.backward()
        self.student_opt.step()
        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_network(x)
        
   
class SCMD(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SCMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        # CLIP image encoder
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        # student network
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        return_cls = self.teacher.return_cls
        if return_cls:
            out_feature_shape = self.teacher.width
        else:
            out_feature_shape = self.teacher.n_outputs
        # multimodal projector
        self.feature_projector = nn.Linear(in_features = self.student_featurizer.n_outputs, out_features = self.teacher.n_outputs, bias = False)

        self.logits_scale = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': self.feature_projector.parameters()},
                {'params': [self.logits_scale], 'lr': 1e-4},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )

        # HP
        self.num_classes = num_classes
        self.loss_weight1 = hparams['lambda1']
        self.loss_weight2 = hparams['lambda2']
        self.k = hparams['last_k_epoch']
        self.p = hparams['worst_case_p']
        self.T = hparams['temperature']
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')

        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled = None, step = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        # select hard-to-learn samples
        if step <= int(5000 * (1 - self.k)):
            with torch.no_grad():
                student_feature, _ = self.student_featurizer(all_x, projection = True)
                student_pred = self.student_classifier(student_feature)
                loss_pre = F.cross_entropy(student_pred, all_y, reduction = 'none')
            _, loss_sort_index = torch.sort(loss_pre, descending = True)
            loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
            all_x = all_x[loss_sort_index]
            all_y = all_y[loss_sort_index]

        with autocast():
            student_feature, student_feature_list = self.student_featurizer(all_x, projection = True)
            student_pred = self.student_classifier(student_feature)
            ce_loss = F.cross_entropy(student_pred, all_y)
            
            # logits distill
            teacher_logits, _ = self.teacher.clip_model(all_x, self.prompt)
            teacher_prob = F.softmax(teacher_logits / self.T, dim = 1)
            student_log_prob = F.log_softmax(student_pred / self.T, dim = 1)
            logits_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2) 
            sample_loss = ce_loss + self.loss_weight1 * logits_loss

            # cross modality distillatoon
            student_image_feature = self.feature_projector(student_feature)
            # l2 norm
            student_image_feature = student_image_feature / student_image_feature.norm(dim = 1, keepdim = True)
            text_feature = self.teacher.clip_model.encode_text(self.prompt)
            text_feature = text_feature / text_feature.norm(dim = 1, keepdim = True)
            logits_scale = self.logits_scale.exp().to(student_image_feature.dtype)
            logits_per_image = logits_scale * student_image_feature @ text_feature.t()
            cross_modal_loss = F.kl_div(F.log_softmax(logits_per_image / self.T, dim = 1), teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2)
            feature_loss = cross_modal_loss
            # total loss
            loss = sample_loss + self.loss_weight2 * feature_loss

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        # clip to prevent scaling the ligits by more than 100
        with torch.no_grad():
            self.logits_scale.data = torch.clamp(self.logits_scale.data, 0, np.log(100))

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}
    
    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))


class SCMD_focal(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SCMD_focal, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        # CLIP image encoder
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        # student network
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        return_cls = self.teacher.return_cls
        if return_cls:
            out_feature_shape = self.teacher.width
        else:
            out_feature_shape = self.teacher.n_outputs
        # multimodal projector
        self.feature_projector = nn.Linear(in_features = self.student_featurizer.n_outputs, out_features = self.teacher.n_outputs, bias = False)

        self.logits_scale = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': self.feature_projector.parameters()},
                {'params': [self.logits_scale], 'lr': 1e-4},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )

        # HP
        self.num_classes = num_classes
        self.loss_weight1 = hparams['lambda1']
        self.loss_weight2 = hparams['lambda2']
        self.k = hparams['last_k_epoch']
        self.p = hparams['worst_case_p']
        self.T = hparams['temperature']
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')

        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo of a {ppt}') for ppt in classnames]).to(self.device)
        self.scaler = GradScaler()
        self.loss = FocalLoss(weights = torch.tensor(self.hparams['alpha']), gamma = self.hparams['gamma'])
        self.loss2 = FocalLoss(weights = torch.tensor(self.hparams['alpha']), gamma = self.hparams['gamma'], reduction = 'none')
    
    def update(self, minibatches, unlabeled = None, step = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        # select hard-to-learn samples
        if step <= int(5000 * (1 - self.k)):
            with torch.no_grad():
                student_feature, _ = self.student_featurizer(all_x, projection = True)
                student_pred = self.student_classifier(student_feature)
                #loss_pre = F.cross_entropy(student_pred, all_y, reduction = 'none')
                loss_pre = self.loss2(F.softmax(student_pred, dim = 1), all_y)
            _, loss_sort_index = torch.sort(loss_pre, descending = True)
            loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
            all_x = all_x[loss_sort_index]
            all_y = all_y[loss_sort_index]

        with autocast():
            student_feature, student_feature_list = self.student_featurizer(all_x, projection = True)
            student_pred = self.student_classifier(student_feature)
            #ce_loss = F.cross_entropy(student_pred, all_y)
            ce_loss = self.loss(F.softmax(student_pred, dim = 1), all_y)
            
            # logits distill
            teacher_logits, _ = self.teacher.clip_model(all_x, self.prompt)
            teacher_prob = F.softmax(teacher_logits / self.T, dim = 1)
            student_log_prob = F.log_softmax(student_pred / self.T, dim = 1)
            logits_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2) 
            sample_loss = ce_loss + self.loss_weight1 * logits_loss

            # cross modality distillatoon
            student_image_feature = self.feature_projector(student_feature)
            # l2 norm
            student_image_feature = student_image_feature / student_image_feature.norm(dim = 1, keepdim = True)
            text_feature = self.teacher.clip_model.encode_text(self.prompt)
            text_feature = text_feature / text_feature.norm(dim = 1, keepdim = True)
            logits_scale = self.logits_scale.exp().to(student_image_feature.dtype)
            logits_per_image = logits_scale * student_image_feature @ text_feature.t()
            cross_modal_loss = F.kl_div(F.log_softmax(logits_per_image / self.T, dim = 1), teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2)
            feature_loss = cross_modal_loss
            # total loss
            loss = sample_loss + self.loss_weight2 * feature_loss

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        # clip to prevent scaling the ligits by more than 100
        with torch.no_grad():
            self.logits_scale.data = torch.clamp(self.logits_scale.data, 0, np.log(100))

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}
    
    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))
        
        
class SCMD_no_CM(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SCMD_no_CM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        # CLIP image encoder
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        # student network
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        return_cls = self.teacher.return_cls
        self.logits_scale = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': [self.logits_scale], 'lr': 1e-4},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )

        # HP
        self.num_classes = num_classes
        self.loss_weight1 = hparams['lambda1']
        self.loss_weight2 = hparams['lambda2']
        self.k = hparams['last_k_epoch']
        self.p = hparams['worst_case_p']
        self.T = hparams['temperature']
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')

        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo a {ppt}') for ppt in classnames]).to(self.device)
        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled = None, step = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        # select hard-to-learn samples
        if step <= int(5000 * (1 - self.k)):
            with torch.no_grad():
                student_feature, _ = self.student_featurizer(all_x, projection = True)
                student_pred = self.student_classifier(student_feature)
                loss_pre = F.cross_entropy(student_pred, all_y, reduction = 'none')
            _, loss_sort_index = torch.sort(loss_pre, descending = True)
            loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
            all_x = all_x[loss_sort_index]
            all_y = all_y[loss_sort_index]

        with autocast():
            student_feature, student_feature_list = self.student_featurizer(all_x, projection = True)
            student_pred = self.student_classifier(student_feature)
            ce_loss = F.cross_entropy(student_pred, all_y)
            
            # logits distill
            teacher_logits, _ = self.teacher.clip_model(all_x, self.prompt)
            teacher_prob = F.softmax(teacher_logits / self.T, dim = 1)
            student_log_prob = F.log_softmax(student_pred / self.T, dim = 1)
            logits_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2) 
            sample_loss = ce_loss + self.loss_weight1 * logits_loss
            # total loss
            loss = sample_loss 

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}
    
    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))
        

class RKD(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RKD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        self.feature_projector = nn.Linear(in_features = self.student_featurizer.n_outputs, out_features = self.teacher.n_outputs, bias = False)
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': self.feature_projector.parameters()},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )
        self.weight = self.hparams['lambda1']
        self.w_dist = self.hparams['w_dist']
        self.w_angle = self.hparams['w_angle']
        self.num_classes = num_classes
        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo a {ppt}') for ppt in classnames]).to(self.device)
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')
        self.scaler = GradScaler()


    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)
        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)
        return loss
    
    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod   = torch.mm(feat, feat.t())
        feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)
        if not squared:
            feat_dist = feat_dist.sqrt()
        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0
        return feat_dist
    
    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist
        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist
        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)
        return loss
    

    def update(self, minibatches, unlabeled = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        with autocast():
            student_feature = self.student_featurizer(all_x)
            teacher_feature = self.teacher(all_x)   # encode_image
            student_feature_projected = self.feature_projector(student_feature)
            ce_loss = F.cross_entropy(self.student_classifier(student_feature), all_y)
            kd_loss = self.w_dist * self.rkd_dist(student_feature_projected, teacher_feature) + self.w_angle * self.rkd_angle(student_feature_projected, teacher_feature)
            loss = ce_loss + self.weight * kd_loss

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))
    

class FitNet(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FitNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        self.feature_projector = networks.FitNet(in_features = self.student_featurizer.n_outputs, out_features = self.teacher.n_outputs)
        #self.feature_projector = nn.Linear(in_features = self.student_featurizer.n_outputs, out_features = self.teacher.n_outputs, bias = False)
        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
                {'params': self.feature_projector.parameters()},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )
        self.weight = self.hparams['lambda1']
        self.num_classes = num_classes
        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo a {ppt}') for ppt in classnames]).to(self.device)
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')
        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        with autocast():
            student_feature = self.student_featurizer(all_x)
            teacher_feature = self.teacher(all_x)   # encode_image
            kd_loss = self.feature_projector(student_feature, teacher_feature)
            ce_loss = F.cross_entropy(self.student_classifier(student_feature), all_y)
            #kd_loss = F.mse_loss(student_feature_projected, teacher_feature)
            loss = ce_loss + self.weight * kd_loss

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))
        
class BSS(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(BSS, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)
        self.attack = networks.AttackBSS(targeted=True, num_steps=10, max_epsilon=16, step_alpha=0.3, cuda=True, norm=2)

        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )
        self.num_classes = num_classes
        self.loss_weight1 = hparams['lambda1']
        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo a {ppt}') for ppt in classnames]).to(self.device)
        self.scaler = GradScaler()
        self.T = hparams['temperature']
        self.attack_size = hparams['attack_size']

        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')

    def update(self, minibatches, unlabeled = None, step = None):
        ratio = max(3 * (1 - step / 5000), 0) + 1
        attack_ratio = max(2 * (1 - 4 / 3 * step / 5000), 0) + 0

        inputs = torch.cat([x for x, _ in minibatches])
        targets = torch.cat([y for _, y in minibatches])
        batch_size1 = inputs.shape[0]
        inputs, targets = Variable(inputs), Variable(targets)

        with autocast():
            out_s = self.student_classifier(self.student_featurizer(inputs))

            # Cross-entropy loss
            loss = F.cross_entropy(out_s[0:batch_size1, :], targets)
            out_t, _ = self.teacher.clip_model(inputs, self.prompt)

            # KD loss
            loss += - ratio * (F.softmax(out_t/self.T, 1).detach() * F.log_softmax(out_s/self.T, 1)).sum() / batch_size1

            if attack_ratio > 0:
                condition1 = targets.data == out_t.sort(dim=1, descending=True)[1][:, 0].data
                condition2 = targets.data == out_s.sort(dim=1, descending=True)[1][:, 0].data
                attack_flag = condition1 & condition2
                if attack_flag.sum():
                    # Base sample selection
                    attack_idx = attack_flag.nonzero().squeeze()
                    if attack_idx.shape[0] > self.attack_size:
                        diff = (F.softmax(out_t[attack_idx,:], 1).data - F.softmax(out_s[attack_idx,:], 1).data) ** 2
                        distill_score = diff.sum(dim=1) - diff.gather(1, targets[attack_idx].data.unsqueeze(1)).squeeze()
                        attack_idx = attack_idx[distill_score.sort(descending=True)[1][:self.attack_size]]

                    # Target class sampling
                    attack_class = out_t.sort(dim=1, descending=True)[1][:, 1][attack_idx].data
                    class_score, class_idx = F.softmax(out_t, 1)[attack_idx, :].data.sort(dim=1, descending=True)
                    class_score = class_score[:, 1:]
                    class_idx = class_idx[:, 1:]

                    rand_seed = 1 * (class_score.sum(dim=1) * torch.rand([attack_idx.shape[0]]).cuda()).unsqueeze(1)
                    prob = class_score.cumsum(dim=1)
                    for k in range(attack_idx.shape[0]):
                        for c in range(prob.shape[1]):
                            if (prob[k, c] >= rand_seed[k]).cpu().numpy():
                                attack_class[k] = class_idx[k, c]
                                break
                    
                    # Forward and backward for adversarial samples
                    attacked_inputs = Variable(self.attack.run(self.teacher.clip_model, inputs[attack_idx, :, :, :].data, attack_class, prompt = self.prompt))
                    batch_size2 = attacked_inputs.shape[0]

                    attack_out_t = self.student_classifier(self.student_featurizer(attacked_inputs))
                    attack_out_s, _ = self.teacher.clip_model(attacked_inputs, self.prompt)

                    # KD loss for Boundary Supporting Samples (BSS)
                    loss += - attack_ratio * (F.softmax(attack_out_t / self.T, 1).detach() * F.log_softmax(attack_out_s / self.T, 1)).sum() / batch_size2

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        if self.sma:
            self.update_sma()

        return {'loss': loss.item()}
    
    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))

class Baseline(Algorithm, MovingAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        # CLIP image encoder
        self.teacher = networks.CLIP_Featurizer(self.hparams)
        # student network
        self.student_featurizer = networks.Student_Featurizer(input_shape, self.hparams)
        self.student_classifier = networks.Classifier(self.student_featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier'])
        self.student_network = nn.Sequential(self.student_featurizer, self.student_classifier)

        self.student_opt = torch.optim.Adam(
            [
                {'params': self.student_featurizer.parameters()},
                {'params': self.student_classifier.parameters()},
            ],
            lr = self.hparams['lr'],
            weight_decay = self.hparams['weight_decay'],
        )

        # HP
        self.num_classes = num_classes
        self.loss_weight1 = hparams['lambda1']
        self.T = hparams['temperature']
        MovingAvg.__init__(self, self.student_network, 300)
        self.sma = hparams['SMA']
        if self.sma:
            print('Using SMA')

        classnames = [name.replace("_", ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(f'this is a photo a {ppt}') for ppt in classnames]).to(self.device)
        self.scaler = GradScaler()

    def update(self, minibatches, unlabeled = None, step = None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        with autocast():
            student_feature = self.student_featurizer(all_x)
            student_pred = self.student_classifier(student_feature)
            ce_loss = F.cross_entropy(student_pred, all_y)
            teacher_logits, _ = self.teacher.clip_model(all_x, self.prompt)
            teacher_prob = F.softmax(teacher_logits / self.T, dim = 1)
            student_log_prob = F.log_softmax(student_pred / self.T, dim = 1)
            logits_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction = 'batchmean') * (self.T ** 2)
            loss = ce_loss + self.loss_weight1 * logits_loss

        self.student_opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.student_opt)
        self.scaler.update()

        if self.sma:
            self.update_sma()
        return {'loss': loss.item()}
    
    def predict(self, x):
        if self.sma:
            self.network_sma.eval()
            return self.network_sma(x)
        else:
            return self.student_classifier(self.student_featurizer(x))
        
#--------------------------------------------------------------------------------------------------------------------------------------------
class W2D(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(W2D, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes
        self.drop_spatial = hparams['rsc_f_drop_factor']
        self.drop_batch = hparams['rsc_b_drop_factor']
        self.p = hparams['worst_case_p']
        self.k = hparams['last_k_epoch']

    # def update(self, minibatches, unlabeled=None, step=None, swa_model=None):
    #     device = "cuda" if minibatches[0][0].is_cuda else "cpu"
    #     # inputs
    #     all_x = torch.cat([x for x, y in minibatches])
    #     # labels
    #     all_y = torch.cat([y for _, y in minibatches])

    #     # sample dim
    #     if step <= int(3000 * (1 - self.k)):
    #         with torch.no_grad():
    #             all_p = self.predict(all_x)
    #             loss_pre = F.cross_entropy(all_p, all_y, reduction='none')
    #         _, loss_sort_index = torch.sort(-loss_pre)
    #         loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
    #         all_x = all_x[loss_sort_index]
    #         all_y = all_y[loss_sort_index]

    #     # all_x = self.featurizer.network.conv1(all_x)
    #     # all_x = self.featurizer.network.bn1(all_x)
    #     # all_x = self.featurizer.network.relu(all_x)
    #     # all_x = self.featurizer.network.maxpool(all_x)
    #     # all_x = self.featurizer.network.layer1(all_x)
    #     # all_x = self.featurizer.network.layer2(all_x)
    #     # all_x = self.featurizer.network.layer3(all_x)
    #     # all_x = self.featurizer.network.layer4(all_x)

    #     # feature dim
    #     # if self.training:
    #         # self.eval()
    #         # x_new = all_x.clone().detach()
    #         # x_new = Variable(x_new.data, requires_grad=True)
    #         # x_new_view = self.featurizer.network.avgpool(x_new)
    #         # x_new_view = x_new_view.view(x_new_view.size(0), -1)
    #         # output = self.classifier(x_new_view)
    #         # class_num = output.shape[1]
    #         # index = all_y
    #         # num_rois = x_new.shape[0]
    #         # num_channel = x_new.shape[1]
    #         # H = x_new.shape[2]
    #         # HW = x_new.shape[2] * x_new.shape[3]
    #         # one_hot = torch.zeros((1), dtype=torch.float32).cuda()
    #         # one_hot = Variable(one_hot, requires_grad=False)
    #         # sp_i = torch.ones([2, num_rois]).long()
    #         # sp_i[0, :] = torch.arange(num_rois)
    #         # sp_i[1, :] = index
    #         # sp_v = torch.ones([num_rois])
    #         # one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
    #         # one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
    #         # one_hot = torch.sum(output * one_hot_sparse)
    #         # self.zero_grad()
    #         # one_hot.backward()
    #         # grads_val = x_new.grad.clone().detach()
    #         # grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
    #         # feature_map_channel = grad_channel_mean
    #         # grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
    #         # cam_all = torch.sum(x_new * grad_channel_mean, 1)
    #         # cam_all = cam_all.view(num_rois, HW)
    #         # self.zero_grad()

    #         # spatial_drop_num = int(HW * self.drop_spatial)
    #         # th18_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, spatial_drop_num]
    #         # th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
    #         # mask_all_cuda = torch.where(cam_all > th18_mask_value, torch.zeros(cam_all.shape).cuda(),torch.ones(cam_all.shape).cuda())
    #         # mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)

    #         # cls_prob_before = F.softmax(output, dim=1)
    #         # x_new_view_after = x_new * mask_all
    #         # x_new_view_after = self.featurizer.network.avgpool(x_new_view_after)
    #         # x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
    #         # x_new_view_after = self.classifier(x_new_view_after)
    #         # cls_prob_after = F.softmax(x_new_view_after, dim=1)
    #         # sp_i = torch.ones([2, num_rois]).long()
    #         # sp_i[0, :] = torch.arange(num_rois)
    #         # sp_i[1, :] = index
    #         # sp_v = torch.ones([num_rois])
    #         # one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
    #         # before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
    #         # after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
    #         # change_vector = before_vector - after_vector - 0.0001
    #         # change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
    #         # th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.drop_batch))]
    #         # drop_index_fg = change_vector.gt(th_fg_value).long()
    #         # ignore_index_fg = 1 - drop_index_fg
    #         # not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
    #         # mask_all[not_01_ignore_index_fg.long(), :] = 1
    #         # self.train()
    #         # mask_all = Variable(mask_all, requires_grad=True)
    #         # all_x = all_x * mask_all

    #     #self.eval()
    #     all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
    #     all_f = self.featurizer(all_x)
    #     all_p = self.classifier(all_f)

    #     all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]
    #     percentiles = np.percentile(all_g.cpu(), self.drop_f, axis = 1)
    #     percentiles = torch.Tensor(percentiles)
    #     percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
    #     mask_f = all_g.lt(percentiles.to(device)).float()
    #     all_f_muted = all_f * mask_f
    #     all_p_muted = self.classifier(all_f_muted)

    #     all_s = F.softmax(all_p, dim = 1)
    #     all_s_muted = F.softmax(all_p_muted, dim = 1)
    #     changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
    #     percentile = np.percentile(changes.detach().cpu(), self.drop_b)
    #     mask_b = changes.lt(percentile).float().view(-1, 1)
    #     mask = torch.logical_or(mask_f, mask_b).float()

    #     #self.train()
    #     all_x = self.classifier(all_f * mask)

    #     # all_x = self.featurizer.network.avgpool(all_x)
    #     # all_x = all_x.view(all_x.size(0), -1)
    #     # all_x = self.classifier(all_x)

    #     loss = F.cross_entropy(all_x, all_y)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return {'loss': loss.item()}
    def update(self, minibatches, unlabeled=None, step=None, swa_model=None):
        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])

        # sample dim
        if step <= int(3000 * (1 - self.k)):
            with torch.no_grad():
                all_p = self.predict(all_x)
                loss_pre = F.cross_entropy(all_p, all_y, reduction='none')
            _, loss_sort_index = torch.sort(-loss_pre)
            loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
            all_x = all_x[loss_sort_index]
            all_y = all_y[loss_sort_index]

        all_x = self.featurizer.network.conv1(all_x)
        all_x = self.featurizer.network.bn1(all_x)
        all_x = self.featurizer.network.relu(all_x)
        all_x = self.featurizer.network.maxpool(all_x)
        all_x = self.featurizer.network.layer1(all_x)
        all_x = self.featurizer.network.layer2(all_x)
        all_x = self.featurizer.network.layer3(all_x)
        all_x = self.featurizer.network.layer4(all_x)

        # feature dim
        if self.training:
            self.eval()
            x_new = all_x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.featurizer.network.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.classifier(x_new_view)
            class_num = output.shape[1]
            index = all_y
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            feature_map_channel = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            cam_all = torch.sum(x_new * grad_channel_mean, 1)
            cam_all = cam_all.view(num_rois, HW)
            self.zero_grad()

            spatial_drop_num = int(HW * self.drop_spatial)
            th18_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
            mask_all_cuda = torch.where(cam_all > th18_mask_value, torch.zeros(cam_all.shape).cuda(),torch.ones(cam_all.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)

            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.featurizer.network.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.drop_batch))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            all_x = all_x * mask_all

        all_x = self.featurizer.network.avgpool(all_x)
        all_x = all_x.view(all_x.size(0), -1)
        all_x = self.classifier(all_x)

        loss = F.cross_entropy(all_x, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

        
class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved