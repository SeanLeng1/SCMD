
python -m domainbed.scripts.sweep launch\
       --datasets PACS VLCS OfficeHome TerraIncognita DomainNet\
       --algorithms CLIP \
       --data_dir ./domainbed/data\
       --command_launcher local\
       --single_test_envs\
       --steps 2\
       --holdout_fraction 0.2\
       --n_hparams 1\
       --n_trials 2\
       --skip_confirmation\
       --output_dir "sweep/PACS/ViT_outputs"