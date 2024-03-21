dataset=PACS
algorithms="SCMD Baseline"
python -m domainbed.scripts.sweep delete_incomplete\
       --datasets ${dataset}\
       --algorithms ${algorithms} \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<sweep/full/hparams.json)"\
       --output_dir "sweep/${dataset}/outputs"


python -m domainbed.scripts.sweep launch\
       --datasets ${dataset}\
       --algorithms ${algorithms} \
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --hparams "$(<sweep/full/hparams.json)"\
       --output_dir "sweep/${dataset}/outputs"

