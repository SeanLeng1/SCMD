dataset=PACS
python -m domainbed.scripts.sweep delete\
       --datasets ${dataset}\
       --algorithms feature_based_KD\
       --data_dir ./domainbed/data\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.2\
       --n_hparams 10\
       --n_trials 2\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams_FD.json)"\
       --output_dir "sweep/${dataset}/outputs"