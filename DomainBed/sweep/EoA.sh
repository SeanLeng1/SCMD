dataset=PACS
python -m domainbed.EOA\
        --data_dir "domainbed/data"\
        --dataset ${dataset}\
        --output_dir "sweep/${dataset}/outputs"\
        --save_dir "sweep/${dataset}"
