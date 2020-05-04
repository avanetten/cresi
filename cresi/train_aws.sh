# train
python prep.py
python 00_gen_folds.py jsons/sn5_baseline_aws.json
python 01_train.py jsons/sn5_baseline_aws.json --fold=0
