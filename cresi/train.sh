# train
JSON=$1
python 00_gen_folds.py $JSON
python 01_train.py $JSON --fold=0
