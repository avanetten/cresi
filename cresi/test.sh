# test
JSON=$1
python 02_eval.py $JSON
python 03a_merge_preds.py $JSON
python 03b_stitch.py $JSON
python 04_skeletonize.py $JSON
python 05_wkt_to_G.py $JSON
python 06_infer_speed.py $JSON
python 07a_create_submission_wkt.py $JSON
