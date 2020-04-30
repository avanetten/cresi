# test
JSON=$1
time python 02_eval.py $JSON
time python 03a_merge_preds.py $JSON
time python 03b_stitch.py $JSON
time python 04_skeletonize.py $JSON
time python 05_wkt_to_G.py $JSON
time python 06_infer_speed.py $JSON
time python 07_create_submission.py $JSON
