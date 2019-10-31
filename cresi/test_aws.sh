# test
python 02_eval.py jsons/sn5_baseline_aws.json
python 03a_merge_preds.py jsons/sn5_baseline_aws.json
python 03b_stitch.py jsons/sn5_baseline_aws.json
python 04_skeletonize.py jsons/sn5_baseline_aws.json
python 05_wkt_to_G.py jsons/sn5_baseline_aws.json
python 06_infer_speed.py jsons/sn5_baseline_aws.json
python 07a_create_submission_wkt.py jsons/sn5_baseline_aws.json
