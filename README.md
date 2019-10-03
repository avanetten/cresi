![Alt text](/results/images/header.png?raw=true "Header")

# CRESI #

## City-scale Road Extraction from Satellite Imagery ##

This repository provides an end-to-end pipeline to train models to detect road networks over entire cities, and also provide speed limits and travel time estimates for each roadway.  We have observed success with both [SpaceNet](https://spacenet.ai) imagery and labels, as well as Google satellite imagery with [OSM](https://openstreetmap.org) labels. The repository consists of pre-processing modules, deep learning segmentation model (based upon the winning SpaceNet 3 submission by [albu]((https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution)), post-proccessing modules to extract the road networks, inferred speed limits, and travel times.  Furthermore, we include modules to scale up network detection to the city-scale, rather than just looking at small image chips. 
For further details see:

1. Our [arXiv](https://arxiv.org/abs/1908.09715) paper
2. Blogs:
	1. [Large Road Networks](https://medium.com/the-downlinq/extracting-road-networks-at-scale-with-spacenet-b63d995be52d), 
	2. [Road Speeds](https://medium.com/the-downlinq/inferring-route-travel-times-with-spacenet-7f55e1afdd6d), 
	3. [OSM+Google Imagery](https://medium.com/the-downlinq/computer-vision-with-openstreetmap-and-spacenet-a-comparison-cc70353d0ace), 
	4. [Data Prep](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-1-imagery-and-label-preparation-598af46d485e)


____
### Install ###

1. Download this repository

2. Build docker image

	`nvidia-docker build -t cresi /path_to_cresi/docker`
	
3. Create docker container (all commands should be run in this container)

	`nvidia-docker run -it --rm -ti --ipc=host --name cresi_image cresi`
	

____
### Prep ###

1. Prepare train/test data, e.g.:

	`python /path_to_cresi/cresi/data_prep/speed_masks.py`
	
2. Edit .json file to select desired variables and point to appropriate directories


____
### Train ###

1. All at once

	`cd /path_to_cresi/cresi`
	
	`./train.sh jsons/sn5_baseline.json`


2. Run commands individually

	A. Generate folds (within docker image)

		`python /path_to_cresi/cresi/00_gen_folds.py jsons/sn5_baseline.json`

	B. Run train script (within docker image)

		`python /path_to_cresi/cresi/01_train.py jsons/sn5_baseline.json --fold=0`
	


____
### Test ###


1. All at once

	`cd /path_to_cresi/cresi`
	
	`./test.sh jsons/sn5_baseline.json`
	

2. Run commands individually


	A. Execute inference (within docker image)

		`python /path_to_cresi/cresi/02_eval.py jsons/sn5_baseline.json`

	B. Merge predictions (if required)

		`python /path_to_cresi/cresi/03a_merge_preds.py jsons/sn5_baseline.json`
	
	C. Stitch together mask windows (if required)

		`python /path_to_cresi/cresi/03b_stitch.py jsons/sn5_baseline.json`

	D. Extract mask skeletons

		`python /path_to_cresi/cresi/04_skeletonize.py jsons/sn5_baseline.json`
	
	E. Create graph

		`python /path_to_cresi/cresi/05_wkt_to_G.py jsons/sn5_baseline.json`

	F. Infer road travel time and speed limit

		`python /path_to_cresi/cresi/06_infer_speed.py jsons/sn5_baseline.json`
	

