# CRESI #

## City-scale Road Extraction from Satellite Imagery ##

This repository is designed to train models to detect road networks and travel time estimates over entire cities.  We expand upon the the [framework](https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution) created by albu for the SpaceNet 3 competition, adding post-processing modules, travel time inference, and the ability to scale to large regions.  

____
### Install ###

0. Download this repository
1. Build docker container

	`nvidia-docker build -t cresi /path_to_cresi/docker`
	
2. Create docker image 

	`nvidia-docker run -it --rm -ti --ipc=host --name cresi_image cresi`

____
### Train ###
0. Prepare train/test data, e.g.:

	`/path_to_cresi/src/create_spacenet_masks.py`
	
1. Edit train/test .json file to point to appropriate directories
2. Generate folds
	`python /path_to_cresi/src/gen_folds.py json/resnet34_ave_speed_mc_focal.json`

3. Run train script (within docker image)

	`python /path_to_cresi/src/train_eval.py json/resnet34_ave_speed_mc_focal.json --fold=0 --training`
	

____
### Test ###

0. Execute inference

	`python /path_to_cresi/src/train_eval.py json/resnet34_ave_speed_mc_focal.json`

1. Merge predictions (if required)

	`python /path_to_cresi/src/merge_preds.py json/resnet34_ave_speed_mc_focal.json`
	
2. Stitch together mask windows

	`python /path_to_cresi/src/stitch.py json/resnet34_ave_speed_mc_focal.json`

3. Extract mask skeletons

	`python /path_to_cresi/src/ave_skeleton.py json/resnet34_ave_speed_mc_focal.json`
	
4. Create graph

	`python /path_to_cresi/src/wkt_to_G.py json/resnet34_ave_speed_mc_focal.json`

5. Infer road travel time and speed limit

	`python /path_to_cresi/src/skeleton_speed.py json/resnet34_ave_speed_mc_focal.json`
