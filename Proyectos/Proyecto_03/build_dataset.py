# USAGE
# python build_dataset.py

# import the necessary packages
from networks import config
from imutils import paths
import random
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
help_ = "The ID of the dataset to be built: 1-Breast Cancer Set (default), 2-Textured Ellipsoids"
parser.add_argument("-ds",
                    "--dataset",
                    help=help_,
                    type=int,
                    default=1)

args = parser.parse_args()
dataset_id = int(args.dataset)

# grab the paths to all input images in the original input directory
# and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_CANCER_DATASET)) if dataset_id == 1 else list(paths.list_images(config.ORIG_INPUT_ELLIPS_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# we'll be using part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the datasets that we'll be building
datasets = None

if dataset_id==1:
    datasets = [
	    ("training", trainPaths, config.TRAIN_CANCER_PATH),
	    ("validation", valPaths, config.VAL_CANCER_PATH),
	    ("testing", testPaths, config.TEST_CANCER_PATH)
    ]
else:
    datasets = [
	    ("training", trainPaths, config.TRAIN_ELLIPS_PATH),
	    ("validation", valPaths, config.VAL_ELLIPS_PATH),
	    ("testing", testPaths, config.TEST_ELLIPS_PATH)
    ]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		filename = None
		label = None
		is_anomaly = False
		# extract the filename of the input image and extract the
		# Cancer Dataset: class label ("0" for "negative" and "1" for "positive")
		# Ellips Dataset: class label ("not_anomalies_range" for "negative" and "anomalies_range" for "positive")
		if dataset_id==1:
		    filename = inputPath.split(os.path.sep)[-1]
		    label = filename[-5:-4]
		    is_anomaly = True if label=="1" else False
		else:
			filename = inputPath.split(os.path.sep)[-1]
			label = filename[filename.find("_") + 1 : filename.find(".")]
			is_anomaly = True if int(label) in range(9300, 10043) else False

		# construct the path to the destination image and then copy
		# the image itself
		if (dType == "training" or dType == "validation") and not is_anomaly:

			# build the path to the label directory
			labelPath = os.path.sep.join([baseOutput, label])

			# if the label output directory does not exist, create it
			if not os.path.exists(labelPath):
				print("[INFO] 'creating {}' directory".format(labelPath))
				os.makedirs(labelPath)

			p = os.path.sep.join([labelPath, filename])
			shutil.copy2(inputPath, p)
		if dType == "testing":

			# build the path to the label directory
			labelPath = os.path.sep.join([baseOutput, label])

			# if the label output directory does not exist, create it
			if not os.path.exists(labelPath):
				print("[INFO] 'creating {}' directory".format(labelPath))
				os.makedirs(labelPath)

			p = os.path.sep.join([labelPath, filename])
			shutil.copy2(inputPath, p)
