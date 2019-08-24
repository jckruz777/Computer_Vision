# import the necessary packages
import os

# network base path
NET_BASE = os.path.sep.join([".", ".."])

# initialize the path to the *original* input directory of images
ORIG_INPUT_CANCER_DATASET = "datasets/orig"
ORIG_INPUT_ELLIPS_DATASET = "datasets/dataset"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_CANCER_PATH = "datasets/idc"
BASE_ELLIPS_PATH = "datasets/ellips"

# derive the training, validation, and testing directories
TRAIN_CANCER_PATH = os.path.sep.join([BASE_CANCER_PATH, "training"])
VAL_CANCER_PATH = os.path.sep.join([BASE_CANCER_PATH, "validation"])
TEST_CANCER_PATH = os.path.sep.join([BASE_CANCER_PATH, "testing"])

PATIENT_NORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_CANCER_DATASET, "10254", "0"])
PATIENT_ANORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_CANCER_DATASET, "10253", "1"])

TRAIN_ELLIPS_PATH = os.path.sep.join([BASE_ELLIPS_PATH, "training"])
VAL_ELLIPS_PATH = os.path.sep.join([BASE_ELLIPS_PATH, "validation"])
TEST_ELLIPS_PATH = os.path.sep.join([BASE_ELLIPS_PATH, "testing"])

ELLIPS_NORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_ELLIPS_DATASET, "no_anomalies"])
ELLIPS_ANORMAL = os.path.sep.join([NET_BASE, ORIG_INPUT_ELLIPS_DATASET, "anomalies"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1
