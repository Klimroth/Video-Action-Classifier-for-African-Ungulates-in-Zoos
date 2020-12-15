#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"



"""
Contains all the needed configurations, i.e.

# Object Detection
- BASE_OD_NETWORK = { 'species': path to od-network.h5 } (for those enclosures using the basenet)
- ZOO_OD_NETWORK = { 'species_zoo' : path to od-network.h5 } (for those enclosures using a different net)
- ENCLOSURE_OD_NETWORK = { 'species_zoo_enclosure' : path to od-network.h5 } (for those enclosures using a different net)
- OD_NETWORK_LABELS = {'key': [list of labels]}
- MINIMUM_CONFIDENCY


# Creating images from videos
- VIDEO_ORDER_PLACEMENT = { 'species_zoo_enclosure': [2,1] } # only gets an entry if the order of the videos should not be the natural order
- VIDEO_BLACK_REGIONS = { 'species_zoo_enclosure': [ np.array() ]  } # contains the polygon endpoints of the black areas in the videos
- CSV_DELIMITER
- ANIMAL_NUMBER_SEPERATOR


"""

import numpy as np


"""
Saving paths
"""
INPUT_CSV_FILE = ''


TMP_STORAGE_IMAGES = ''
TMP_STORAGE_CUTOUT = ''  

FINAL_STORAGE_CUTOUT = ''
FINAL_STORAGE_PREDICTION_FILES = ''

"""
General
"""

BEHAVIOR_NAMES = ['standing', 'lhu', 'lhd', 'out', 'truncated']
COLOR_MAPPING = {'standing': 'cornflowerblue', 
                 'lying': 'forestgreen', 
                 'sleeping': 'limegreen', 
                 'out': 'darkgrey',
                 'truncated': 'grey'}

BASE_PATH_DATA = ''
INTERVAL_LENGTH = 7
IMG_SIZE = (300, 300)

BATCH_SIZE_BEHAVIOR = 1

"""
Video (Enclosure specific)
"""

VIDEO_START_TIME = 17
VIDEO_END_TIME = 7
VIDEO_HOUR_DURATION = 14

VIDEO_LENGTH = VIDEO_HOUR_DURATION*3600
CUT_OFF = int(VIDEO_LENGTH / INTERVAL_LENGTH)


"""
Post Processing Rules
"""

STANDARD = True # if true, outputs one normal file and one binary prediction.

BEHAVIOR_MAPPING = {0:0, 1:1, 2:2, 3:3, 4:4} # merges behaviors accordingly

WEIGHTS_POSTPROCESS = [0.5, 0.5] # weight of the multiple-frame encoded network and the single frame network

TRUNCATION_X_MIN = 0 # marks bounding boxes below (height of image) this value as truncated 
TRUNCATION_Y_MIN = 2000 # marks bounding boxes right of (width of image) this value as truncated 

# strength of the rolling average
ROLLING_AVERAGE_SINGLE_FRAMES = 4*INTERVAL_LENGTH # strength of the rolling average
ROLLING_AVERAGE_JOINT_IMAGES = 4
ROLLING_AVERAGE_ENSEMBLE = 4

ROLLING_AVERAGE_WEIGHTS = np.array([1.0, 1.0, 1.0, 10**(-15), 1.0])

# minimum length of behaviors. S = lying-head down, L = lying-head up, A = standing
# =============================================================================
MIN_LEN_SLS = 6 
MIN_LEN_SLA = 6 
MIN_LEN_ALS = 6 
MIN_LEN_ALA = 25
#
MIN_LEN_SAS = 25
MIN_LEN_SAL = 25
MIN_LEN_LAS = 25
MIN_LEN_LAL = 25
#
MIN_LEN_ASA = 3 
MIN_LEN_ASL = 3 
MIN_LEN_LSA = 3 
MIN_LEN_LSL = 3 


MIN_LEN_OUT = 50
MIN_TIME_OUT = 200
#
MIN_LEN_TRUNCATION = 10 # shorter truncations will just count as the previous behavior.
MIN_LEN_TRUNCATION_SWAP = 10 # longer truncations will count as REAL_BEHAVIOR_LONG
TRUNCATION_REAL_BEHAVIOR_LONG = 1 # transfers those truncation of at least 70 seconds to lying. 
# to output the result without any changes in behavior apply MIN_LEN = 1 and REAL_BEHAVIOR = 4.



"""
Behavior Prediction
"""
BEHAVIOR_NETWORK_JOINT = {   
	'basenet_antilopes': '',
	'basenet_zebras': ''
    }
BEHAVIOR_NETWORK_SINGLE_FRAME = {   
	'basenet_antilopes': '',
	'basenet_zebras': ''
    }

def get_behaviornetwork_joint(individual_code):
        
    if individual_code in BEHAVIOR_NETWORK_JOINT.keys():
        return BEHAVIOR_NETWORK_JOINT[individual_code]
        
    species, zoo = individual_code.split("_")[0], individual_code.split("_")[1]
        
    if species + '_' + zoo in BEHAVIOR_NETWORK_JOINT.keys():
        return BEHAVIOR_NETWORK_JOINT[species + '_' + zoo]
        
    if species.startswith('Zebra'): # TODO: Add missing networks from time to time
        return BEHAVIOR_NETWORK_JOINT['basenet_zebra']
        
    return BEHAVIOR_NETWORK_JOINT['basenet_antilopes']

def get_behaviornetwork_single(individual_code):
        
    if individual_code in BEHAVIOR_NETWORK_SINGLE_FRAME.keys():
        return BEHAVIOR_NETWORK_SINGLE_FRAME[individual_code]
        
    species, zoo = individual_code.split("_")[0], individual_code.split("_")[1]
        
    if species + '_' + zoo in BEHAVIOR_NETWORK_SINGLE_FRAME.keys():
        return BEHAVIOR_NETWORK_SINGLE_FRAME[species + '_' + zoo]
        
    if species.startswith('Zebra'): # TODO: Add missing networks from time to time
        return BEHAVIOR_NETWORK_SINGLE_FRAME['basenet_zebra']
        
    return BEHAVIOR_NETWORK_SINGLE_FRAME['basenet_antilopes']


"""
IMAGE CUTOUT
"""
DETECTION_MIN_CONFIDENCE = 0.9 # bounding boxes with a lower score are dismissed.
DETECTION_MAX_INSTANCES = 1 # maximum number of bounding boxes per image.


BASE_OD_NETWORK = {
	'Ungulate' : '',
	'Eland': '',
	'Wildebeest': ''
}

ZOO_OD_NETWORK = {
    }

ENCLOSURE_OD_NETWORK = {
    }

OD_NETWORK_LABELS = {
    # general
	'Ungulate': ['Ungulate'],
	# species    
	'Eland': ['Eland'],
	'Wildebeest': ['Wildebeest'],
    # zoo

    # enclosures
}





"""
IMAGE CREATION FROM VIDEOS
"""
CSV_DELIMITER = ","
ANIMAL_NUMBER_SEPERATOR = ";"
IMAGES_PER_INTERVAL = 4

VIDEO_ORDER_PLACEMENT = {
}

"""
draws a black polygon onto the corresponding image, for instance 
'Species_Zoo_Enclosure' : [ np.array( [ [500,0], [600,360], [640, 360], [640,0] ]),
                         np.array( [ [1130,0], [1280, 360], [1280,0] ] ) ]
will draw one filled black rectangle and one black filled triangle.
"""

VIDEO_BLACK_REGIONS ={ 
}