# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "Matterport, Inc."]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "M. Hahn-Klimroth"
__status__ = "Development"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
MASK_RCNN_LIBRARY = ''
sys.path.append(MASK_RCNN_LIBRARY)

import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import mrcnn.model as modellib

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import imgaug.augmenters as iaa

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = InteractiveSession(config=config)
from mrcnn import utils


BASE_MODEL_PATH = '/home/z42/KI_Projekt/Object_Detection_Networks/basenet_elen.h5'
IMAGES_PATHS = [ ['/home/z42/Schreibtisch/Videos_Jenny/Annotation/Bilder/Elen/Hannover/1/',
                  '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Bilder/Elen/Hannover/2/',
                  '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Bilder/Elen/Hannover/3/'] ]
LABEL_PATHS = [ ['/home/z42/Schreibtisch/Videos_Jenny/Annotation/Label/Elen/Hannover/1/',
                  '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Label/Elen/Hannover/2/',
                  '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Label/Elen/Hannover/3/'] ]
AUGMENTATION = [True]
NAMES_LABELS = [["Elen_Hannover"]]
BASENAME = 'Elen_Hannover'

#NEW_MODEL_NAMES = [BASENAME + '250_2.h5', BASENAME + '250_2_augment.h5',
#                   BASENAME + '500_2.h5', BASENAME + '500_2_augment.h5',
#                   BASENAME + '500_viele.h5', BASENAME + '500_viele_augment.h5',
#                   BASENAME + '750_viele.h5', BASENAME + '750_viele_augment.h5']

NEW_MODEL_NAMES = ['Elen_Hannover']
NEW_MODEL_OUTPUT = ['/home/z42/KI_Projekt/Object_Detection_Trainingspath/']
CONFIG_NAME = 'Elen_Hannover'


#TRAININGSTEPS_PER_EPOCH = [400, 400, 800, 800, 800, 800, 1200, 1200]
TRAININGSTEPS_PER_EPOCH = [1200]
NUM_EPOCHS = 20
LAYERS = "heads" #heads or all


# class that defines and loads the kangaroo dataset
class Training_Dataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, images_paths, label_paths, names, is_train=True):
        n = 0
        for name in names:
            # define one class
            self.add_class("dataset", n+1, name)
            n += 1

        for filename in os.listdir(images_paths):
            # define data locations
            images_dir = images_paths
            annotations_dir = label_paths
            # find all images
            if not filename.endswith('.jpg'):
                continue
            anno_file = filename[:-4] + '.xml'
            if os.path.isfile(annotations_dir+anno_file):
                # extract image id
                image_id = filename.split('_')[-1][:-4] # check if annotations exists

                img_path = images_dir + filename
                ann_path = annotations_dir + anno_file
                # add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
    # load all bounding boxes for an image
    def extract_boxes(self, filename):
        # load and parse the file
        root = ElementTree.parse(filename)
        boxes = list()
        # extract each bounding box
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def extract_labels(self, filename):
        # load and parse the file
        root = ElementTree.parse(filename)
        labels = list()
        # extract each bounding box
        for label in root.findall('.//name'):
            labelid = self.class_names.index(label.text)
            labels.append(labelid)

        return labels

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        labels = self.extract_labels(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(labels[i])
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class Training_Config(Config):
    # define the name of the configuration
    NAME = CONFIG_NAME
    # number of classes (background + kangaroo)
    #NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #STEPS_PER_EPOCH = 400
    
    def __init__(self, trainingssteps_per_epoch, num_classes):
        self.STEPS_PER_EPOCH = int(trainingssteps_per_epoch / self.GPU_COUNT)
        self.NUM_CLASSES = 1 + num_classes
        self.BATCH_SIZE = 1
        Config.__init__(self)

        



def load_train_set(images_paths, label_paths, names_labels):
    # load the train dataset
    train_set = Training_Dataset()
    train_set.load_dataset(images_paths, label_paths, names=names_labels, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    return train_set

def load_test_set(images_paths, label_paths, names_labels):
    # load the test dataset
    test_set = Training_Dataset()
    test_set.load_dataset(images_paths, label_paths, names=names_labels, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    return test_set

# augmentation sequence
augmentation_sequence = iaa.Sequential([
    # Gaussian Blurring in 40%
    iaa.Sometimes(
        0.4,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.0),
    # Make some images brighter and some darker.
    iaa.Multiply((0.7, 1.35), per_channel=0.0),
], random_order=True)


def evaluate_model(images = '/home/z42/KI_Projekt/Annotation_Basenet_Ungulates/Bilder/', 
                   labels = '/home/z42/KI_Projekt/Annotation_Basenet_Ungulates/Label/', 
                   classes = ['Ungulate'],
                   model_weights = '/home/z42/KI_Projekt/Object_Detection_Networks/basenet_ungulates.h5',
                   samples = 1000,
                   IoU = 0.5):
    cfg = Training_Config(10, len(classes))
    model = MaskRCNN(mode='inference', config=cfg, model_dir='./')
    model.load_weights(model_weights, by_name = True)
    test_set = load_train_set(images, labels, classes)    
   
    image_ids = np.random.choice(test_set.image_ids, samples)
    APs = {}
    for j in range(1,100):
        APs[j] = []
    
    i = 0
    for image_id in image_ids:
        if i % 50 == 0:
            print(i)
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(test_set, cfg,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        for j in range(1,100):            
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'], 
                                 iou_threshold=j/100)
            APs[j].append(AP)
        i += 1
        
    
    ret = {}
    for j in range(1,100):
        ret[j] = np.mean(APs[j])
        
    return ret


# =============================================================================
# for j in range(len(IMAGES_PATHS)):
#     print("***********************************************")
#     print(NEW_MODEL_NAMES[j])
#     
#     # create testset and trainset
#     train_set = load_train_set(images_paths = IMAGES_PATHS[j], label_paths = LABEL_PATHS[j], names_labels = NAMES_LABELS[j])
#     test_set = load_test_set(images_paths = IMAGES_PATHS[j], label_paths = LABEL_PATHS[j], names_labels = NAMES_LABELS[j])
#     
#     
#     config = Training_Config(trainingssteps_per_epoch = TRAININGSTEPS_PER_EPOCH[j], num_classes = len(NAMES_LABELS[j]))
# 
#     model = MaskRCNN(mode='training', model_dir=NEW_MODEL_OUTPUT[j] + NEW_MODEL_NAMES[j][:-3], config=config)
#     model.load_weights(BASE_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
#     
#     if AUGMENTATION[j]:    
#         model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=NUM_EPOCHS, layers=LAYERS, augmentation = augmentation_sequence)
#     else:
#         model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=NUM_EPOCHS, layers=LAYERS)
#     
#     try:
#         session.close()
#         tf.reset_default_graph()
#         
#         config = ConfigProto()
#         config.gpu_options.allow_growth = True  
#         config.allow_soft_placement = True 
#         session = InteractiveSession(config=config)
#     except:
#         pass
# 
# 
# =============================================================================




#ap_all = evaluate_model(images = '/home/z42/KI_Projekt/Annotation_Basenet_Ungulates/Bilder/', 
#                   labels = '/home/z42/KI_Projekt/Annotation_Basenet_Ungulates/Label/', 
#                   classes = ['Ungulate'],
#                   model_weights = '/home/z42/KI_Projekt/Object_Detection_Networks/basenet_ungulates.h5',
#                   samples = 5000)
ap_elen = evaluate_model(images = '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Bilder/Elen/Kronberg/1_Studie/', 
                   labels = '/home/z42/Schreibtisch/Videos_Jenny/Annotation/Label/Elen/Kronberg/1_Studie/', 
                   classes = ['Elenantilope'],
                   model_weights = '/home/z42/KI_Projekt/Object_Detection_Networks/basenet_elen.h5',
                   samples = 100)

