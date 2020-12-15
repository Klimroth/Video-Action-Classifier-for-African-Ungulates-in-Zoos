#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"

"""
Contains the functionalities to identify individuals and cut them out.
"""

import sys
MASK_RCNN_LIBRARY = ''
sys.path.append(MASK_RCNN_LIBRARY)

import configuration as cf
import os, shutil
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image, load_image_gt
from mrcnn.utils import Dataset
import numpy as np
from datetime import datetime
import cv2


### mrcnn class
class Prediction_Config(Config):
    NAME = " "
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MIN_CONFIDENCE = cf.DETECTION_MIN_CONFIDENCE

    def _setClassVariables(self, name, names_labels):
        self.NAME = name
        self.NUM_CLASSES = 1 + len(names_labels)
        self.DETECTION_MAX_INSTANCES = len(names_labels)


class Prediction_Dataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, night_images_path, names):

        n = 0
        for name in names:
            # define one class
            self.add_class("dataset", n + 1, name)
            n += 1

        # find all images
        for interval_num in sorted(os.listdir(night_images_path)):
            for filename in sorted(os.listdir(night_images_path + interval_num)):
                if not filename.endswith('.jpg'):
                    continue
                # extract image id
                image_id = filename[:-4]
                img_path = night_images_path + interval_num + '/' + filename
                # add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path)


    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_mask(self, image_id):
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

def _get_enclosurecode(species, zoo, enclosure):
    return species+'_'+zoo+'_'+str(enclosure)

def _check_configuration(species, zoo, enclosure, base_input):
    ret = True
    enclosure_code = _get_enclosurecode(species, zoo, enclosure)
    net, label = _get_network_and_label(species, zoo, enclosure)
    if not os.path.exists(base_input+enclosure_code):
        print("Error: Input folder for object detection not found:", base_input, enclosure_code)
        return False
    if not net:
        print("Error: No object detection network was found:", enclosure_code)
        return False
    if not label:
        print("Error: No labels were found:", enclosure_code)
        return False
        
    return ret

def _get_network_and_label(species, zoo, enclosure_num, 
                 basenets = cf.BASE_OD_NETWORK, 
                 zoonets = cf.ZOO_OD_NETWORK,
                 enclosurenets = cf.ENCLOSURE_OD_NETWORK,
                 labels = cf.OD_NETWORK_LABELS):
    
    enclosure_code = species+'_'+zoo+'_'+str(enclosure_num)
    zoo_code = species+'_'+zoo
    
    net = False
    label = False
    
    if enclosure_code in enclosurenets.keys():
        net = enclosurenets[enclosure_code]
    elif zoo_code in zoonets.keys():
        net = zoonets[zoo_code]
    elif species in basenets.keys():
        net = basenets[species]
    
    if enclosure_code in labels.keys():
        label = labels[enclosure_code]
    elif zoo_code in labels.keys():
        label = labels[zoo_code]
    elif species in labels.keys():
        label = labels[species]
        
    return net, label



def postprocess_boxes(yhat):
    
    def non_max_suppression(boxes, scores, threshold):
        
        def compute_iou(box, boxes, box_area, boxes_area):
            assert boxes.shape[0] == boxes_area.shape[0]
            ys1 = np.maximum(box[0], boxes[:, 0])
            xs1 = np.maximum(box[1], boxes[:, 1])
            ys2 = np.minimum(box[2], boxes[:, 2])
            xs2 = np.minimum(box[3], boxes[:, 3])
            intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
            unions = box_area + boxes_area - intersections
            ious = intersections / unions
            return ious
        
        assert boxes.shape[0] == scores.shape[0]
        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                               areas[scores_indexes])
            filtered_indexes = set((ious > threshold).nonzero()[0])
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)

    def remove_clones(boxes, class_ids, scores):
        # -! works only on two individuals so far
        if class_ids[0] == class_ids[1]:
            x = np.argmax(scores)  # number 0 or 1
            if class_ids[x] == 1:
                correct_label = 1
                false_label = 2
            else:
                correct_label = 2
                false_label = 1
            class_ids[x] = correct_label
            class_ids[1 - x] = false_label
        return boxes, class_ids


    def remove_double_boxes(boxes, class_ids, scores):
        # how to set iou-threshold ?
        iou_threshold = 0.8
        keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        boxes_ret = [boxes[i] for i in keep_indices]
        class_ids_ret = [class_ids[i] for i in keep_indices]
        scores_ret = [scores[i] for i in keep_indices]
    
        return boxes_ret, class_ids_ret, scores_ret
        
    def remove_small_boxes(boxes):
        return boxes # dirty fix because it is not necessary anymore

    
    # yhat is model detection return.
    boxes = yhat['rois']
    class_ids = yhat['class_ids']
    scores = yhat['scores']


    # -! works only on two individuals so far
    if len(boxes) >= 2:
        # remove multiple instances of individuals
        boxes, class_ids = remove_clones(boxes, class_ids, scores)
        # suppress double boxing
        boxes, class_ids, scores = remove_double_boxes(boxes, class_ids, scores)

    boxes = remove_small_boxes(boxes)
    
    return boxes, class_ids, scores

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _cut_out_prediction_data(od_prediction_set, od_model, od_cfg, enclosure_code, output_folder, label_names, img_size=cf.IMG_SIZE):
    
    def _individual_name_from_boxcode(label_names, boxcode):
        tmp = label_names[boxcode - 1]
        if tmp.startswith("Elenantilope"):
            tmp = tmp.replace("Elenantilope", "Elen")
        return tmp
    
    
    
    # load image and mask
    i = 1

    for image_id in od_prediction_set.image_ids:
        i += 1

        # load the image and mask
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(od_prediction_set, od_cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, od_cfg)
        sample = np.expand_dims(scaled_image, 0)
        
        # make prediction
        yhat = od_model.detect(sample, verbose=0)[0]
        img_path = od_prediction_set.image_info[image_id]['path']
        img_name = img_path.split("/")[-1]
        
        interval_num = img_path.split("/")[-2]       
     
        if i % 6250 == 0:
            print(datetime.now())
            print("Predicted " + str(i) + " images of " + output_folder)
            

        # plot each box
        boxes, classes, scores = postprocess_boxes(yhat)  # OnePerClassOutOfTwo(yhat)
        box_num = 0
        for box in boxes:
            pred_class = classes[box_num]
            ind_name = _individual_name_from_boxcode(label_names = label_names, boxcode=pred_class)
            # get coordinates
            y1, x1, y2, x2 = box

            box_part = image[y1:y2, x1:x2]
            box_part_rs = cv2.resize(box_part, img_size, interpolation=cv2.INTER_AREA)
            
            save_path = output_folder + ind_name + '/' + interval_num + '/'
            ensure_dir(save_path)
            cv2.imwrite(save_path + img_name, box_part_rs)
            
               
            text_file = open(save_path + "box_info.txt", "a+")
            text_file.write(img_name + "-" + str(y1) + "*" + str(x1) + "*" + str(y2) + "*" + str(x2) + "\n")
            text_file.close()

            box_num += 1

        os.remove(img_path)

def _predict_one_night(input_path_night, od_model, od_cfg, od_labels, enclosure_code, output_folder):
    
    def load_prediction_set(path = input_path_night, label_names = od_labels):
        prediction_set = Prediction_Dataset()
        prediction_set.load_dataset(night_images_path=path, names=label_names)
        prediction_set.prepare()
        print('In this folder, there are %d images to predict.' % len(prediction_set.image_ids))
        return prediction_set
     
    prediction_set = load_prediction_set()    
    _cut_out_prediction_data(prediction_set, od_model, od_cfg, enclosure_code, output_folder, od_labels)










    
        


def merge_timeinterval_images(path_to_intervalfolders, output_path_intervalimg, output_path_single_frame, output_path_text):
    
        
    """
    

    Parameters
    ----------
    path_to_intervalfolders : string
        TMP_STORAGE/intervals/enclosure_code/datum/
        contains for each individual a folder, each of those contains folders of intervals
    output_path_intervalimg : TYPE
        TMP_STORAGE/joint_images/enclosure_code/datum/.
    output_path_single_frame : TYPE
        TMP_STORAGE/single_frames/enclosure_code/datum/.

    Returns
    -------
    Writes output_path_intervalimg/interval_num.jpg and  output_path_single_frame/frame_num.jpg (up to 2 out of 4)

    """
    
    def _write_joint_image(imgpath_list, out_directory, time_interval):
        img_list = []
        for imgpath in imgpath_list:
            img = cv2.imread(imgpath)
            img_list.append(img)
        
        if len(img_list) == 0:
            return
        
        h, w, d = img_list[0].shape
        
        img_black = np.zeros([w, h, d],dtype=np.uint8)
        if len(img_list) == 1:
            vis1 = np.concatenate((img_list[0], img_black), axis=1)            
            vis2 = np.concatenate((img_black, img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
                
        elif len(img_list) == 2:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_black, img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        elif len(img_list) == 3:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_list[2], img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        elif len(img_list) == 4:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_list[2], img_list[3]), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        vis = cv2.resize(vis, (h, w), interpolation=cv2.INTER_AREA)
        
        ensure_dir(out_directory)
        cv2.imwrite(out_directory + str(time_interval).zfill(7) + ".jpg", vis)
    
    def _write_single_frame(imgpath_list, out_directory):
        
        img_list = []
        img_names = []
        ensure_dir(out_directory)
        
        for imgpath in imgpath_list:
            img = cv2.imread(imgpath)
            img_list.append(img)
            name = imgpath.split("/")[-1]
            img_names.append(name)
        
        if len(img_list) == 0:
            return
        
        if len(img_list) == 1:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
                
        elif len(img_list) in [2, 3]:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
            cv2.imwrite(out_directory + img_names[1], img_list[1])
        
        elif len(img_list) == 4:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
            cv2.imwrite(out_directory + img_names[2], img_list[2])
        
            
    
    for individual_name in os.listdir(path_to_intervalfolders):
                    
        for time_interval in os.listdir(path_to_intervalfolders + individual_name + "/" ):
            curr_path = path_to_intervalfolders + individual_name + "/" + time_interval + "/"
            imgpath_list = [curr_path + f for f in os.listdir(curr_path) if f.endswith("jpg")]            
            position_file = [curr_path + f for f in os.listdir(curr_path) if f.endswith("txt")]
            
            out_directory_joint = output_path_intervalimg + individual_name + '/0/'        
            _write_joint_image(imgpath_list, out_directory_joint, time_interval)
            
            out_directory_single = output_path_single_frame + individual_name + '/0/'
            _write_single_frame(imgpath_list, out_directory_single)
            
            # write text document with the position information
            if len(position_file):
                position_file = position_file[0]
                out_directory_text = output_path_text + individual_name + '/'
                ensure_dir(out_directory_text)
                shutil.copy(position_file, out_directory_text + time_interval.zfill(7) + '.txt')



def predict_multiple_nights(species, zoo, enclosure_num, input_path = cf.TMP_STORAGE_IMAGES, output_folder_base = cf.TMP_STORAGE_CUTOUT):
    """
    

    Parameters
    ----------
    input_path : string
        Path to .../enclosure_code/, contains the dates (folders) to predict        
    species : TYPE
        DESCRIPTION.
    zoo : TYPE
        DESCRIPTION.
    enclosure_num : TYPE
        DESCRIPTION.
    output_folder_base : string
        Folder of the form .../, will create 
            first, a subfolder intervals/enclosurecode/date/interval/frame_num.jpg (during prediction)
            
            then, (during merging)
            a subfolder joint_images/enclosurecode/date/individualname/0/interval_num.jpg
            a subfolder single_frames/enclosurecode/date/individualname/0/frame_num.jpg
            a subfolder position_files/enclosurecode/date/individualname/interval_num.txt

    Returns
    -------
    None.

    """
    net_path, od_labels = _get_network_and_label(species, zoo, enclosure_num)
    if not net_path or not od_labels:
        print("Network and / or Labels are not found.", net_path, od_labels)
        return
        
    enclosure_code = _get_enclosurecode(species, zoo, enclosure_num)
        
    cfg = Prediction_Config()
    cfg._setClassVariables(name=enclosure_code, names_labels=od_labels)
        
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights(net_path, by_name=True)

    print("Configuration Name: " + str(cfg.NAME))
    print("Configuration maximum Boxes: " + str(cfg.DETECTION_MAX_INSTANCES))
    print("Network: " + net_path)
    print("Possible labels: " + str(od_labels))
        
    for datum in os.listdir(input_path):
        print("Object detection: ", species, zoo, enclosure_num, datum)
        print(datetime.now())
        _predict_one_night(input_path + datum + '/', model, cfg, od_labels, enclosure_code, output_folder_base + 'intervals/' + enclosure_code + '/' + datum +'/')
        
        merge_timeinterval_images(path_to_intervalfolders = output_folder_base + 'intervals/' + enclosure_code + '/' + datum +'/', 
                                  output_path_intervalimg = output_folder_base + 'joint_images/' + enclosure_code + '/' + datum +'/',                                  
                                  output_path_single_frame = output_folder_base + 'single_frames/' + enclosure_code + '/' + datum +'/',
                                  output_path_text = output_folder_base + 'position_files/' + enclosure_code + '/' + datum +'/')

    shutil.rmtree(output_folder_base + 'intervals/', ignore_errors = True)
    