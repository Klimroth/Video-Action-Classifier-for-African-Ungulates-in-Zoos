#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"

import tensorflow as tf
import csv
import configuration as cf

import os
from keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import preprocess_input


from copy import deepcopy





def load_cnn(modelpath):
    model = tf.keras.models.load_model(modelpath)
    return model

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
            
def _load_data_generator(input_path , size = cf.IMG_SIZE, batch_s = cf.BATCH_SIZE_BEHAVIOR):
    
    prediction_datagen = ImageDataGenerator(brightness_range = (1.0, 1.0),
                                                    rescale=1./255,
                                                    preprocessing_function=preprocess_input)
        
    prediction_generator = prediction_datagen.flow_from_directory(
        input_path,
        shuffle=False,
        class_mode='categorical',
        target_size= size, batch_size=batch_s)
    
    return prediction_generator



def _sort_prediction(y_pred_stall, data_names_stall):
    # sort y_pred_stall alphabetically
    y_pred_stall_sorted = []
    data_names_stall_copy = deepcopy(data_names_stall)     
    data_names_stall_copy = sorted(data_names_stall_copy)
    numbers_in_prediction = []
    
    
    for j in range(len(data_names_stall_copy)):
        curr_name = data_names_stall_copy[j]        
        curr_img_num = int(curr_name.split("_")[-1].split(".")[0])
        
        orig_pos = data_names_stall.index(curr_name)
        to_app = deepcopy(y_pred_stall[orig_pos])
        
        y_pred_stall_sorted.append(to_app)           
        numbers_in_prediction.append(curr_img_num)
        

    return y_pred_stall_sorted, data_names_stall_copy, numbers_in_prediction

def predict_folder_joint_intervals(folder_path, individual_code, datum, 
                                 output_folder = cf.FINAL_STORAGE_PREDICTION_FILES, 
                                 amount_frames = cf.VIDEO_LENGTH, 
                                 batch_s = cf.BATCH_SIZE_BEHAVIOR):
    
    def _save_joint_prediction_to_csv(y_pred, y_data_names, individual_code, datum, output_folder, 
                                      amount_frames, interval_len = cf.INTERVAL_LENGTH):    
    
        y_pred, y_data_names, predicted_intervals = _sort_prediction(y_pred, y_data_names)
        _ensure_dir(output_folder + 'raw_csv/joint_images/')
        csv_file = output_folder + 'raw_csv/joint_images/' + datum + '_' +individual_code + '.csv'
        
        print("Creating CSV-file", csv_file)
        
        with open(csv_file, mode='w+') as output_csv:
            csv_write = csv.writer(output_csv, delimiter=",")
            csv_write.writerow(["interval number", "startframe", "endframe", 
                                cf.BEHAVIOR_NAMES[0],
                                cf.BEHAVIOR_NAMES[1],
                                cf.BEHAVIOR_NAMES[2],
                                cf.BEHAVIOR_NAMES[3]
                                ])
            amount_frames = int(amount_frames / interval_len) - 1
            
            index_img = 0
            for j in range(amount_frames):
                s_frame = interval_len*j + 1
                e_frame = interval_len*(j+1)
                if j in predicted_intervals: # bild vorhanden                    
                    csv_write.writerow([str(j+1).zfill(7), s_frame, e_frame, y_pred[index_img][0], y_pred[index_img][1], y_pred[index_img][2], 0])
                    index_img += 1
                else:
                    file_name = "No prediction on this image."
                    csv_write.writerow([ file_name , s_frame, e_frame, 0, 0, 0, 1])               



    print("Starting to predict behavior (joint images) of " + individual_code)
    path_to_network = cf.get_behaviornetwork_joint(individual_code)
    prediction_generator_stall = _load_data_generator(input_path = folder_path)    
    stall_model = load_cnn(modelpath = path_to_network)
    
    y_pred_stall = stall_model.predict_generator(prediction_generator_stall, verbose = 1)
    
    data_names_stall = [f.split("/")[-1] for f in prediction_generator_stall.filenames ]        
    _save_joint_prediction_to_csv(y_pred_stall, data_names_stall, individual_code, datum, 
                                  output_folder, amount_frames, interval_len = cf.INTERVAL_LENGTH)
    
def predict_folder_single_frames(folder_path, individual_code, datum, 
                                 output_folder = cf.FINAL_STORAGE_PREDICTION_FILES, 
                                 amount_frames = cf.VIDEO_LENGTH, 
                                 batch_s = cf.BATCH_SIZE_BEHAVIOR):
    
    def _save_frame_prediction_to_csv(y_pred, y_data_names, individual_code, datum, output_folder, 
                                      amount_frames):    
    
        y_pred_stall,data_names_stall, numbers_in_prediction = _sort_prediction(y_pred, y_data_names)
        _ensure_dir(output_folder + 'raw_csv/single_frames/')
        csv_file = output_folder + 'raw_csv/single_frames/' + datum + '_' +individual_code + '.csv'
        
        print("Creating CSV-file", csv_file)
        
        with open(csv_file, mode='w+') as output_csv:
            csv_write = csv.writer(output_csv, delimiter=",")
            csv_write.writerow(["image_name", "startframe", "endframe", 
                                cf.BEHAVIOR_NAMES[0],
                                cf.BEHAVIOR_NAMES[1],
                                cf.BEHAVIOR_NAMES[2],
                                cf.BEHAVIOR_NAMES[3]
                                ])
            amount_frames -= 1
            
            index_img = 0
            for j in range(1, amount_frames):
                if j in numbers_in_prediction: # bild vorhanden
                    img_name = data_names_stall[index_img]
                    frame_num = int(img_name.split("_")[-1][:-4])
                    csv_write.writerow([data_names_stall[index_img], frame_num, frame_num, 
                                        y_pred_stall[index_img][0], y_pred_stall[index_img][1], y_pred_stall[index_img][2], 0 ])
                    index_img += 1
                else:
                    file_name = "No prediction on this image."
                    csv_write.writerow([ file_name , j, j, 0, 0, 0, 1])

    
    print("Starting to predict behavior (single frames) of " + individual_code)
    path_to_network = cf.get_behaviornetwork_single(individual_code)
    
    prediction_generator_stall = _load_data_generator(input_path = folder_path)
    stall_model = load_cnn(modelpath = path_to_network) 
    y_pred_stall = stall_model.predict_generator(prediction_generator_stall, verbose = 1)

    data_names_stall = [f.split("/")[-1] for f in prediction_generator_stall.filenames ]
    _save_frame_prediction_to_csv(y_pred_stall, data_names_stall, individual_code, datum, output_folder, 
                                      amount_frames)