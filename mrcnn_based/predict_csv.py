#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"


import csv, os, configuration, image_creation_functions, image_cutout_functions, behavior_prediction_functions, post_processing, shutil
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

# specify gpu to use ("0" or "1")
GPU_TO_USE = "0"


# Declare Error level and set precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_TO_USE

tf.compat.v1.logging.set_verbosity(0)
tf.autograph.set_verbosity(0)


def _get_basic_constants():
    # Make sure, there is only one enclosure that will be predicted and all video numbers and individual codes align
    enc_info = _get_enclosure_and_individuals(configuration.INPUT_CSV_FILE)
    if not enc_info:
        return False, False, False, False, False, False, False
    enclosure_code, video_numbers, individual_numbers, species, zoo = enc_info[0], enc_info[1], enc_info[2], enc_info[3], enc_info[4]
    enclosure_number = enclosure_code.split("_")[-1]
    return enclosure_code, video_numbers, individual_numbers, species, zoo, enclosure_number



def predict_enclosure_by_csv(gpu_code = GPU_TO_USE,
                             skip_image_creation = False,
                             skip_individual_detection = False,
                             skip_single_frame_behavior = False,
                             skip_joint_images_behavior = False,
                             skip_post_processing = False,
                             skip_moving_files = False):
    
    # load basic informations
    enclosure_code, video_numbers, individual_numbers, species, zoo, enclosure_number = _get_basic_constants()
    
    if not skip_image_creation:
        # create raw images
        print("*****************************************************************")
        print("Step 1: Create raw images from video files.")
        image_creation_functions.generate_raw_images()
    
    
    ### Create a tensorflow session
    tf_cfg = _get_tf_config()
    session = InteractiveSession(config=tf_cfg)
    if not skip_individual_detection:
        # Cut out images for each individual
        print("*****************************************************************")
        print("Step 2: Cut out individuals.")
        ### Create a tensorflow session
        tf_cfg = _get_tf_config()
        session = InteractiveSession(config=tf_cfg)
        
        image_cutout_functions.predict_multiple_nights(species = species, zoo = zoo, enclosure_num = enclosure_number)
    
    
    
    if not skip_single_frame_behavior:
        print("*****************************************************************")
        print("Step 3.1: Predict behavior - Single frames.")
        # Predict each night - single frame behavior
        # storage of temporary files as the input
        base_folder_single_frame = configuration.TMP_STORAGE_CUTOUT+'single_frames/'+enclosure_code+'/'
        
        if not os.path.exists(base_folder_single_frame):
            print("Error: Single frame images do not exist.", base_folder_single_frame)
            return
        
        
        # iterating through folder structure, each prediction (each night) gets a fresh session
        for datum in sorted(os.listdir(base_folder_single_frame)):
            j = 0
            for od_individual_code in sorted(os.listdir(base_folder_single_frame + datum + '/')):
                individual_code = species + '_' + zoo + '_' + individual_numbers[j]
                
                try:
                    session.close()
                    tf.reset_default_graph()
                    session = InteractiveSession(config=tf_cfg)
                except:
                    pass
                behavior_prediction_functions.predict_folder_single_frames(folder_path = base_folder_single_frame + datum + '/' + od_individual_code + '/',
                                                                           individual_code = individual_code, 
                                                                           datum = datum)
                j += 1
    
    if not skip_joint_images_behavior:
        print("*****************************************************************")
        print("Step 3.2: Predict behavior - Joint Images.")
        # same game with joint images
        # iterating through folder structure, each prediction (each night) gets a fresh session
        base_folder_joint_images = configuration.TMP_STORAGE_CUTOUT+'joint_images/'+enclosure_code+'/'
        if not os.path.exists(base_folder_joint_images):
            print("Error: Interval images do not exist.", base_folder_joint_images)
            return
        for datum in sorted(os.listdir(base_folder_joint_images)):
            j = 0
            for od_individual_code in sorted(os.listdir(base_folder_joint_images + datum + '/')):
                individual_code = species + '_' + zoo + '_' + individual_numbers[j]
                
                try:
                    session.close()
                    tf.reset_default_graph()
                    session = InteractiveSession(config=tf_cfg)
                except:
                    pass
                behavior_prediction_functions.predict_folder_joint_intervals(folder_path = base_folder_joint_images + datum + '/' + od_individual_code + '/',
                                                                           individual_code = individual_code, 
                                                                           datum = datum)
                j += 1
            
    
    
    if not skip_moving_files:
        # move temporary files (cut out images)
        shutil.move(configuration.TMP_STORAGE_CUTOUT, configuration.FINAL_STORAGE_CUTOUT, copy_function = shutil.copy2)
        
        # remove temporary files
        shutil.rmtree(configuration.TMP_STORAGE_CUTOUT, ignore_errors = True)
        shutil.rmtree(configuration.TMP_STORAGE_IMAGES, ignore_errors = True)
       
        
    
    # base output paths for csv files of the predictions
    base_output_single_frames = configuration.FINAL_STORAGE_PREDICTION_FILES + 'raw_csv/single_frames/'
    base_output_joint_images = configuration.FINAL_STORAGE_PREDICTION_FILES + 'raw_csv/joint_images/'
        
    if not skip_post_processing:
        if not os.path.exists(base_output_single_frames) or not os.path.exists(base_output_joint_images):
            print("Error: Could not find prediction csv folders.", base_output_joint_images, base_output_single_frames)    
        
        
        
        print("*****************************************************************")
        print("Step 4: Post-Processing")
        # Apply post processing per night
        sf_files = [f for f in os.listdir(base_output_single_frames) if f.endswith('.csv')]
        ji_files = [f for f in os.listdir(base_output_joint_images) if f.endswith('.csv')]
        
        for csv_filename in sf_files:
            if csv_filename not in ji_files:
                print("Warning: File was not found in joint images predictions. Night skipped.", csv_filename)
                continue
            datum = csv_filename.split("_")[0]
            individual_code = csv_filename.split("_")[1] + "_" + csv_filename.split("_")[2] + "_" + csv_filename.split("_")[3][:-4]
            
            
            # we need to somehow match the od folder of the position files with the actual individual...
            pos_file_folder_base = configuration.FINAL_STORAGE_CUTOUT + 'position_files/' + enclosure_code + '/' + datum + '/'
            j = 0
            for od_individual_code in sorted(os.listdir(pos_file_folder_base)):
                individual_code2 = species + '_' + zoo + '_' + individual_numbers[j]
                if individual_code == individual_code2:
                    pos_file_folder = pos_file_folder_base + od_individual_code + '/'
            
            if not configuration.STANDARD:
                post_processing.post_process_night(single_frame_csv = base_output_single_frames + csv_filename, 
                                               joint_interval_csv = base_output_joint_images + csv_filename, 
                                               individual_code = individual_code, 
                                               datum = datum, position_files = pos_file_folder, is_test = False)
            else:
                # standing, lying, sleeping, out
                post_processing.post_process_night(single_frame_csv = base_output_single_frames + csv_filename, 
                                               joint_interval_csv = base_output_joint_images + csv_filename, 
                                               individual_code = individual_code, 
                                               output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES,
                                               behavior_mapping = {0:0, 1:1, 2:2, 3:3, 4:4},
                                               datum = datum, position_files = pos_file_folder, is_test = False)
                # active, inactive, out
                post_processing.post_process_night(single_frame_csv = base_output_single_frames + csv_filename, 
                                               joint_interval_csv = base_output_joint_images + csv_filename, 
                                               individual_code = individual_code, 
                                               behavior_mapping = {0:0, 1:1, 2:1, 3:3, 4:4},
                                               output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES + 'binary/',
                                               datum = datum, position_files = pos_file_folder, is_test = False, extension = '_binary')
            print("**" + csv_filename)
        print("Process finished.")
    
    


def _get_tf_config(gpu_usage = GPU_TO_USE):
    config_tf = ConfigProto()
    #config_tf.gpu_options.visible_device_list = GPU_TO_USE
    config_tf.gpu_options.allow_growth = True
    config_tf.allow_soft_placement = True
    
    return config_tf
    
    
    
def _get_enclosure_and_individuals(csv_filepath = configuration.INPUT_CSV_FILE):
    
    ret = False
    animal_sep = configuration.ANIMAL_NUMBER_SEPERATOR
    delim = configuration.CSV_DELIMITER
    
    if not os.path.exists(csv_filepath):
        print("Error: Input-CSV-file was not found.")
        return ret
    
    enclosure_code, video_nums, individual_nums = "", [], []
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                if len(row) != 6:
                    print("Overview file has the wrong format.")
                    return ret
            elif line_count == 1:
                enclosure_code = row[1] + "_" + row[2] + "_" + row[3]
                video_nums = row[4].split(animal_sep)
                individual_nums = row[5].split(animal_sep)
                species = row[1]
                zoo = row[2]
            else:
                if row[1] + "_" + row[2] + "_" + row[3] != enclosure_code:
                    print("Error: Multiple enclosures given as input.")
                    return ret
                if video_nums != row[4].split(animal_sep):
                    print("Error: Different combinations of enclosure and video files given.")
                    return ret
                if individual_nums != row[5].split(animal_sep):
                    print("Error: Different individual numbers given in different nights.")
                    return ret
            line_count += 1
        return [enclosure_code, video_nums, individual_nums, species, zoo]