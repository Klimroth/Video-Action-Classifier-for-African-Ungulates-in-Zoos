#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"


"""
Contains the functionality to cut out images from the videos.
generate_raw_images: 
    Input: csv file (like training data all), output path and the cut-off value
    Writes the images to output path/enclosure_code/date/time-interval/framenumer.zfill(7).jpg
"""
import os, csv, cv2, configuration
import numpy as np
from datetime import datetime


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_raw_images(csv_input_file = configuration.INPUT_CSV_FILE, 
                        output_path = configuration.TMP_STORAGE_IMAGES, 
                        cut_off = configuration.CUT_OFF, 
                        csv_delimiter = configuration.CSV_DELIMITER, 
                        animal_num_sep = configuration.ANIMAL_NUMBER_SEPERATOR, 
                        interval_len = configuration.INTERVAL_LENGTH, 
                        images_per_interval = configuration.IMAGES_PER_INTERVAL, 
                        box_placement = configuration.VIDEO_ORDER_PLACEMENT, 
                        polygon_endpoints = configuration.VIDEO_BLACK_REGIONS, 
                        base_data_path = configuration.BASE_PATH_DATA):
    """
    

    Parameters
    ----------
    csv_input_file : string
        Path to a csv file that contains the nights which should be predicted.
    csv_delimiter : TYPE
        DESCRIPTION.
    animal_num_sep : TYPE
        DESCRIPTION.
    interval_len : TYPE
        DESCRIPTION.
    cut_off : TYPE
        DESCRIPTION.
    images_per_interval : TYPE
        DESCRIPTION.
    box_placement : TYPE
        DESCRIPTION.
    polygon_endpoints : TYPE
        DESCRIPTION.
    base_data_path : TYPE
        DESCRIPTION.
    output_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    video_list_merged_by_enclosure = _create_videolist_for_prediction(overview_file = csv_input_file, delim = csv_delimiter, animal_sep = animal_num_sep)
    
    for enclosure_code in video_list_merged_by_enclosure:
        print("Starting with enclosure "+enclosure_code)   
        for videolist_per_night in video_list_merged_by_enclosure[enclosure_code]:
            _create_pictures_from_videos(videolist = videolist_per_night, enclosure_code = enclosure_code, cut_off = cut_off, interval_len = interval_len,
                                                        images_per_interval = images_per_interval, output_path = output_path)




def _create_pictures_from_videos(videolist, enclosure_code, cut_off, interval_len, 
                                                images_per_interval, output_path, polygon_mapping = configuration.VIDEO_BLACK_REGIONS):
    """
    Input: videolist and corresponding labelfile
    Output: No ouput.
    I/O operations:
    - merges videos to one image
    - saves pictures to folders due to interval number
    """
    
    def _save_image(frame, path, filename):
        """
        Saves the np.array() frame as an image to path. 
        """
        ensure_dir(path)    
        cv2.imwrite(path+"/"+filename, frame) 
    
    def _add_black_polygon(img, config, polygon_mapping = polygon_mapping):
        """
        Input: image (np array), enclosure_code and an information where to put black polygons
        Output: Returns original image in there is no information for the given enclosure,
        otherwise it will add the designated black polygons
        """
        if not config in polygon_mapping:
            return img
        
        for polygon in polygon_mapping[config]:
            pts = polygon
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, np.int32([pts]), (0,0,0))
        return img

    
    def _get_date_from_videofile(videopath):
        """
        Input: path to videofile.
        Output: something like 2017-10-09_Elen_Kronberg_3.avi (where the 3 is the videonumber)
        requirement: Filename is like: /home/path/.../2017-10-09_Elen_Kronberg_3_SUM-10s_pred.csv
        """
        parts = videopath.split("/")[-1].split("_")
        return parts[0]
    
    def _get_frames_per_interval(interval_len, images_per_interval):
        return [int(1 + x * (interval_len - 1)/images_per_interval) for x in range(images_per_interval)]
    
    
    def _decide_width_height(width_dims, height_dims, amount_streams):
        is_low_res = False
        ratios = [width_dims[i]*1.0/height_dims[i] for i in range(len(width_dims))]
        #low res: 1.333
        #high res: 1.777 oder 1.666
        if min(ratios) < 1.4:
            is_low_res = True
    
        res = (0, 0)
    
    
        # 1 image
        # only one video_file (no arrangement necessary)
        # 1280, 720 (hd), 800, 600 (low res)
            
        # 2 images
        # there will be exactly two pictures side by side
        # 1280 x 360 (hd), 1280 x 480 (low res)
            
        # 3 or 4 images
        # square of pictures with one (zero) black frame
        # 1280 x 720 (hd), 1280 x 960 (low res)
    
        # 5 or 6 images
        # first row 3, second row 2 + one (no) black frame
        # 1278 x 720 (hd), 1278 x 480 (low res)
        if amount_streams == 1:
            if is_low_res:
                res = (800, 600)
            else:
                res = (1280, 720)    
        elif amount_streams in [2,3,4]:        
            if is_low_res:
                res = (640, 480)
            else:
                res = (640, 360)
        elif amount_streams in [5,6]:
            if is_low_res:
                res = (426, 320)
            else:
                res = (426, 240)
        else:
            print("Error: It is currently not supported to have more than 6 video streams!")
            return False, res
        return True, res
    
    def  _concatenate_frames(enclosure_code, frames, res, amount_streams):
        
        def _order_frames(enclosure_key, frame_arr, video_order_array = configuration.VIDEO_ORDER_PLACEMENT):
            """
            Input: array of frames from up to 6 videofiles.
            Output: the same frames but in an order that is given by the configuration
            """
            if not enclosure_key in video_order_array:
                return frame_arr
        
            perm = video_order_array[enclosure_key]
        
            if len(perm) != len(frame_arr):
                return frame_arr
            
            ret_list = [frame_arr[i-1] for i in perm]
            return ret_list

        """
        Input: Enclosure_code, Array of Frames of length at most 6, desired resolution res
        Output: One frame with ordered pictures side by side 
        """
        frames = _order_frames(enclosure_code, frames)
        img_black = np.zeros([res[1],res[0],3],dtype=np.uint8)
        if amount_streams == 1:
            vis = frames[0]
                
        elif amount_streams == 2:            
            vis = np.concatenate((frames[0], frames[1]), axis=1)
                
        elif amount_streams == 3:
            vis1 = np.concatenate((frames[0], frames[1]), axis=1)            
            vis2 = np.concatenate((frames[2], img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
    
        elif amount_streams == 4:
            vis1 = np.concatenate((frames[0], frames[1]), axis=1)
            vis2 = np.concatenate((frames[2], frames[3]), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
                
        elif amount_streams == 5:
            vis1 = np.concatenate((frames[0], frames[1], frames[2]), axis=1)
            vis2 = np.concatenate((frames[3], frames[4], img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
    
        elif amount_streams == 6:
            vis1 = np.concatenate((frames[0], frames[1], frames[2]), axis=1)
            vis2 = np.concatenate((frames[3], frames[4], frames[5]), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
    
        return vis
    
    
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))    

    if len(videolist) == 0:
        return
    
    date = _get_date_from_videofile(videolist[0])
    print("Creating pictures from videos for enclosure " + enclosure_code + " of date " + date + ".")
    
    # initialize counters
    frames_per_interval = _get_frames_per_interval(interval_len, images_per_interval)    
    interval_num = 0
    frame_within_interval = 0
    frame_num = 0
    success = True
    
    videos = []
    width_dims = []
    height_dims = []
    
    for vid_path in videolist:
        vcap = cv2.VideoCapture(vid_path)
        videos.append(vcap)
        width_dims.append(int(vcap.get(3)))
        height_dims.append(int(vcap.get(4)))
    
    
    frames = []
    helpcounter = 0
    while success:
        helpcounter += 1
        
        frames_suc = []
        for vid in videos:
            suc, frame = vid.read()
            frames_suc.append( (suc, frame) )
            success = success*suc

        if not success:
            continue
        
        frames = [x[1] for x in frames_suc]
        
           
        if frame_num == 0:
            success, res = _decide_width_height(width_dims, height_dims, len(videolist))

        if interval_num > cut_off:
            success = False
            
        if not success:
            continue

        if frame_num % interval_len == 0:
            interval_num += 1
            frame_within_interval = 0
            
        if helpcounter % 25000 == 0:
            print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ": Processing of "+str(helpcounter)+" of " + str(int(cut_off*interval_len))+ " frames is done.")

        # rescale each frame
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], res, interpolation=cv2.INTER_AREA)
        
        # concatenate single pictures
        vis = _concatenate_frames(enclosure_code, frames, res, len(videolist))
        # turn whole image to grayscale
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        # add black polygones (TODO: maybe add the polygon array as parameter here?)
        vis = _add_black_polygon(vis, enclosure_code)        
        
        
        if frame_within_interval not in frames_per_interval:
            frame_num += 1
            frame_within_interval += 1
            continue        
        

        # save the whole frame
        desig_path = output_path + date + "/" + str(interval_num) + "/"
        desig_filename = str(frame_num-1).zfill(7) + ".jpg"
        
        if frame_num > 1:
            _save_image(frame = vis, path = desig_path, filename = desig_filename)
        
  
                   
        frame_num += 1
        frame_within_interval += 1
        


    print("**********************************************************************")




def _create_videolist_for_prediction(overview_file, delim, animal_sep):
    """
    Returns a dictionary {Art_Zoo_Enclosure_Num: [video_list_per_day] }
    """
    return_dict = {}
    if not os.path.exists(overview_file):
        print("Error: Overview-CSV-file was not found.")
        return return_dict

    with open(overview_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                if len(row) != 6:
                    print("Overview file has the wrong format.")
                    return return_dict
            else:
                date = row[0]
                species = row[1]
                zoo = row[2]
                enclosure_num = row[3]
                video_nums = row[4].split(animal_sep)
                    
                avi_video_filelist = _get_videofile_list(species, zoo, video_nums, date)
           
                if len(avi_video_filelist) < 1:
                    continue

                dict_key = species+"_"+zoo+"_"+str(enclosure_num)
                if not dict_key in return_dict.keys():
                    return_dict[dict_key] = [ avi_video_filelist ]
                else:
                    return_dict[dict_key].append(avi_video_filelist)

            line_count += 1

    return return_dict




def _get_videofile_list(species, zoo, videolist, date, base_path = configuration.BASE_PATH_DATA):
    """
    

    Parameters
    ----------
    species : string
        species key.
    zoo : string
        zooname.
    videolist : list of integers
        contains the video numbers that need to be collected.
    date : string
        date in the form YYYY-MM-DD or .
    base_path : string
        starting point (directory) for the data.

    Returns
    -------
    list of strings containing the paths to the videos

    """
    
    def _correct_date(date):
        if not "." in date:
            return date
        return date.split(".")[2] + "-" + date.split(".")[1] + "-" + date.split(".")[0]
    
    vid_list = []
    for vid_num in videolist:
        vid_path = base_path+species+"/"+zoo+"/Videos/"+species+"_"+str(vid_num)+"/"+_correct_date(date)+"_"+species+"_"+zoo+"_"+str(vid_num)+".avi"
        if not os.path.exists(vid_path):
            print("Error: "+vid_path+" was not found.")
            return []
        vid_list.append(vid_path)
    return vid_list