# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Development"


import pickle
import os

import keras
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator

from efficientnet.keras import EfficientNetB3
from efficientnet.keras import preprocess_input

from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Sequential

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

GPU_USAGE = 1
NUM_GPUS = 1

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = str(GPU_USAGE)
session = InteractiveSession(config=config)

# User configuration

BS_PER_GPU = 8
NUM_EPOCHS = 30
SAVE_EVERY_EPOCH = 5

HEIGHT = 300
WIDTH = 300


BEHAVIOR_LIST = ["standing", "lying", "sleeping"]
NUM_CLASSES = len(BEHAVIOR_LIST)


DATA_PATH = ""
VAL_PATH = "" 
MODEL_SAVE_PATH_BASE = ""
MODEL_NAME_BASE = ""


NUM_CHANNELS = 3
INPUT_SHAPE = (WIDTH, HEIGHT, NUM_CHANNELS)




def augment_data(img):    
   
    
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)), 
        # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), 
        # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)),
        # blur images with a sigma of 0 to 3.0
        iaa.Multiply((0.7, 1.4), per_channel=0.0),
        iaa.LinearContrast((0.75, 1.25)),
        iaa.Affine(rotate=(-25, 25), shear=(-8, 8))   
        ])
    
    seq_det = seq.to_deterministic()
    aug_image = seq_det.augment_image(img)

    return preprocess_input(aug_image)
    
  

def get_steps(val_split = 0.08):
    len_base = len(os.listdir(DATA_PATH+"0/")) + len(os.listdir(DATA_PATH+"1/")) + len(os.listdir(DATA_PATH+"2/") )
    len_train = int(len_base*(1-val_split))
    len_val = int(len_base*(val_split))
    
    steps_per_epoch = int( 1.0*len_train / float(BS_PER_GPU*NUM_GPUS) )
    val_steps = int( 1.0*len_val / float(BS_PER_GPU*NUM_GPUS) )
    
    return steps_per_epoch, val_steps
   

def prepare_datasets(input_path = DATA_PATH, val_path = VAL_PATH, 
                     batch_size = BS_PER_GPU, size = (HEIGHT, WIDTH)):
    
    train_datagen = ImageDataGenerator(
             rescale=1./255,
             zoom_range=0.1,
             preprocessing_function=augment_data,
             validation_split=0)

    
    train_generator = train_datagen.flow_from_directory(
        input_path,
        shuffle = True,
        batch_size= batch_size,
        class_mode='categorical',
        target_size = size)
    
    validation_datagen = ImageDataGenerator(brightness_range = (1.0, 1.0),
                                            rescale=1./255,
                                            preprocessing_function=preprocess_input)
    
    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        shuffle=False,
        class_mode='categorical',
        target_size= size)
    
    return train_generator, validation_generator



def train_model(train_generator, validation_generator, save_path, checkpoint_path, model_name):
    
    base_model = EfficientNetB3(weights='noisy-student', include_top=False)                      
    
   
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(name="gap"),
        Dropout(0.2, name="dropout_out"),
        Dense(3, activation='softmax')
    ])
    
    opt = keras.optimizers.adam()
    
    model.compile(
                  optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if not os.path.exists(save_path+"logs/fit/"):
        os.makedirs(save_path+"logs/fit/")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        

    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path+model_name+"_cp-{epoch:04d}.ckpt",
                                                     verbose=1, 
                                                     save_weights_only=True,
                                                     save_best_only=False,
                                                     period=SAVE_EVERY_EPOCH)
    
    steps_per_epoch, val_steps = get_steps()
    history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=val_steps,
                              callbacks=[cp_callback],
                              initial_epoch = 0
                             )

    
    
    model.save(save_path+model_name+'.h5')
    
    with open(save_path+model_name+"_history.txt", "wb") as fp:
        pickle.dump(history.history, fp)




model_name_resnet = MODEL_NAME_BASE + '_efficientnetb5'
resnet_savepath = MODEL_SAVE_PATH_BASE + model_name_resnet + '/'
checkpoint_path_resnet = resnet_savepath + model_name_resnet + '_checkpoint/'

train_generator, validation_generator = prepare_datasets(input_path = DATA_PATH, val_path = VAL_PATH)

print("Created datasets... Starting with training.")
train_model(train_generator, validation_generator,
            save_path = resnet_savepath, 
            checkpoint_path = checkpoint_path_resnet, 
            model_name = model_name_resnet)


