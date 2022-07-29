import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import gc
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
# from tensorflow.keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm
from tensorflow.keras.models import Model
from matplotlib.ticker import NullFormatter
from utils.map_handler import *
from utils.model_util import *
from copy import deepcopy
import cv2
import numpy as np
import random
# from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, ZeroPadding2D
from tensorflow.keras.models import Model
from utils.data_generator import AugmentingDataGenerator
from sklearn.utils import shuffle
from PIL import Image 

random.seed(42)

def load_dataset(data_path, local_path, num_train=20000, num_val=2000, img_size=256):
    
    train_maps_file = os.path.join(local_path, 'train_maps.npy')
#     train_masks_file = os.path.join(local_path, 'train_masks.npy')
    
    masks_all = np.load(data_path + 'train_mask_data_22987.npy')
    
    if os.path.exists(train_maps_file):
        train_maps = np.load(train_maps_file) 
        print('load data from disk')
    else:            
        kernel = np.ones((4,4), np.uint8)
        train_maps = []
#         train_masks = []
        margin = 3
        maps_all = np.load(data_path + 'train_rnflt_data_22953.npy')
#         maps_all = np.load(data_path + 'paramtune_rnflt_data_2k.npy')

        for i in range(len(maps_all)):
            img_t = maps_all[i][margin:-margin, margin:-margin]
            img_t = cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, kernel)
            img_t = cv2.resize(img_t, (img_size, img_size))
            img = to_3dmap(img_t, N=256)
#             img = img[6:(img_size-6), 6:(img_size-6), :]
#             img = cv2.resize(img, (img_size, img_size))
            train_maps.append(img)
        
        ids = list(range(len(train_maps)))
        random.shuffle(ids)
        train_maps = np.array(train_maps)[ids]
        
        np.save(train_maps_file, train_maps)
   

    #train_masks = np.ones_like(train_maps, dtype=np.uint8)
    datagen = AugmentingDataGenerator(train_masks=masks_all[:num_train], 
                                      val_masks=masks_all[-num_val:], rescale=1./255)
   
    return datagen, train_maps[:num_train], train_maps[-num_val:]


def load_dataset_tuneparams(data_path, local_path, num_train=1800, num_val=200, img_size=256):
    
    train_maps_file = os.path.join(local_path, 'train_maps_tuneparams.npy')
    train_mds_file = os.path.join(local_path, 'train_mds_tuneparams.npy')
    train_labs_file = os.path.join(local_path, 'train_labs_tuneparams.npy')
    masks_all = np.load(data_path + 'paramtune_mask_data_2k.npy')
    
    
    if os.path.exists(train_maps_file) and os.path.exists(train_mds_file) and os.path.exists(train_labs_file):
        train_maps = np.load(train_maps_file) 
        train_mds = np.load(train_mds_file)
        train_labs = np.load(train_labs_file)
        print('load data from disk')
    else:            
        kernel = np.ones((4,4), np.uint8)
        train_maps = []
        margin = 3
        maps_all = np.load(data_path + 'paramtune_rnflt_data_2k.npy')
        val_label_all = np.load(data_path + 'paramtune_label_data_2k.npy')
        val_md_all = np.load(data_path + 'paramtune_md_data_2k.npy')

        for i in range(len(maps_all)):
            img_t = maps_all[i][margin:-margin, margin:-margin]
            img_t = cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, kernel)
            img_t = cv2.resize(img_t, (img_size, img_size))
            img = to_3dmap(img_t, N=256)    
#             img = cv2.resize(img, (img_size, img_size))
            train_maps.append(img/255)
        
        ids = list(range(len(train_maps)))
        random.shuffle(ids)
        train_maps = np.array(train_maps)[ids]
        train_mds = val_md_all[ids]
        train_labs = val_label_all[ids]
        
        np.save(train_maps_file, train_maps)
        np.save(train_maps_file, train_maps)
        np.save(train_maps_file, train_maps)
   
    #train_masks = np.ones_like(train_maps, dtype=np.uint8)
                   
    # Create training generator
    datagen = AugmentingDataGenerator(train_masks=masks_all[:num_train], 
                                      val_masks=masks_all[-num_val:])
#     # Create validation generator
#     val_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    
    return datagen, train_maps[:num_train], train_maps[-num_val:], train_mds[:num_train], train_labs[:num_train]
