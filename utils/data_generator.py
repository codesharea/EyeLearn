from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import seed
from copy import deepcopy
import numpy as np
import gc
import tensorflow as tf
import cv2
from tensorflow.keras import layers

import random

class AugmentingDataGenerator(ImageDataGenerator):
    
    def __init__(self, train_masks, val_masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #np.random.seed(42)
        self.train_masks = train_masks
        self.val_masks = val_masks
#         self.augmentation = augmentation()
    
    def self_mask_generator(self, rand_seed=42):
        if rand_seed:
            seed(rand_seed)
        kid = np.random.choice(range(len(self.train_masks)))
        selected_class = self.train_masks[kid]
        return selected_class
    
    def random_transform(self, target_img, aug_type=0):
        aug_params = {}
        imgsize = target_img.shape[0]
#         print('aug_type', aug_type)
        if aug_type == 0: # Random center Crop 0~0.2
            ratio = np.random.uniform(0, 0.2)
            cropsize = int(imgsize*ratio)
            img = target_img[cropsize:(imgsize-cropsize), cropsize:(imgsize-cropsize), :]
            img = cv2.resize(img, (imgsize, imgsize), interpolation = cv2.INTER_NEAREST)
            return img
        if aug_type == 1: # Rorate between -45 and 45 degrees 
            theta = np.random.uniform(-45, 45)
            return self.apply_transform(target_img, {"theta":theta})
        if aug_type == 2: # vertically room in or out 0~20%
            zx, zy = np.random.uniform(-0.2, 0.2, 2)
            return self.apply_transform(target_img, {"zx":zx, "zy":zy})
        if aug_type == 3: # height shift
            tx = np.random.uniform(-0.2, 0.2)
            return self.apply_transform(target_img, {"tx":tx})
        if aug_type == 4: # width shift
            ty = np.random.uniform(-0.2, 0.2)
            return self.apply_transform(target_img, {"ty":ty})
        
    
    def flow(self, x, y, batch_size, *args, **kwargs):
        generator = super().flow(x, y, batch_size=batch_size, seed=42, *args, **kwargs)

        seed = None if 'seed' not in kwargs else kwargs['seed']

        while True:
            # construct train samples for inpainting optimization 
            ori, idx_ori = next(generator) 

            mask = []
            for i in range(ori.shape[0]):
                imgi = ori[i]
                maski = self.self_mask_generator(seed)
                red, green, blue= imgi.T
                white_areas = (red == 0) & (blue == 0) & (green == 0)
                maski[white_areas.T]=(1,1,1)
                mask.append(maski)
            mask = np.array(mask)
            masked = deepcopy(ori)
            masked[mask==0] = 1.
            
            # construct the first augmented image
#             ori1_mask = []
            ori_1 = []
            for i in range(ori.shape[0]):
                imgi = self.random_transform(ori[i], aug_type=random.randint(0, 4))
                ori_1.append(imgi)
#                 maski = np.ones_like(imgi) # self.self_mask_generator(seed)
#                 red, green, blue= imgi.T
#                 white_areas = (red == 0) & (blue == 0) & (green == 0)
#                 maski[white_areas.T]=(1,1,1)
#                 ori1_mask.append(maski)
#             ori1_mask = np.array(ori1_mask)
#             ori_1[ori1_mask==0] = 1.
            ori_1 = np.array(ori_1)
            
            # construct the second augmented image
            ori_2 = []
            for i in range(ori.shape[0]):
                imgi = self.random_transform(ori[i], aug_type=random.randint(0, 4))
                ori_2.append(imgi)
#                 maski = np.ones_like(imgi) # self.self_mask_generator(seed)
#                 red, green, blue= imgi.T
#                 white_areas = (red == 0) & (blue == 0) & (green == 0)
#                 maski[white_areas.T]=(1,1,1)
#                 ori2_mask.append(maski)
#             ori2_mask = np.array(ori2_mask)
#             ori_2[ori2_mask==0] = 1.
            ori_2 = np.array(ori_2)
            one_mask = np.ones_like(ori_2, dtype=np.uint8)
                  
            gc.collect()
            yield [masked, mask, ori_1, ori_2, one_mask], [ori, idx_ori]
#             yield [masked, mask, masked_cc, mask_cc], [ori, lab_ori, lab_ori]
    
    def get_contrastive_index(self, lab_ori, bank_labs, num_negatives):
        # sample contrastive indexes 
        intra_indices = []
        inter_pos_indices = []
        inter_neg_indices = []
        bank_size = len(bank_labs)
        idx_cands = np.array(list(range(bank_size)))
        for i in range(len(lab_ori)):
            # intra-contrastive negative samples
            idx_intra = np.random.choice(idx_cands, num_negatives)
            intra_indices.append(idx_intra)
            
            # inter-contrastive positive (from same cluster) and negative (from different clusters) samples
            try:
                idx_inter_pos = np.random.choice(np.where(bank_labs==lab_ori[i])[0], 1)
                inter_pos_indices.append(idx_cands[idx_inter_pos])
            except: # if empty then randomly sample a positive from the current bank once
                inter_pos_indices.append(np.random.choice(idx_cands, 1))
            try:
                idx_inter_neg = np.random.choice(np.where(bank_labs!=lab_ori[i])[0], num_negatives)
            except: # if empty then randomly sample negatives from the current bank once
                idx_inter_neg = np.random.choice(np.where(bank_labs==lab_ori[i])[0], num_negatives)
            inter_neg_indices.append(idx_cands[idx_inter_neg])
            
        intra_indices = np.array(intra_indices)
        inter_indices = np.concatenate([inter_pos_indices, inter_neg_indices], axis=1) 
            
        return intra_indices, inter_indices      
            
        
    def flow_val(self, x, batch_size, *args, **kwargs):

        generator = super().flow(x, self.val_masks, batch_size=batch_size, seed=42, *args, **kwargs)

        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # construct train samples for inpainting optimization 
            ori, masks = next(generator) 
            mask = []
            for i in range(ori.shape[0]):
                imgi = ori[i]
                maski = masks[i]
#                 maski = self.self_mask_generator()
                red, green, blue= imgi.T
                white_areas = (red == 0) & (blue == 0) & (green == 0)
                maski[white_areas.T]=(1,1,1)
                mask.append(maski)
            mask = np.array(mask)
            masked = deepcopy(ori)
            masked[mask==0] = 1.

            gc.collect()
            
            yield [masked, mask], ori
    
    
                
