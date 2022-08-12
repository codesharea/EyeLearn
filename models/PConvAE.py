import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D, Flatten, Layer, Multiply, Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from enum import Enum, unique
from utils import model_util
import tensorflow_probability as tfp
# from models import loss
tfd = tfp.distributions
kl = tf.keras.losses.KLDivergence()

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import warnings
warnings.filterwarnings('ignore')

from models.PartialConv import PConv2D

class BaseEnum(Enum):
    @classmethod
    def values(cls):
        list(map(lambda x: x.value, cls))

@unique
class EncoderReduction(BaseEnum):
    """
    Define various methods for reducing the encoder output of shape (w, h, f) to
    """
    GA_POOLING = 'ga_pooling'
    FLATTEN = 'flatten'

class PConvUnet(object):

    def __init__(self, img_rows=256, img_cols=256, reconstruct_w=1.0, intracon_w=0.001, intercon_w=0.001,
                 batch_size=4, num_clusters=8, num_negatives=20, bank_size=6000, vgg_weights="imagenet", 
                 inference_only=False, net_name='default', learn_rate=0.0002, momentum=0.9, weight_decay=1e-4,
                 gpus=2, vgg_device=None, encoder_reduction=EncoderReduction.GA_POOLING, projection_dim=100,
                 temperature=0.5, embed_dim=512, projection_head_layers=1, activation='relu'):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.inference_only = inference_only
        self.net_name = net_name
        self.w1 = reconstruct_w
        self.w2 = intracon_w
        self.w3 = intercon_w
        self.gpus = gpus
        self.vgg_device = vgg_device
        self.encoder_reduction = encoder_reduction
        self.projection_dim = projection_dim
        self.embed_dim = embed_dim
        self.projection_head_layers = projection_head_layers
        self.bank_size = bank_size
        self.num_clusters = num_clusters
        self.num_negatives = num_negatives
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.activation = activation

        # Scaling for VGG input
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Assertions
        assert self.img_rows >= 256, 'Height must be >256 pixels'
        assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0
        
        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
        self.vgg_layers = [3, 6, 10]

        # Instantiate the vgg network
        if self.vgg_device:
            with tf.device(self.vgg_device):
                self.vgg = self.build_vgg(vgg_weights)
        else:
            self.vgg = self.build_vgg(vgg_weights)
        
        # Create UNet-like model
        if self.gpus >= 1:
            self.model, inputs_mask = self.build_pconv_unet()
            self.compile_pconv_unet(self.model, inputs_mask, learn_rate)            
        else:
            with tf.device("/cpu:0"):
                self.model, inputs_mask = self.build_pconv_unet()
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, inputs_mask, learn_rate)
        
    def build_vgg(self, weights="imagenet"):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """        
            
        # Input image to extract features from
        img = Input(shape=(self.img_rows, self.img_cols, 3))

        # Mean center and rescale by variance as in PyTorch
        processed = Lambda(lambda x: (x-self.mean) / self.std)(img)
        
        # If inference only, just return empty model        
        if self.inference_only:
            model = Model(inputs=img, outputs=[img for _ in range(len(self.vgg_layers))])
            model.trainable = False
            model.compile(loss='mse', optimizer='adam')
            return model
                
        # Get the vgg network from Keras applications
        if weights in ['imagenet', None]:
            vgg = VGG16(weights=weights, include_top=False)
        else:
            vgg = VGG16(weights=None, include_top=False)
            vgg.load_weights(weights, by_name=True)
            print('load pretrained VGG model')

        # Output the first three pooling layers
        outputs = [vgg.layers[i](processed) for i in self.vgg_layers]        
        
        # Create model and compile
        model = Model(img, outputs)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model
    
#     def set_memory(self, memory_feats=None, memory_labs=None, clu_centers=None):
    def set_memory(self, memory_feats=None, clu_centers=None):
        # Memory bank
        K.set_value(self.memory_feats, memory_feats)
#         K.set_value(self.memory_labs, memory_labs)
        K.set_value(self.clu_centers, clu_centers)
    
    def build_pconv_unet(self, train_bn=True):  
        
        # define memory bank for representations and labels
        self.clu_centers = K.variable(np.zeros((self.num_clusters, self.projection_dim)),  dtype='float32')
        self.memory_feats = K.variable(np.zeros((self.bank_size, self.projection_dim)),  dtype='float32')
#         self.memory_labs = K.variable(np.zeros(self.bank_size, ),  dtype='int32')
        self.memory_ptr = K.variable(0, dtype='int32')
        
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, 3), name='inputs_img')
        inputs_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_mask')
        inputs_img_1 = Input((self.img_rows, self.img_cols, 3), name='inputs_img_1')
#         inputs_mask_1 = Input((self.img_rows, self.img_cols, 3), name='inputs_mask_1')
        inputs_img_2 = Input((self.img_rows, self.img_cols, 3), name='inputs_img_2')
        inputs_one_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_one_mask')
        
#         inputs_img_cc = Input((self.img_rows, self.img_cols, 3), name='inputs_img_cc')
#         inputs_mask_cc = Input((self.img_rows, self.img_cols, 3), name='inputs_mask_cc')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True, act='relu'):
            conv, mask = PConv2D(kernel_size=kernel_size, n_channels=3, mono=False, filters=filters,
                                 strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation(act)(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, self.embed_dim, 3)
        e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, self.embed_dim, 3, act=activation)
        
        print(self.encoder_reduction, "embed_dim", self.embed_dim)
        
        # LATENT REPRSENTATION
        reduced = self.reduce_encoder_output(encoder_output=e_conv8, encoder_reduction=self.encoder_reduction)
        latent_representation = Layer(name='encoder_output')(reduced) # final RNFLT representations
        projected_representation = self.add_contrastive_output(input=latent_representation, projection_dim=self.projection_dim,
                                        projection_head_layers=self.projection_head_layers)
        
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PConv2D(kernel_size=kernel_size, padding='same', filters=filters)([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
            
        d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
        d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
        outputs = Conv2D(3, 1, activation = 'sigmoid', name='reconstruct')(d_conv16)
#         with tf.compat.v1.get_default_graph().as_default():
        # output from the projection layer
        output_projection_model = Model(inputs=[inputs_img, inputs_mask], outputs=projected_representation, name='embed_model')
        output_projection_model.compile(loss='mse', optimizer='adam', experimental_run_tf_function=False)
        
        inputs_img_cc = tf.keras.layers.concatenate([inputs_img_1, inputs_img_2], axis=0)
        inputs_mask_cc = tf.keras.layers.concatenate([inputs_one_mask, inputs_one_mask], axis=0)
        
        projection_out_intra = output_projection_model([inputs_img_cc, inputs_mask_cc], training=True)
        projection_out_inter = output_projection_model([inputs_img, inputs_one_mask], training=True)
        
        projection_out_intra = Layer(name='intra_contrast')(projection_out_intra)
        projection_out_inter = Layer(name='inter_contrast')(projection_out_inter)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask, inputs_img_1, inputs_img_2, inputs_one_mask], 
                      outputs=[outputs, projection_out_intra, projection_out_inter])
        
        return model, inputs_mask


#     def update_memory(self, batch_feats, batch_labs):
    def update_memory(self, batch_feats):
        batch_feats = tf.math.l2_normalize(batch_feats, -1)
        ptr = int(self.memory_ptr)
        temp = ptr + self.batch_size
        if temp > self.bank_size:
            temp = temp % self.bank_size
            self.memory_feats[ptr:, :].assign(batch_feats[:-temp])
            self.memory_feats[:temp, :].assign(batch_feats[-temp:])
            self.memory_ptr = temp
#         assert self.bank_size % self.batch_size == 0
        else:
            self.memory_feats[ptr:temp, :].assign(batch_feats)
            self.memory_ptr = temp % self.bank_size  # move pointer
        
        print('memory_ptr:', self.memory_ptr)
    
    # intra-relationships sampled by indices
    def sample_by_indices(self):
        # sample K indices
        neg_feats = []
        for i in range(self.batch_size):
            idx = tf.random.shuffle(tf.range(self.bank_size))[:self.num_negatives]
            neg_feat = K.gather(self.memory_feats, idx)
            neg_feats.append(neg_feat)
        
        neg_feats = K.reshape(tf.concat(neg_feats, 0), (self.batch_size, self.num_negatives, -1))
        
        return neg_feats
    
    # inter-relationships sampled by clusters
    def sample_by_clusters(self, labs):
#         with tf.compat.v1.get_default_graph().as_default():
            # sample one pos from the same cluster
        pos_feats = []
        neg_feats = []
        for lab in [0,1,2,3]:
            pos_idx = tf.where(K.equal(self.memory_labs, lab))
            pos_idx = tf.random.shuffle(pos_idx)[:1]
            pos_feat = K.gather(self.memory_feats, pos_idx)
            pos_feats.append(pos_feat)

            neg_idx = tf.where(K.not_equal(self.memory_labs, lab))
            neg_idx = tf.random.shuffle(neg_idx)[:self.num_negatives]
            neg_feat = K.gather(self.memory_feats, neg_idx)
            neg_feats.append(neg_feat)

        pos_feats = K.reshape(tf.concat(pos_feats, 0), (self.batch_size, -1))
        neg_feats = K.reshape(tf.concat(neg_feats, 0), (self.batch_size, self.num_negatives, -1))

        return pos_feats, neg_feats
    
    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        
        # reconstruction loss
        reconstruct_loss = self.ae_loss_total(inputs_mask)
              
        # intra contrastive loss
        intra_contrastive_loss = self.intra_NTXentLoss()
        # inter contrastive loss
        inter_contrastive_loss = self.inter_NTXentLoss()
        
        weights = [self.w1, self.w2, self.w3]
        print('weight', weights)
        
        model.compile(
                 optimizer = Adam(lr=lr),
                 loss=[reconstruct_loss, intra_contrastive_loss, inter_contrastive_loss],
                 loss_weights=weights,
                 metrics={'reconstruct':'mse'}
                 )
        
        
    def reduce_encoder_output(self, encoder_output, encoder_reduction):
        if encoder_reduction == EncoderReduction.GA_POOLING:
            reduced = GlobalAveragePooling2D()(encoder_output)
        elif encoder_reduction == EncoderReduction.FLATTEN:
            reduced = Flatten()(encoder_output)
        else:
            raise ValueError()
        #reduced = Dense(512)(reduced)
        return reduced
    
    def add_contrastive_output(self, input, projection_head_layers, projection_dim):
        mid_dim = int(input.shape[-1])
        
        ph_layers = []
        for _ in range(projection_head_layers - 1):
            ph_layers.append(mid_dim)
        if projection_head_layers > 0:
            ph_layers.append(projection_dim)
        contrast_head = model_util.projection_head(input, ph_layers=ph_layers)
        print(contrast_head.shape)
        con_output = Flatten(name='con_output')(contrast_head)
#         con_output = Layer(name='con_output')(contrast_head)
        return tf.math.l2_normalize(con_output, -1)
    
    
#     def intra_NTXentLoss(self, hidden_norm=True):
#         cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
#         def loss(y_true, hidden):
#             if hidden_norm:
#                 hidden = tf.math.l2_normalize(hidden, -1)           
            
#             intra_neg_feats = self.sample_by_indices()
            
#             hidden1, hidden2 = tf.split(hidden, 2, 0)
#             labels = tf.one_hot(tf.range(self.batch_size), self.num_negatives)
#             masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)
#             logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / self.temperature
#             logits_ab = logits_ab * masks
            
#             logits_ac = tf.matmul(hidden1, intra_neg_feats, transpose_b=True) / self.temperature
#             logits_bc = tf.matmul(hidden2, intra_neg_feats, transpose_b=True) / self.temperature
            
#             print(labels.shape, logits_ab.shape, logits_ac.shape)
            
#             loss_a = cross_entropy_loss(labels, tf.concat([logits_ab, logits_ac], 1))
#             loss_b = cross_entropy_loss(labels, tf.concat([logits_ab, logits_bc], 1))
#             loss = loss_a + loss_b

#         return loss

    def intra_NTXentLoss(self, hidden_norm=False):
        
        def loss(y_true, hidden):
            #if 10>0:
            #    return 0.            

            if hidden_norm:
                hidden = K.l2_normalize(hidden, -1)           
            intra_indices = K.reshape(K.cast(y_true, dtype='int32'), (-1,))
#             intra_neg_feats = self.sample_by_indices()
            intra_neg_feats = K.gather(self.memory_feats, intra_indices)
            intra_neg_feats = K.reshape(intra_neg_feats, (self.batch_size, self.num_negatives, -1))
            
            hidden1, hidden2 = tf.split(hidden, 2, 0)
            labels = tf.sequence_mask([1]*self.batch_size, maxlen=(self.num_negatives+1), dtype=tf.int32)
            logits_ab = tf.reduce_sum(tf.multiply(hidden1, hidden2), axis=1, keepdims=True) / self.temperature
            
            logits_ac = tf.reduce_sum(tf.multiply(tf.expand_dims(hidden1, axis=1), intra_neg_feats), axis=-1) / self.temperature
            logits_bc = tf.reduce_sum(tf.multiply(tf.expand_dims(hidden2, axis=1), intra_neg_feats), axis=-1) / self.temperature
            
            logits_ac = tf.concat([logits_ab, logits_ac], axis=1)
            logits_bc = tf.concat([logits_ab, logits_bc], axis=1)
                        
            print(labels.shape, logits_ab.shape, logits_ac.shape)
            
            loss_a = self.cross_entropy_loss(labels, logits_ac)
            loss_b = self.cross_entropy_loss(labels, logits_bc)
            loss = loss_a + loss_b
            
            return loss

        return loss
    
    def inter_NTXentLoss(self, hidden_norm=False):
        
        def loss(y_true, hidden):
            #if 10>0:
            #    return 0.
            if hidden_norm:
                hidden = K.l2_normalize(hidden, -1)
            
            inter_indices = K.cast(y_true, dtype='int32')
            
            inter_indices_pos = inter_indices[:, 0]
            inter_indices_neg = K.reshape(inter_indices[:, 1:], (-1,))
            
#             inter_pos_feats, inter_neg_feats = self.sample_by_clusters(batch_labs)
#             intra_feats = K.gather(self.memory_feats, inter_indices)
#             inter_pos_feats = K.reshape(intra_feats[:self.batch_size],  (self.batch_size, -1))
#             inter_neg_feats = K.reshape(intra_feats[self.batch_size:], (self.batch_size, self.num_negatives, -1))
            inter_pos_feats = K.reshape(K.gather(self.memory_feats, inter_indices_pos), (self.batch_size, -1))
            inter_neg_feats = K.reshape(K.gather(self.memory_feats, inter_indices_neg), (self.batch_size, self.num_negatives, -1))

            print(inter_pos_feats.shape, inter_neg_feats.shape)

            labels = tf.sequence_mask([1]*self.batch_size, maxlen=(self.num_negatives+1), dtype=tf.int32)
            logits_ab = tf.reduce_sum(tf.multiply(hidden, inter_pos_feats), axis=1, keepdims=True) / self.temperature

            logits_ac = tf.reduce_sum(tf.multiply(tf.expand_dims(hidden, axis=1), inter_neg_feats), axis=-1) / self.temperature
            logits_ac = tf.concat([logits_ab, logits_ac], axis=1)

            loss = self.cross_entropy_loss(labels, logits_ac)

            # update the memory
#             self.update_memory(hidden, batch_labs)
            self.update_memory(hidden)

            return loss
            
        return loss
    
    def ae_loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components 
        and multiplies by their weights. See paper eq. 7.
        """
        def loss(y_true, y_pred):
            
            # Compute predicted image with non-hole pixels set to ground truth
            y_comp = Multiply()([mask,y_true]) + Multiply()([1-mask, y_pred])
#             y_comp = mask * y_true + (1-mask) * y_pred
            # Compute the vgg features. 
            if self.vgg_device:
                with tf.device(self.vgg_device):
                    vgg_out = self.vgg(y_pred)
                    vgg_gt = self.vgg(y_true)
                    vgg_comp = self.vgg(y_comp)
            else:
                vgg_out = self.vgg(y_pred)
                vgg_gt = self.vgg(y_true)
                
#                 print(y_true.shape, y_pred.shape, y_comp.shape)
                
                vgg_comp = self.vgg(y_comp)
            
#             print('y_comp:', y_comp.shape)
            # Compute loss components
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_out, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)

            # Return loss function
            return l1 + 6*l2 + 0.05*l3 + 1*(l4+l5) + 0.1*l6

        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)
    
    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)
    
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
        
    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss
    
    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b

    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )
        
    def fit(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        history = self.model.fit(generator,
            *args, **kwargs)
        return history
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model, inputs_mask = self.build_pconv_unet(train_bn)
#         self.compile_pconv_unet(self.model, inputs_mask, lr) 
        self.compile_pconv_unet(self.model, inputs_mask, lr)  

        # Load weights into model
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)
        
        print("pretrained model loaded")
    
    def save_weight(self, filepath):
        self.model.save_weights(filepath)

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
    
    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""
        
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram
    
    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
