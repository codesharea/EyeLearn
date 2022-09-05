import tensorflow as tf
import argparse
from models import rnflt2vec
from utils import data_process
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
# import tensorflow_addons as tfa
# from keras_tqdm import TQDMNotebookCallback
import datetime
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from utils.map_handler import *
import cv2
import random
from tensorflow.keras.models import Model
from copy import deepcopy
from utils import model_util
import numpy as np
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(description='Tensorflow EyeLearn Training')

parser.add_argument('--model-name', default='EyeLearn')
parser.add_argument('--img-rows', type=int, default=256)
parser.add_argument('--img-cols', type=int, default=256)
parser.add_argument('--projection-dim', type=int, default=128, help='Dimension at the output of the projection head')
parser.add_argument('--embed-dim', type=int, default=512)
parser.add_argument('--projection-layers', type=int, default=3, help='Number of projection layers')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--cluster-model', default='kmeans', help='either kmeans or gmm')
parser.add_argument('--data-path', help='path to the original dataset')
parser.add_argument('--pretrained', help='path to the pretrained model')
parser.add_argument('--gpus', default=1, help="number of gpus used")
parser.add_argument('-vgg-weights', default='imagenet', help='pretraind vgg model weights')
parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
parser.add_argument('--reconstruct_w', default=1, type=float, help='reconstruct_w')
parser.add_argument('--activation', default='relu')
parser.add_argument('--num-clusters', default=7, type=int, help='number of clusters (default: 8)')
parser.add_argument('--bank_size', default=800, type=int, help='memory bank size')
parser.add_argument('--num-negatives', default=16, type=int, help='number of negative contrastive samples')
parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size (default: 4)')
parser.add_argument('--intracon_w', default=0.002, type=float, help='intracon_w')
parser.add_argument('--intercon_w', default=0.001, type=float, help='intercon_w')
parser.add_argument('--momentume', default=0.8, type=float, help='momentum for embedding update')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay for SGD')

parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('-f')
args = parser.parse_args()

FOLDER = 'checkpoint/'
args.data_path = 'dataset/'
args.vgg_weights = 'pytorch_to_keras_vgg16.h5'
args.local_data_path = 'dataset/'
args.pretrained = False

print(args.model_name, args.num_clusters, args.bank_size, args.num_negatives, args.batch_size, args.intracon_w, args.intercon_w, args.momentum)

if args.pretrained:
    model = rnflt2vec.construct_model_from_args(args)
    model.load(
        r"checkpoint/",
        train_bn=False,
        lr=0.00005
    )
#     model = tf.keras.models.load_model(args.pretrained, compile=False)
else:
    model = rnflt2vec.construct_model_from_args(args)
    
datagen, train_maps, val_maps = data_process.load_dataset(args.data_path, 
                                                                     args.local_data_path,
                                                                     num_train=20000, num_val=2000
                                                                    )

train_masks = np.ones_like(train_maps, dtype=np.uint8)

print("load embed model")
project_model = model.model.get_layer('embed_model')
# obtain representations 
embeddings = project_model.predict([train_maps, train_masks])
# perform clustering 
print(embeddings.shape, "perform k-means")
train_labs, centers = model_util.perform_kmeans(feats=embeddings, num_cluster=args.num_clusters)

train_size = len(train_labs)

# initialize the memory bank
idx = random.sample(list(range(args.batch_size, train_size)), args.bank_size)
bank_labs = train_labs[idx]

(unique, counts) = np.unique(bank_labs, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)


# model.set_memory(memory_feats=embeddings[idx], memory_labs=train_labs[idx], clu_centers=centers)
model.set_memory(memory_feats=embeddings[idx], clu_centers=centers)

# evaluations = []
val_generator = datagen.flow_val(val_maps, batch_size=args.batch_size)
(masked_val, mask_val), ori_val = next(val_generator)
evaluations = []
# (masked, mask, masked_cc, mask_cc, id_pairs), ori1 = next(test_generator)
def update_memory_bank(model, embeddings):
    """perform k-means clustering and update the memory bank of representations"""
    project_model = model.model.get_layer('embed_model')
#     decoder_direct = Model(inputs=embed_model.inputs, outputs=embed_model.get_layer('encoder_output').output)
    # obtain representations 
    embeddings = project_model.predict([train_maps/255, train_masks])
#     embeddings = (1 - args.momentume) * embeddings + args.momentume * project_model.predict([train_maps, train_masks])
    # perform clustering 
    labels, centers = model_util.perform_kmeans(feats=embeddings, num_cluster=args.num_clusters)
    # update the memory bank
    idx = random.sample(list(range(args.batch_size, train_size)), args.bank_size)
    bank_labs = train_labs[idx]
    model.set_memory(memory_feats=embeddings[idx], clu_centers=centers)
    
    # Get samples & Display them, outputs, projection_out, imputed_out
    predict_model = Model(inputs=[model.model.inputs[0], model.model.inputs[1]], 
                                 outputs=model.model.outputs[0])

    impute_img = predict_model.predict([masked_val, mask_val])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    for i in range(masked_val.shape[0]):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked_val[i,:,:,:])
        axes[1].imshow(impute_img[i,:,:,:] * 1.)
        axes[2].imshow(ori_val[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')

        plt.savefig(r'/imgs/{}_img_{}.png'.format(args.model_name, i))
        plt.close()

    return labels, bank_labs, embeddings


train_map_ids = np.array(list(range(train_size)))
train_generator = datagen.flow(x=train_maps, y=train_map_ids, batch_size=args.batch_size)


best_loss = float('inf')
step_per_epoch = int(train_size/args.batch_size)
ptr = 0
epoch_log = []
for epoch in range(args.epochs):
#     batches = 0
    batch_log = []
    for k in range(step_per_epoch):
#     for x_batch, y_batch in train_generator:
        x_batch, y_batch = next(train_generator)
        ori, idx_ori = y_batch
        batch_labs = train_labs[idx_ori]
        intra_indices, inter_indices = datagen.get_contrastive_index(batch_labs, 
                                                                     bank_labs, 
                                                                     args.num_negatives)
        # update the labels in the memory bank
        temp = ptr + args.batch_size
        if temp > args.bank_size:
            temp = temp % args.bank_size
            bank_labs[ptr:] = batch_labs[:-temp]
            bank_labs[:temp] = batch_labs[-temp:]
            ptr = temp
#         assert self.bank_size % self.batch_size == 0
        else:
            bank_labs[ptr:temp] = batch_labs
            ptr = temp % args.bank_size  # move pointer
        
        y_batch = (ori, intra_indices, inter_indices)
        history = model.fit(x_batch, y_batch, batch_size=args.batch_size, verbose=0)
        
        batch_log.append([history.history['loss'][0], history.history['reconstruct_mse'][0]])

    if train_size % args.batch_size != 0:
        next(train_generator)
            
    epoch_log.append(np.mean(batch_log, 0))
    
    # evaluate the trained model
    predict_model = Model(inputs=[model.model.inputs[0], model.model.inputs[1]], 
                                 outputs=model.model.outputs[0])
    predict_model.compile(loss='mse', optimizer='adam')
    eva_loss = predict_model.evaluate(val_generator, steps=int(len(val_maps)/args.batch_size))

    print("Epoch: {}/{}, loss:{:.4f}, reconstruct_mse:{:.4f}".format(epoch+1, args.epochs, epoch_log[-1][0], epoch_log[-1][1]))
    
    # update clutsering labels after every epoch
    train_labs, bank_labs, embeddings = update_memory_bank(model, embeddings)
    
    # save the best model for evaluation
    if best_loss > eva_loss:
        best_loss = eva_loss
        weight_path = FOLDER+'{}_weights.{epoch:02d}-{loss:.4f}.h5'.format(args.model_name, epoch=epoch, 
                                                                                     loss=best_loss)
        model.save_weight(weight_path)
        
        (unique, counts) = np.unique(bank_labs, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)
    
    print('Current loss / global loss: {loss:.4f} / {best_loss:.4f}'.format(loss=eva_loss, best_loss=best_loss))
