from models.PConvAE import PConvUnet
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
import tensorflow as tf

def construct_model_from_args(args):
    
    rnfl2vec_model = PConvUnet(img_rows = args.img_rows,
                               img_cols = args.img_cols,
                               reconstruct_w=args.reconstruct_w, 
                               intracon_w=args.intracon_w, 
                               intercon_w=args.intercon_w,
                               batch_size = args.batch_size,
                               num_clusters = args.num_clusters,
                               num_negatives = args.num_negatives,
                               bank_size = args.bank_size,
                               learn_rate = args.lr,
                               gpus = args.gpus,
                               momentum = args.momentum,
                               weight_decay = args.weight_decay,
                               vgg_weights = args.vgg_weights,
                               projection_dim = args.projection_dim,
                               embed_dim = args.embed_dim,
                               projection_head_layers = args.projection_layers,
                               activation = args.activation)
    
    return rnfl2vec_model


