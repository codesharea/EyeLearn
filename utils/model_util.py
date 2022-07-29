from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import statistics
import random

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9


def batch_norm(x, center=True):
    return BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=True,
        gamma_initializer='ones')(x)

def linear_layer(x, units, use_bias=True, use_bn=False):
    x = Dense(units,
              use_bias=use_bias and not use_bn,
              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.01))(x)
    if use_bn:
        x = batch_norm(x, center=use_bias)
    return x

def projection_head(inputs, ph_layers, return_all=False):
    x = inputs
    outputs = []
    for i, layer_dim in enumerate(ph_layers):
        if i != len(ph_layers) - 1:
            # for the middle layers, use bias and relu for the output.
            dim, bias_relu = layer_dim, True
        else:
            # for the final layer, neither bias nor relu is used.
            dim, bias_relu = layer_dim, False
        x = linear_layer(x, dim, use_bias=bias_relu, use_bn=True)
#         x = ReLU(name=f'proj-head-{i}')(x) if bias_relu else x
        outputs.append(x)
    if return_all:
        return outputs
    return x


def perform_kmeans(feats, num_cluster=8):
    model = KMeans()
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#     print(feats)
#     feats = imp.fit_transform(feats)
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=20000, random_state=0)
    kmeans.fit(feats)
    
#     (unique, counts) = np.unique(kmeans.labels_, return_counts=True)
#     frequencies = np.asarray((unique, counts)).T
#     print(frequencies)
    
    return kmeans.labels_, kmeans.cluster_centers_
    
def perform_gmm(feats, cluster_num=8):
    gmm = GaussianMixture(n_components=cluster_num)
    gmm.fit(feats)
    
    return gmm.labels_, gmm.cluster_centers_

def perform_evaluation(embeds, mds, labs, num_run = 40, test_num=400, train_ratio = 1.0):
    MAEs = []
    R2s = []
    Accs = []
    AUCs = []
    F1s = []
    test_ids = random.sample(range(len(labs)), test_num)
    for k in range(num_run):

        train_ids = random.sample(list(set(range(len(labs))) - set(test_ids)), int((len(labs)-test_num)*train_ratio))

        # VF Prediction
        train_x, test_x, train_y, test_y = embeds[train_ids], embeds[test_ids], mds[train_ids], mds[test_ids]
        regressor = Ridge()
        hisroty = regressor.fit(train_x, train_y)
        y_pred = regressor.predict(test_x)
        MAE = mean_absolute_error(test_y, y_pred)
        R2 = r2_score(test_y, y_pred)
        MAEs.append(MAE)
        R2s.append(R2)
    
        # Glaucomatous Classification
        train_y_lab, test_y_lab = labs[train_ids], labs[test_ids]
        classifier = LinearSVC(max_iter=20000, dual=False)
        hisroty = classifier.fit(train_x, train_y_lab)
        y_pred = classifier.predict(test_x)
        Acc = accuracy_score(test_y_lab, y_pred)
        F1 = f1_score(test_y_lab, y_pred)
        fpr, tpr, thresholds = roc_curve(test_y_lab, y_pred)
        AUC = auc(fpr, tpr)
        Accs.append(Acc)
        AUCs.append(AUC)
        F1s.append(F1)
    
    MAE_mean, MAE_std = statistics.mean(MAEs), statistics.stdev(MAEs)
    R2_mean, R2_std = statistics.mean(R2s), statistics.stdev(R2s)
    Acc_mean, Acc_std = statistics.mean(Accs), statistics.stdev(Accs)
    F1_mean, F1_std = statistics.mean(F1s), statistics.stdev(F1s)
    print('MAE_mean, MAE_std: ', MAE_mean, MAE_std)
    print('R2_mean, R2_std:', R2_mean, R2_std, '\n')
    print('Acc_mean, Acc_std: ', Acc_mean, Acc_std)
    print('F1_mean, F1_std: ', F1_mean, F1_std)
    
    return [MAE_mean, R2_mean, Acc_mean, F1_mean], [MAE_std, R2_std, Acc_std, F1_std]

