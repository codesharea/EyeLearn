import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

def gen_cmap(N=10000):
    
    return matplotlib.colors.LinearSegmentedColormap.from_list("", [(0,'black'), (0.06,'blue'), 
                                                                  (0.23, '#2ab6c6'), (0.38,'yellow'), 
                                                                  (0.6,'red'), (1,'white')], N=N)   

def obtain_cup_and_mask(img, cm=None):
    if cm == None:
        cm = gen_cmap(256)
    diskcup = to_3dmap(img, cm)
    mask = np.ones_like(diskcup)
    red, green, blue= diskcup.T
    white_areas = (red == 0) & (blue == 0) & (green == 0)
    mask[[white_areas.T]]=(0,0,0)
    mask[img<0] = 1.
    diskcup[img>=0]=1.
    diskcup[img<0]=0.

    return diskcup, mask

def to_2dmap(img, colrange=[0,1], N=10000):
    cm = gen_cmap(N)
    img = np.array(img) 
    scalars = []
    row, col, _ = img.shape
    r = np.linspace(colrange[0], colrange[1], N)
    for i in range(row):
        for j in range(col):
            color = img[i,j,:3]
            norm = matplotlib.colors.Normalize(colrange[0], colrange[1])
            mapvals = cm(norm(r))[:, :3] # there are 4 channels: r,g,b,a
            distance = np.sum((mapvals - color) ** 2, axis=1)
            scalars.append(r[np.argmin(distance)]*350)
    scalars = np.reshape(np.array(scalars), (row, col))
    
    return scalars

def to_3dmap(img, cm=None, N=10000):
    if cm == None:
        cm = gen_cmap(N)
    img[img>350] = 350.
    img = cm((img/350.))
#     colored_image = (img[:, :, :3])
    colored_image = (img[:, :, :3] * 255).astype(np.uint8)
    
    return colored_image

def plot_2dmap(img, show_colorbar=True, show_cup=False, with_ref=False, ref_img=None, delartifact=False, cm=None, title_on=True):
    if cm == None:
        cm = gen_cmap(256)
        
    img_copy2d = deepcopy(img)
    # mark the rim and cup regions by -1 and -2 locations
    if show_cup:
        img_copy2d[img_copy2d==-1] = np.nan
        cm.set_bad("gray")
        cm.set_under(color='lightgray')
    # mark rim and cup regions from reference img
    if with_ref:
        img_copy2d[ref_img==-2] = -2
        img_copy2d[ref_img==-1] = np.nan
        cm.set_bad(color="gray")
        cm.set_under(color='lightgray')
    # delete artifact locations defined by <=30 and >=0
    if delartifact:
        img_copy2d[(img_copy2d<=30) & (img_copy2d>0)] = 0
    fig = plt.figure()
    ax = plt.subplot(111)
    img_copy2d = ax.imshow(img_copy2d, cmap=cm, vmin=-0.00000001, vmax=350)
    if show_colorbar:
        cbar = plt.colorbar(img_copy2d, pad=0.01, aspect=12, location='left')
        cbar.set_ticks([0, 175,350])
        cbar.ax.set_yticklabels(['0 μm', '175 μm', '350 μm'])
        cbar.ax.tick_params(labelsize=14)
        if title_on:
            ax.set_title('RNFL Thickness Map', fontsize=15)
        ax.axis('off')
    else:
        ax.axis('off')
    plt.show()
    
def plot_3dmap(img, show_colorbar=True, cm=None, title_on=True):
    if cm == None:
        cm = gen_cmap(256)
    img_copy3d = deepcopy(img)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    img_copy3d = ax.imshow(img_copy3d*1., cmap=cm, vmin=0, vmax=1)
    if show_colorbar:
        cbar = plt.colorbar(img_copy3d, pad=0.01, aspect=12, location='left')
        cbar.set_ticks([0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1'])
        cbar.ax.tick_params(labelsize=14) 
        if title_on:
            ax.set_title('3D RNFL Thickness Map', fontsize=15)
        ax.axis('off')
    plt.show()
    
def cal_MAE(ori_map, inv_map):
    img_copy2d_1 = deepcopy(ori_map)
    img_copy2d_2 = deepcopy(inv_map)
    img_copy2d_1[img_copy2d_1<0]=0
    img_copy2d_2[img_copy2d_2<0]=0
    
    return np.mean(abs(img_copy2d_1-img_copy2d_2))
