import numpy as np
import cv2
import matplotlib.pyplot as plt

def channel(img, ch):
    return img[:, :, ch]

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb2lab(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2hls(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2hsv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2yuv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]
    
    
def rgb2YCrCb(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def plot(img, ch=-1, size=(5, 3), cmap=None, name=None, title=None, fontsize=14):
    plt.figure(figsize=size)
    if ch < 0:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img[:, :, ch], cmap=cmap)
    if title:
            plt.set_title(title, fontsize=fontsize)
    if name:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    

def plot_many(imgs, cmaps=[], name=None, title=None, fontsize=14):
    n = len(imgs)
    f, ax = plt.subplots(1, n, figsize=(n*10,5))
    for i in range(n):
        if not cmaps:
            ax[i].imshow(imgs[i])
        else:
            ax[i].imshow(imgs[i], cmap = cmaps[i])
        if title:
            ax[i].set_title(title[i], fontsize=fontsize)
    
    f.tight_layout()
    if name:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    
def white_select(img): 
    lower = np.array([0,210,0], dtype=np.uint8)
    upper = np.array([360,255,255], dtype=np.uint8)
    
    channel_h = img[:, :, 0]
    channel_l = img[:, :, 1]
    channel_s = img[:, :, 2]
    
    binary_output = np.zeros_like(img[:,:,0])
    binary_output[((channel_h > lower[0]) & (channel_h <= upper[0])) 
                  & ((channel_l > lower[1]) & (channel_l <= upper[1])) 
                  & ((channel_s > lower[2]) & (channel_s <= upper[2]))] = 1
    
    return binary_output

def channel_thresholding(img, ch, thresh=(0, 255)):
    channel = img[:,:,ch]
    binary_output = np.zeros_like(channel)
    binary_output[(channel>thresh[0]) & (channel<=thresh[1])] = 1

    return binary_output