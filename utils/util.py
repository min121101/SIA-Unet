import numpy as np
import torch
import os
import random
import cv2
from torch.optim import lr_scheduler


def update_config(config, args):
    # Enable the args to overlay yaml configuration
    args_list = list(vars(args).keys())
    if "exp_name" in args_list:
        config['metainfo']['exp_name'] = get_value(config['metainfo']['exp_name'], args.exp_name)
    return config


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


def id2mask(id_):
    idf = df[df['id'] == id_]
    wh = idf[['height', 'width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class'] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0),(0, 0),(1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def load_img(path, CFG):
    if not CFG['2.5D']:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    else:
        img = np.load(path)
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def fetch_scheduler(optimizer, CFG):
    if CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['T_max'],
                                                   eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'],
                                                             eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG['min_lr'], )
    elif CFG['scheduler'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG['scheduler'] == None:
        return None

    return scheduler