#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install pycocotools
# !pip install git+https://github.com/CellProfiling/HPA-Cell-Segmentation.git # This one install pytorch_zoo
__author__ = 'MPWARE: https://www.kaggle.com/mpware'


# In[ ]:


from hpa_training import *


# In[ ]:


import os, sys, random, gc, math
import numpy as np
import pandas as pd
import operator
import h5py
import cv2
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torch.nn.functional as F
import functools
from collections import OrderedDict, defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from ast import literal_eval
import timm
from PIL import Image
import skimage.io
import skimage.transform
import warnings
import argparse
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 4000)
import matplotlib.pyplot as plt


# In[ ]:


import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib

# Vanilla HPA
import hpacellseg
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei

# Custom implementation
import scipy.ndimage as ndi
from skimage.morphology import (closing, disk, remove_small_holes, remove_small_objects)
from skimage import filters, measure, segmentation
from skimage import transform, util


# In[ ]:


PREDICTION_STRING = "PredictionString"
CELL_CONF = "cell_conf"

# Enable segmentation optimization
ON_FLY_MASK = True
# Enable to high speed inference
FAST_INSTANCES = True # False
SEG_SCALE_FACTOR = 0.25 # Not used for fast instances
CENTER_PAD = 384


# In[ ]:


def prepare_data(filename, meta_pd=None, remove_ids=None):
    train_pd = pd.read_csv(filename)
    
    train_pd[LABEL] = train_pd[LABEL].apply(literal_eval)
    train_pd[LABEL] = train_pd[LABEL].apply(lambda x: [int(l) for l in x])    
    if EXT not in train_pd.columns:
        train_pd.insert(2, EXT, DEFAULT)
    train_pd = train_pd.drop_duplicates(subset=[ID]).reset_index(drop=True)
    assert(np.argwhere(train_pd.columns.values == EXT)[0][0] == 2)
    
    le_ = LabelEncoder()
    le_.fit(train_pd[ID])
    train_pd[EID] = le_.transform(train_pd[ID])
    
    if meta_pd is not None:
        train_pd = pd.merge(train_pd, meta_pd[[ID, IMAGE_HEIGHT, IMAGE_WIDTH]], on=[ID], how="left")
    else:        
        raise Exception("Meta-data file required")
        
    if remove_ids is not None:
        print("Dropping:", len(remove_ids))
        train_pd = train_pd[~train_pd[ID].isin(remove_ids)].reset_index(drop=True)

    return train_pd, le_


# In[ ]:


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    if True:
        x1, x2, y1, y2 = bbox2_int(mask)
        return ( int(np.round((x1+x2)/2.0)), int(np.round((y1+y2)/2.0)) )
        
    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION) # zlib.Z_BEST_SPEED
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


# In[ ]:


def bbox2_int(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x1 = cmin
    x2 = (cmax + 1)
    y1 = rmin
    y2 = (rmax + 1)

    return x1, x2, y1, y2

# 0-1.0 normalized image (H, W, 4)
def generate_masks(img, nuclei_only=False, verbose=False):
    
    nuclei_mask, cell_mask = None, None
    
    # Nuclei segmentation
    nuc_segmentations = segmentator.pred_nuclei([img[:,:,2]])
    print("Nuclei", len(nuc_segmentations), nuc_segmentations[0].shape, nuc_segmentations[0].dtype, nuc_segmentations[0].max(), np.unique(nuc_segmentations[0])) if verbose is True else None
    
    if nuclei_only is False:
        img_ryb = np.dstack([img[:,:,0], img[:,:,3], img[:,:,2]])
        print("Image", img_ryb.shape, img_ryb.dtype, img_ryb.max()) if verbose is True else None
        # Cell segmentation: Requires list of RYB when multi_channel_model is True, it will apply scale factor
        cell_segmentations = segmentator.pred_cells([img_ryb], precombined=True)
        print("Cells", len(cell_segmentations), cell_segmentations[0].shape, cell_segmentations[0].dtype, cell_segmentations[0].max(), np.unique(cell_segmentations[0])) if verbose is True else None
        # Extract instances
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
        print(cell_mask.shape, cell_mask.dtype, cell_mask.max()) if verbose is True else None # (H, W) uint16
    else:
        # Extract instances
        nuclei_mask = label_nuclei(nuc_segmentations[0])
        
    return nuclei_mask, cell_mask


# In[ ]:


def denormalize_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if std is not None:
        std = np.array(std)
        x = x * std
        
    if mean is not None:
        mean = np.array(mean)
        x = x + mean
        
    if input_range is not None:
        if input_range[1] == 1:
            x = x * 255.0
    
    return x

def get_preprocessing_fn(cfg):
    params = {"mean": cfg.IMG_MEAN, "std": cfg.IMG_STD, "input_range": cfg.INPUT_RANGE} if ON_FLY_MASK is False else {"mean": None, "std": None, "input_range": cfg.INPUT_RANGE}
    return functools.partial(preprocess_input, **params)

def get_denormalize_fn(cfg):
    params = {"mean": cfg.IMG_MEAN, "std": cfg.IMG_STD, "input_range": cfg.INPUT_RANGE}  if ON_FLY_MASK is False else {"mean": None, "std": None, "input_range": cfg.INPUT_RANGE}
    return functools.partial(denormalize_input, **params)


# In[ ]:


class MeanStdNormalize(torch.nn.Module):

    def __init__(self, cfg, mean, std, verbose=False):
        super(MeanStdNormalize, self).__init__()
        self.mean = torch.as_tensor(mean, device=cfg.L_DEVICE)
        self.std = torch.as_tensor(std, device=cfg.L_DEVICE)
        print(self.mean, self.mean[:, None, None].shape) if verbose is not None else None

    def forward(self, data):
        # (BS, CHANNELS, H, W) with [0-1] range
        data_ = data.clone()
        data_ = data_.sub_(self.mean[:, None, None]).div_(self.std[:, None, None]) # mean=torch.Size([CHANNELS, 1, 1])
        return data_


class CellSegmentation(nn.Module):
    def __init__(self, cfg, cell_model_path=None, nu_model_path=None, verbose=False):
        super().__init__()
        
        # Normalize applies on (3, H, W) with [0-1] range
        NORMALIZE = {"mean": [124 / 255, 117 / 255, 104 / 255], "std": [1 / (0.0167 * 255)] * 3}        
        self.normalize = MeanStdNormalize(cfg, NORMALIZE["mean"], NORMALIZE["std"], verbose=verbose)
        
        self.nuclei_model = torch.load(nu_model_path, map_location=conf.map_location)
        self.nuclei_model.eval()
        print("nuclei_model loaded from %s" % nu_model_path)
                
        self.cell_model = torch.load(cell_model_path, map_location=conf.map_location)
        self.cell_model.eval()
        print("cell_model loaded from %s" % cell_model_path)
    
        self.verbose = verbose

    def forward(self, data):
        # (BS, 4, H, W) # 0-1 normalized, RGBY
        batch_size, channels, height, width = data.shape
        print("data", data.shape) if self.verbose is True else None
        
        # Prepare for nucleui segmentation
        # Build (BS, 3 H, W) with B channel only
        data_blue = data[:, [2], :, :].expand(batch_size, 3, height, width).clone() # Size([BS, 3, H, W])
        print("data_blue", data_blue.shape) if self.verbose is True else None
        
        # Normalize
        data_blue = self.normalize(data_blue)
        # data_blue = data_blue.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        print("data_blue norm", data_blue.shape) if self.verbose is True else None
        
        # Predict nuclei mask
        x_nuclei = self.nuclei_model(data_blue)
        x_nuclei = F.softmax(x_nuclei, dim=1)
        print("x_nuclei", x_nuclei.shape, x_nuclei.dtype, x_nuclei.min(), x_nuclei.max()) if self.verbose is True else None
        
        # Prepare for cells segmentation
        # Build (BS, 3, H, W) with RYB channels only
        data_ryb = data[:, [0,3,2], :, :]
        # Normalize
        data_ryb = self.normalize(data_ryb)
        # data_ryb = data_ryb.sub_(self.mean[:, None, None]).div_(self.std[:, None, None]) # mean=torch.Size([3, 1, 1]) 
        # Predict cells mask
        x_cells = self.cell_model(data_ryb)
        x_cells = F.softmax(x_cells, dim=1)
        print("x_cells", x_cells.shape, x_cells.dtype, x_cells.min(), x_cells.max()) if self.verbose is True else None

        return x_nuclei, x_cells
    
def __resize_img(img_, orig_width, orig_height):
    if img_.shape[1] != orig_height:
        # img_ = cv2.cvtColor(cv2.cvtColor(img_, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        img_ = cv2.resize(img_, (orig_width, orig_height), interpolation=cv2.INTER_AREA)
    return img_

def __resize_masks(mask, new_width, new_height):
    cell_ids = [c for c in np.unique(mask) if c > 0]
    resized_mask = np.zeros((new_height, new_width), dtype=np.uint8) # Might be better to use np.uint16 on inference
    for cell_id in cell_ids:
        cell_mask_bool = mask.squeeze() == cell_id
        mask_ = cv2.resize(cell_mask_bool.astype(np.uint8), (new_height, new_width), interpolation = cv2.INTER_AREA) if cell_mask_bool is not None else None
        mask_ = np.clip(mask_, 0, 1).astype(np.uint8) * cell_id
        resized_mask = resized_mask + mask_
    mask = resized_mask
    return mask

# Post process to extract instances
def __fill_holes(image):
    """Fill_holes for labelled image, with a unique number."""
    boundaries = segmentation.find_boundaries(image)
    image = np.multiply(image, np.invert(boundaries))
    image = ndi.binary_fill_holes(image > 0)
    image = ndi.label(image)[0]
    return image

# Updated to support different scale factors
def label_cell_(nuclei_pred, cell_pred, scale_factor=0.25):
    """Label the cells and the nuclei.

    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.
    cell_pred -- a 3D numpy array of a prediction from a cell image.

    Returns:
    A tuple containing:
    nuclei-label -- A nuclei mask data array.
    cell-label  -- A cell mask data array.

    0's in the data arrays indicate background while a continous
    strech of a specific number indicates the area for a specific
    cell.
    The same value in cell mask and nuclei mask refers to the identical cell.

    NOTE: The nuclei labeling from this function will be sligthly
    different from the values in :func:`label_nuclei` as this version
    will use information from the cell-predictions to make better
    estimates.
    """
    rounder = np.round # np.ceil
    
    def __wsh(
        mask_img,
        threshold,
        border_img,
        seeds,
        threshold_adjustment=0.35,
        small_object_size_cutoff=10,
    ):
        img_copy = np.copy(mask_img)
        m = seeds * border_img  # * dt
        img_copy[m <= threshold + threshold_adjustment] = 0
        img_copy[m > threshold + threshold_adjustment] = 1
        img_copy = img_copy.astype(np.bool)
        
        img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(np.uint8) # 

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        
        mask_img = remove_small_holes(mask_img, max(1, int(rounder(1000*scale_factor))) )
        mask_img = remove_small_objects(mask_img, max(1, int(rounder(8*scale_factor))) ).astype(np.uint8)
        
        markers = ndi.label(img_copy, output=np.uint32)[0]
        labeled_array = segmentation.watershed(mask_img, markers, mask=mask_img, watershed_line=True)
        return labeled_array

    nuclei_label = __wsh(
        nuclei_pred[..., 2] / 255.0,
        0.4,
        1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
        nuclei_pred[..., 2] / 255,
        threshold_adjustment=-0.25,
        small_object_size_cutoff=max(1, int(rounder(500*scale_factor))), # 
    )

    # for hpa_image, to remove the small pseduo nuclei
    nuclei_label = remove_small_objects(nuclei_label, max(1, int(rounder(2500*scale_factor))))
    nuclei_label = measure.label(nuclei_label)
    # this is to remove the cell borders' signal from cell mask.
    # could use np.logical_and with some revision, to replace this func.
    
    # Tuned for segmentation hpa images
    threshold_value = max(0.22, filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
    # exclude the green area first
    cell_region = np.multiply(
        cell_pred[..., 2] / 255 > threshold_value,
        np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
    )
    sk = np.asarray(cell_region, dtype=np.int8)
    distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[..., 2])
    cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)
    
    cell_label = remove_small_objects(cell_label, max(1, int(rounder(5500*scale_factor))) ).astype(np.uint8) # 
    
    selem = disk(max(1, int(rounder(6*np.sqrt(scale_factor) ))) )
    
    cell_label = closing(cell_label, selem)
    cell_label = __fill_holes(cell_label)
    # this part is to use green channel, and extend cell label to green channel
    # benefit is to exclude cells clear on border but without nucleus
    sk = np.asarray(
        np.add(
            np.asarray(cell_label > 0, dtype=np.int8),
            np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
        )
        > 0,
        dtype=np.int8,
    )
    cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
    cell_label = __fill_holes(cell_label)
    cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
    cell_label = measure.label(cell_label)
    
    cell_label = remove_small_objects(cell_label, max(1, int(rounder(5500*scale_factor))) ) # 
    
    cell_label = measure.label(cell_label)
    cell_label = np.asarray(cell_label, dtype=np.uint16)
    nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
    nuclei_label = measure.label(nuclei_label)
    
    nuclei_label = remove_small_objects(nuclei_label, max(1, int(rounder(2500*scale_factor))) ) # 
    
    nuclei_label = np.multiply(cell_label, nuclei_label > 0)

    return nuclei_label, cell_label    


# In[ ]:


# (*, C, H, W)
def ident(data):
    return data

# (*, C, H, W)
def rot180(data):
    return torch.flip(data, [-2, -1])

# (*, C, H, W)
def hflip(data):
    w = data.shape[-1]
    return data[..., torch.arange(w - 1, -1, -1, device=data.device)]

# (*, C, H, W)
def vflip(data):
    h = data.shape[-2]
    return data[..., torch.arange(h - 1, -1, -1, device=data.device), :]

# Idempotent TTA
class EnsembleTTA(nn.Module):
    def __init__(self, model, ttas=None, verbose=True):
        super(EnsembleTTA, self).__init__()
        self.model = model
        self.conf = self.model.conf
        self.ttas = ttas
        self.preprocess_input_fn = self.model.preprocess_input_fn
        if verbose is True:
            print("EnsembleTTA:", self.ttas, self.preprocess_input_fn)
    
    def eval(self):
        self.model.eval()
    
    def forward(self, data):
        if self.ttas is None or len(self.ttas) == 0:
            return self.model(data)
        else:            
            output = {}
            # Concatenate model outputs
            output_cat = defaultdict(list)
            for tta in self.ttas:
                output_dict = self.model(tta(data))
                for k, v in output_dict.items():
                    output_ = output_cat[k]
                    output_.extend([tta(v) if "features" in k else v])
            # Average all outputs
            for k, v in output_cat.items():
                output[k] = torch.mean(torch.stack(v), dim=0)
            return output


# In[ ]:


class Ensemble(nn.Module):
    def __init__(self, models, conf, preprocess_input_fn, verbose=True):
        super(Ensemble, self).__init__()
        self.models = models
        self.conf = conf
        self.preprocess_input_fn = preprocess_input_fn
        if verbose is True:
            print("Ensemble of %d %s_%s model(s)" % (len(self.models), self.conf.mtype, self.conf.backbone))
    
    def eval(self):
        for i, m in enumerate(self.models):
            m.eval()
    
    def forward(self, data):        
        output = {}
        # Concatenate model outputs
        output_cat = defaultdict(list)
        for m in self.models:
            output_dict = m(data)
            for k, v in output_dict.items():
                output_ = output_cat[k]
                output_.extend([v])
        # Average all outputs
        for k, v in output_cat.items():
            output[k] = torch.mean(torch.stack(v), dim=0)
        return output
    
def build_ensemble(name, cfg, model_dict):
    tmp_models_ = []
    for k, v in model_dict.items():
        # Load each model
        if "fold" in k:
            model_path = model_dict.get(k)
            if os.path.exists(model_path):
                model_, _, _ = build_model(cfg, cfg.L_DEVICE)
                model_.load_state_dict(torch.load(model_path, map_location=cfg.map_location))
                model_.preprocess_input_fn = get_preprocessing_fn(cfg) # Override preprocessing_fn for fast instances
                print("Loading %s: %s" % (name, model_path))
                model_.eval()
                tmp_models_.append(model_)
    ensemble_ = Ensemble(tmp_models_, cfg, tmp_models_[-1].preprocess_input_fn)
    return ensemble_


# In[ ]:


# Test loop
def test_loop_fn(batches, preprocessing, normalization, model, tmp_conf, device, stage="Test", verbose=True):    
    model.eval()
    predictions = []
    
    with tqdm(batches, desc=stage, file=sys.stdout, disable=not(verbose)) as iterator:
        for batch in iterator:
            try:                
                for k, v in batch.items():
                    if k in ["image"]:
                        batch[k] = v.to(device)
                
                samples_data = batch.get("image") # GPU [BS, 4, H, W] [0-1.0]
                metas_data = batch.get(META).numpy()              

                with torch.no_grad():
                    # Preprocessing (Cells segmentation)
                    masks_nu_data, masks_data = preprocessing(samples_data) if preprocessing is not None else (None, None)
                    
                    # Feed another model
                    # Normalization (Std/Mean if not done before)
                    data = normalization(samples_data) if normalization is not None else samples_data
                    
                    # NN model
                    with torch.cuda.amp.autocast(enabled=tmp_conf.fp16):
                        output = model(data) # forward pass
                        if tmp_conf.mtype == "siamese":                            
                            predicted_extras = output[tmp_conf.output_key_extra] if tmp_conf.output_key_extra is not None else None                                                    
                            output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output
                        else:                            
                            predicted_extras = output[tmp_conf.output_key_extra] if tmp_conf.output_key_extra is not None else None                            
                            output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output

                    # Labels predictions
                    predicted_probs = torch.sigmoid(output) if tmp_conf.post_activation == "sigmoid" else output
                    
                if preprocessing is not None:
                    # Finalize nuclei, cells segmentation
                    masks_nu_data = masks_nu_data.cpu().numpy().transpose([0, 2, 3, 1])
                    masks_data = masks_data.cpu().numpy().transpose([0, 2, 3, 1]) # (BS, H, W, 3), float32 [0-1]
                    # Background as zero
                    masks_nu_data[..., 0] = 0
                    masks_data[..., 0] = 0
                    # Move to np.uint8 [0-255] range
                    nuc_segmentations = list(map(util.img_as_ubyte, masks_nu_data))
                    cell_segmentations = list(map(util.img_as_ubyte, masks_data))
                    nuclei_masks, cell_masks = [], []
                    for nuc_segmentation, cell_segmentation, meta_data in zip(nuc_segmentations, cell_segmentations, metas_data):
                        if FAST_INSTANCES is True:
                            # Auto scale because image size can be 4096, 2048, 3072, 1728
                            auto_scale_factor = tmp_conf.image_size/meta_data[1] # tmp_conf.seg_scale_factor
                            nuclei_mask_, cell_mask_ = label_cell_(nuc_segmentation, cell_segmentation, scale_factor=auto_scale_factor*auto_scale_factor) # tmp_conf.seg_scale_factor                                          
                        else:
                            # Resize to original image before applying morphology to extract instances
                            nuc_segmentation = __resize_img(nuc_segmentation.astype(np.uint8), meta_data[1], meta_data[2])
                            cell_segmentation = __resize_img(cell_segmentation.astype(np.uint8), meta_data[1], meta_data[2])
                            auto_scale_factor = 1.0
                            nuclei_mask_, cell_mask_ = label_cell_(nuc_segmentation, cell_segmentation, scale_factor=1.0)                 
                            # Resize masks for further intersect with CAM: To improve
                            nuclei_mask_ = __resize_masks(nuclei_mask_.astype(np.uint8), tmp_conf.image_size, tmp_conf.image_size)
                            cell_mask_ = __resize_masks(cell_mask_.astype(np.uint8), tmp_conf.image_size, tmp_conf.image_size)
                        nuclei_masks.append(nuclei_mask_)
                        cell_masks.append(cell_mask_)
                    nuclei_masks = np.expand_dims(np.array(nuclei_masks), axis=-1)
                    cell_masks = np.expand_dims(np.array(cell_masks), axis=-1)                     

                # Compute predictions from Pytorch model for this batch
                tmp_result_ = compute_predictions(predicted_probs, predicted_extras, cell_masks, nuclei_masks, metas_data)
                                                
                predictions.extend(tmp_result_)

            except Exception as ex:
                print("Test batch error:", ex)
    
    return predictions


# In[ ]:


def resize_and_rle_encode_one_cell(target_masks, cell_id, orig_width, orig_height):
    mask_bool = target_masks.squeeze() == cell_id
    mask_bool = cv2.resize(mask_bool.astype(np.uint8), (orig_width, orig_height), interpolation = cv2.INTER_AREA)
    mask_bool = np.clip(mask_bool, 0, 1).astype('bool')
    cell_rle = encode_binary_mask(mask_bool)
    return cell_rle


# In[ ]:


def find_centers_countours(ids, masks, bb=None):
    centers_ = {}
    for uid in ids:
        try:
            contours, hierarchy= cv2.findContours((masks.squeeze() == uid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            M = cv2.moments(contours[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if bb is not None:
                if not (bb[0] < cY < bb[0]+bb[2] and bb[1] < cX < bb[1]+bb[3]):
                    centers_[uid] = (cX, cY, contours[0])
            else:
                centers_[uid] = (cX, cY, contours[0])
        except Exception as e:
            pass
    return centers_

def get_border_cells_ids(target_masks, target_masks_nu):
    border_cells = {}
    if BORDER_SIZE_PERCENT > 0:
        cell_ids = [c for c in np.unique(target_masks) if c > 0]
        nu_ids = [c for c in np.unique(target_masks_nu) if c > 0]
        # Nuclei is used as seed so we should have the same total? Answer is NO
        cells_with_nu = np.intersect1d(np.array(cell_ids), np.array(nu_ids))
        # Find center of each nucluei (if available) outside a bounding box
        border_cells = find_centers_countours(cells_with_nu, target_masks_nu, bb=(int(np.round(target_masks_nu.shape[0]*BORDER_SIZE_PERCENT)), int(np.round(target_masks_nu.shape[1]*BORDER_SIZE_PERCENT)),
                                                                                  int(np.round(target_masks_nu.shape[0]*(1-2*BORDER_SIZE_PERCENT))), int(np.round(target_masks_nu.shape[1]*(1-2*BORDER_SIZE_PERCENT)))) )
    return border_cells


# In[ ]:


# preds: [(class_id, class_prob), ...] for selected classes above THR
# predicted_cams: CAMs generated from features model output (BS, CLASSES, N, N)
# target_masks: cells masks
# target_masks_nu: nuclei masks
# cell_ids: list of cells identifier to predict
# image: optional for debug
# Return dictionary per cell_id = { class_id: {"class_prob": ..., "cell_conf": ..., other meta data} ...}
def cells_prediction_puzzlecam(selected_class_preds, predicted_cams, target_masks, target_masks_nu, cell_ids, image=None):    
    cells_masks_meta = dict()
    empty_intersection = False
    
    # Only on confident predicted classes
    for pair in selected_class_preds:
        class_, class_prob_ = pair[0], pair[1]
        
        # Cam for the predicted class
        cam_class_ = predicted_cams[class_].astype(np.float32) # (Hc, Wc)
        
        # Interpolated cam to initial size (different interpolation to test as results are different especially on borders)        
        cam_class_ = skimage.transform.resize(cam_class_, (target_masks.shape[1], target_masks.shape[0]), anti_aliasing=True) # Works with float32 image
        
        # Quantile for full image
        total_class_cam_qt = np.quantile(cam_class_.flatten(), CAM_PRED_QUANTILE)
        
        # Find cells for the predicted class
        for cell_id in cell_ids:
            cell_mask_bool = target_masks.squeeze() == cell_id
            
            # Cam/mask intersection + quantile for cell only
            cell_class_cam_qt = np.quantile(cam_class_[cell_mask_bool], CAM_PRED_QUANTILE)
            cell_class_cam_max = cam_class_[cell_mask_bool].max()
            # Cell confidence
            cell_conf = np.clip(cell_class_cam_qt*class_prob_/total_class_cam_qt, 0, class_prob_)
            # Final score
            score = cell_conf/class_prob_
            if score > SCORE_TH:
                # We keep this cell                
                meta_coords_centroid = (int(np.where(cell_mask_bool)[1].mean()), int(np.where(cell_mask_bool)[0].mean()) ) # Cell centroid
                meta_coords_max = (np.where(cam_class_ == cell_class_cam_max)[1][0], np.where(cam_class_ == cell_class_cam_max)[0][0])
                cell_dict_ = {"class_prob": class_prob_, "total_class_cam_qt": total_class_cam_qt, "cell_class_cam_qt": cell_class_cam_qt, "cell_class_cam_max": cell_class_cam_max, CELL_CONF: cell_conf, "score": score, "meta_coords_max": meta_coords_max, "meta_coords_centroid": meta_coords_centroid}
                if cell_id not in cells_masks_meta:
                    cells_masks_meta[cell_id] = {class_: cell_dict_}
                else:
                     cells_masks_meta[cell_id][class_] = cell_dict_
            else:
                empty_intersection = True
        
        if image is not None:
            cam = cv2.applyColorMap((cam_class_*255).astype(np.uint8), cv2.COLORMAP_JET)
            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]        
        
    return cells_masks_meta, empty_intersection, image


# In[ ]:


def compute_predictions(all_predicted_probs, all_predicted_extras, all_target_masks, all_target_masks_nu, all_metas_data):
    
    results = []
    
    # Compute CAMs on GPU then move to CPU
    all_predicted_cams = make_cam(all_predicted_extras).cpu().numpy()
    
    # Move other Tensors to CPU
    all_target_masks = all_target_masks
    all_target_masks_nu = all_target_masks_nu
    all_metas_data = all_metas_data
    all_predicted_probs = all_predicted_probs.cpu().numpy()
    
    for predicted_probs, predicted_cams, target_masks, target_masks_nu, metas_data in zip(all_predicted_probs, all_predicted_cams, all_target_masks, all_target_masks_nu, all_metas_data):
        # Image uid
        uid = le_.inverse_transform([metas_data[0]])[0]
        orig_width = metas_data[1]
        orig_height = metas_data[2]
                
        # Class predicted
        preds = [(i, x) for i, x in enumerate(predicted_probs) if x > CLASS_PRED_THR]
        
        # Detected cells (without background)
        cell_ids = [c for c in np.unique(target_masks) if c > 0]        
        
        # --------------------
        # Identify border cells
        # Remove border cells: A good rule of thumb (that our annotators used in generating ground truth) is if more than half of the cell is not present, don't predict it!
        border_cells = get_border_cells_ids(target_masks, target_masks_nu) 

        # --------------------
        # Per cell predictions    
        cells_masks_meta, empty_intersection, image = cells_prediction_(preds, predicted_cams, target_masks, target_masks_nu, cell_ids, image=None)
                            
        # --------------------
        # Per class prediction
        cell_class_ids = defaultdict(list)
        cell_class_confidences_rle = defaultdict(list)
        for cell_id, cell_dict in cells_masks_meta.items():
            for cell_class, cell_meta in cell_dict.items():
                # Find unique cells per class
                ids = cell_class_ids[cell_class]            
                if cell_id not in ids:
                    ids.extend([cell_id])
                    confidences_list = cell_class_confidences_rle[cell_class]                
                    # Extract cell mask, resize and encode
                    cell_rle = resize_and_rle_encode_one_cell(target_masks, cell_id, orig_width, orig_height) 
                    confidences_list.extend([(cell_class, cell_meta[CELL_CONF], cell_rle, cell_id)])
        
        # Flatten all list for each class (Fixed!)
        flatten_list = None
        for cell_class, cells_ids in cell_class_ids.items():
            confidences_list_ = cell_class_confidences_rle[cell_class] # [ (cell_class1, cell_class1_confidence1, cell_mask1_rle, cell_id), (cell_class1, cell_class1_confidence2, cell_mask2_rle, cell_id) ...]
            if flatten_list is None:
                flatten_list = confidences_list_  
            else:
                flatten_list.extend(confidences_list_)

        # Nothing detected, RLE encode all cells (if available) with Negative class. It could be because:
        # No cells/masks detected by segmentation
        # No class detected (all class_prob below THR)
        # No cells/cams intersection
        # Detected cells must be labeled as Negative
        if flatten_list is None:
            print("%s - Nothing detected" % uid)
            flatten_list = [] if len(cell_ids) > 0 else None
            for cell_id in cell_ids:
                cell_rle = resize_and_rle_encode_one_cell(target_masks, cell_id, orig_width, orig_height)
                negative_conf = np.nanmax(predicted_probs) if empty_intersection is True else (1.0 - np.nanmax(predicted_probs))
                flatten_list.append((NEGATIVE_CLASS, negative_conf, cell_rle, cell_id))
        else:
            # Final cleanup
            # Sort detected class by max confidence and clip on TOP_N_PER_IMAGE
            unique_class_max_confidence = defaultdict(float)
            for item in flatten_list:
                confidence_ = unique_class_max_confidence[item[0]] # 0 if not set
                unique_class_max_confidence[item[0]] = max(item[1], confidence_)
            unique_class_max_confidence = OrderedDict(sorted(unique_class_max_confidence.items(), key=operator.itemgetter(1), reverse=True)[:TOP_N_PER_IMAGE]) # [:TOP_N_PER_IMAGE]
            if (NEGATIVE_CLASS in list(unique_class_max_confidence.keys())):
                print("%s - Warning 'Negative' class might be mixed" % uid)
            # Keep cells with filtered TOP_N classes per image
            flatten_list = [item for item in flatten_list if item[0] in list(unique_class_max_confidence.keys())]        
            # All non-labeled cells must be labeled as Negative
            if LABEL_MISSING_AS_NEGATIVE is True:
                missing_labeled_cells_ids = np.setxor1d(np.array(cell_ids), np.array([item[3] for item in flatten_list]))
                if len(missing_labeled_cells_ids) > 0:
                    # print("%s - Missing labels for cells, Negative label applied" % uid, missing_labeled_cells_ids)
                    for missing_cell_id in missing_labeled_cells_ids:
                        # What confidence should we set here? predicted_probs[NEGATIVE_CLASS] or 1-predicted_probs[NEGATIVE_CLASS] or or 1-max(predicted_probs[NEGATIVE_CLASS]) ?
                        flatten_list.append((NEGATIVE_CLASS, predicted_probs[NEGATIVE_CLASS], resize_and_rle_encode_one_cell(target_masks, missing_cell_id, orig_width, orig_height), missing_cell_id))
        
        # Remove border cells (if any)
        # print("%s border cells" % uid, list(border_cells.keys()))
        flatten_list = [item for item in flatten_list if item[3] not in list(border_cells.keys())] if flatten_list is not None else None
        
        results.append((uid, orig_width, orig_height, flatten_list))
            
    return results


# In[ ]:


NEGATIVE_CLASS = 18
LABEL_MISSING_AS_NEGATIVE = True
BORDER_SIZE_PERCENT = 0 # 0.005 # 0.5%

# CAM/Segmented cells intersection: Different solutions/approaches
cells_prediction_ = cells_prediction_puzzlecam # cells_prediction_puzzlecam_nu

TOP_N_PER_IMAGE = 18 # TOP_N classes max per image: 3 to 5 (only used for Pytorch model)
CLASS_PRED_THR = 0.01 # 0.001
CAM_PRED_QUANTILE = 0.99 # 0.995
SCORE_TH = 0.02 # 0.03

# Class confidence threshold for ensemble
ENSEMBLE_TOP_N_PER_CELL = 18
ENSEMBLE_THR = 0.005


# In[ ]:


def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factory', default="HDF5", type=none_or_str, help='HDF5 or None')
    parser.add_argument('--mtype', default='siamese', type=str)
    parser.add_argument('--backbone', default='seresnext50_32x4d', type=str)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--workers', default=8 if PT_SCRIPT is True else 0, type=int)
    parser.add_argument('--fold', default=1, type=int, help='Fold number to generate OFF, zero for holdhout', required=True)
    parser.add_argument('--weights_files', nargs="+", default=["model_best.pt"], help='list of path to weights filename', required=True)
    parser.add_argument('--metadata_file', default='meta_cleaned_default_external.csv', type=str, help='CSV file with width/height for each image')
    parser.add_argument('--labels_file', default='train_cleaned_default_external.csv', type=str, help='CSV file with labels')
    parser.add_argument('--oof_folder', default='./oof/', type=str, help='OOF output folder')
    return parser


# In[ ]:


if __name__ == '__main__':
    
    import os, sys, random, math
    
    if PT_SCRIPT is False:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        
    print('Python        : ' + sys.version.split('\n')[0])
    print('Numpy         : ' + np.__version__)
    print('Pandas        : ' + pd.__version__)
    print('PyTorch       : ' + torch.__version__)
    print('Albumentations: ' + A.__version__)
    print('Timm          : ' + timm.__version__)
    print('HPA CellSeg   : ' + hpacellseg.__version__)
    
    # Parse arguments
    args = get_argsparser().parse_args() if PT_SCRIPT is True else get_argsparser().parse_args(['--seed', '12120', '--fold', '1', '--backbone', 'gluon_seresnext101_32x4d', '--weights_files', './models/siamese_gluon_seresnext101_32x4d_512_384_RGBY_fp16_CV4_v2.0/fold1/stage1/snapshots/model_best.pt'])
    # args = get_argsparser().parse_args() if PT_SCRIPT is True else get_argsparser().parse_args(['--seed', '12120', '--labels_file', 'holdout.csv', '--fold', '0', '--backbone', 'gluon_seresnext101_32x4d', '--weights_files', './models/siamese_gluon_seresnext101_32x4d_512_384_RGBY_fp16_CV4_v2.0/fold1/stage1/snapshots/model_best.pt', './models/siamese_gluon_seresnext101_32x4d_512_384_RGBY_fp16_CV4_v2.0/fold2/stage1/snapshots/model_best.pt', './models/siamese_gluon_seresnext101_32x4d_512_384_RGBY_fp16_CV4_v2.0/fold3/stage1/snapshots/model_best.pt', './models/siamese_gluon_seresnext101_32x4d_512_384_RGBY_fp16_CV4_v2.0/fold4/stage1/snapshots/model_best.pt'])
    
    # Fixed seed for reproducibility
    seed = args.seed
    seed_everything(seed)
    
    # All data
    PARTS = ["external"]
    # if args.additional_labels_file is not None:
    #     PARTS = PARTS + ["additional"]
    
    if args.factory == "HDF5":
        ALL_IMAGES = {
            DEFAULT: TRAIN_HOME + 'images_%d.hdf5' % IMAGE_SIZE,
        }
        for p in PARTS:
            ALL_IMAGES[p] = TRAIN_HOME + 'images_%s_%d.hdf5' % (p, IMAGE_SIZE)
        DataFactory_ = HDF5DataFactory
    else:
        raise Exception("Only HDF5 factory supported for OOF")
    print("Factory", DataFactory_, ALL_IMAGES)

    INFERENCE_PATH = HOME + args.oof_folder
    if not os.path.exists(INFERENCE_PATH):
        os.makedirs(INFERENCE_PATH)    

    # Override basic configuration
    conf = raw_conf(args.factory)
    conf.mtype = args.mtype
    conf.backbone = args.backbone
    conf.BATCH_SIZE = args.batch_size
    conf.WORKERS = args.workers
    conf.image_size = IMAGE_SIZE
    conf.inference = True
    conf.seg_scale_factor = SEG_SCALE_FACTOR 
    
    # Load meta-data
    meta_pd = pd.read_csv(args.metadata_file)
    
    # Merge with labels
    submission_pd, le_ = prepare_data(args.labels_file, meta_pd=meta_pd)
    print("Total:", len(submission_pd))
    
    # Load models
    segmentator_, normalizer_ = None, None

    # HPA weights
    NUC_MODEL = HOME + "models/hpa_segmentation/nuclei-model.pth"
    CELL_MODEL = HOME + "models/hpa_segmentation/cell-model.pth"

    # Cell segmentation
    if ON_FLY_MASK is False:
        segmentator = cellsegmentator.CellSegmentator(
            NUC_MODEL,
            CELL_MODEL,
            scale_factor=conf.seg_scale_factor,
            device=conf.L_DEVICE if isinstance(conf.L_DEVICE, str) else conf.L_DEVICE.type,
            padding=False,
            multi_channel_model=True, # RYB
        )
        print("Segmentator", segmentator.device)
    else:
        segmentator_ = CellSegmentation(conf, cell_model_path=CELL_MODEL, nu_model_path=NUC_MODEL).to(conf.L_DEVICE)
        normalizer_ = MeanStdNormalize(conf, conf.IMG_MEAN, conf.IMG_STD, verbose=True)
        print()

    MODELS_NN = {
        "model": {
            "conf": conf,            
        }
    }
    for i, w in enumerate(args.weights_files, 1):
        MODELS_NN["model"]["fold%d"%i] = w
    
    # Load Classifier models
    models_ = []
    for name, info in MODELS_NN.items():    
        cfg = info.get("conf")
        if cfg is not None:
            # Folds ensembling
            models_.append(build_ensemble(name, cfg, info))
            print()
        else:
            print("Warning: Only works if preprocess_input_fn is the same for sub models")
            sub_models_ = []
            for subname, subinfo in info.items():
                subcfg = subinfo.get("conf")
                if subcfg is not None:
                    sub_models_.append(build_ensemble(subname, subcfg, subinfo))
                    print()
            models_.append(Ensemble(sub_models_, subcfg, sub_models_[-1].preprocess_input_fn))

    # Final Ensemble
    model_ = models_[0]
    
    # Pytorch TTA
    LOCAL_TTAS = None # [ident, vflip, hflip] # None
    model_ = EnsembleTTA(model_, ttas=LOCAL_TTAS) if LOCAL_TTAS is not None else model_    

    device = conf.L_DEVICE
    
    SANITY = False # True
    SANITY_SIZE = 256
    LABELS_OHE_START = 3

    train_pd = submission_pd if SANITY is False else submission_pd.sample(SANITY_SIZE)
    
    if args.fold > 0:
        # Prepare X_valid matching to target fold.
        output_file = "oof_%d.csv" % args.fold
        default_pd = train_pd      
        kf = MultilabelStratifiedKFold(n_splits=conf.FOLDS, shuffle=True, random_state=seed)        
        # Display default balance
        display_fold(default_pd, kf)
        FOLD = 0
        RESUME_FOLD = args.fold
        arguments = []
        for (train_idx, valid_idx) in kf.split(default_pd, default_pd.iloc[:, LABELS_OHE_START:LABELS_OHE_START+19]):
            FOLD = FOLD + 1
            if FOLD < RESUME_FOLD: continue
            X_train, X_valid = default_pd.iloc[train_idx], default_pd.iloc[valid_idx]            
            print('Fold', FOLD, 'train default size:', X_train.shape, 'valid default size:', X_valid.shape)             
            break
    else:
        output_file = "holdout.csv"
        X_valid = train_pd # All data for holdhout
        print('Holdout', 'valid size:', X_valid.shape)                

    # Run OOF/Holdhout
    factory = DataFactory_(ALL_IMAGES, conf=conf)
    dataset = HPADataset(X_valid, factory, conf, subset="valid", verbose=False, augment=None, modelprepare=get_preprocessing(model_.preprocess_input_fn))
    print("Device:", device, "workers:", conf.WORKERS, "post_activation:", conf.post_activation, "batch size:", conf.BATCH_SIZE, "test dataset:", len(dataset), 
          "num_classes:", conf.num_classes, "fp16:", conf.fp16, "compose:", conf.compose)
    loader = DataLoader(dataset, batch_size=conf.BATCH_SIZE, num_workers=conf.WORKERS if ON_FLY_MASK is True else 0, drop_last = False, pin_memory=conf.pin_memory, sampler=None, shuffle=False)
    predictions_ = test_loop_fn(loader, segmentator_, normalizer_, model_, conf, device)
    factory.cleanup()

    # Dump OOF/Holdout file
    cols = [ID, IMAGE_WIDTH, IMAGE_HEIGHT, "cell_id", "cell_center", "classes_prob"]
    predictions_pd = []
    for prediction_ in predictions_:
        uid, orig_width, orig_height, flatten_list = prediction_
        cell_predictions_dict = {}
        cell_centers_dict = {}
        if flatten_list is not None: # None means no cells detected
            for item in flatten_list:
                cell_class, cell_class_confidence, cell_mask1_rle, cell_id = item
                classes_probs_ = cell_predictions_dict.get(cell_id)
                cell_centers = cell_centers_dict.get(cell_id)
                if cell_centers is None:
                    cell_centers_dict[cell_id] = cell_mask1_rle            
                if classes_probs_ is None:
                    classes_probs_ = np.zeros((conf.num_classes), dtype=np.float32)
                classes_probs_[cell_class] = cell_class_confidence
                cell_predictions_dict[cell_id] = classes_probs_
            # Sort by cell_id
            cell_predictions_dict = OrderedDict(sorted(cell_predictions_dict.items(), key=operator.itemgetter(0), reverse=False))
            for cell_id, classes_probs in cell_predictions_dict.items():
                center = cell_centers_dict.get(cell_id)
                predictions_pd.append((uid, orig_width, orig_height, cell_id, center, classes_probs))

    MODEL_NAME = "%s_%s_%d_%d_%s_%s%s_v3.0" % (conf.mtype, conf.backbone, IMAGE_SIZE, CENTER_PAD if CENTER_PAD is not None else IMAGE_SIZE, COMPOSE if COMPOSE is not None else "RGBY", "fp16_" if conf.fp16 is True else "", "CV%d" % conf.FOLDS if conf.FOLDS > 0 else "FULL")
    if not os.path.exists(INFERENCE_PATH + MODEL_NAME):
        os.makedirs(INFERENCE_PATH + MODEL_NAME)

    predictions_pd = pd.DataFrame(predictions_pd, columns=cols)
    predictions_pd[[str(i) for i in range(conf.num_classes)]] = predictions_pd.apply(lambda x: x["classes_prob"], axis=1, result_type="expand")
    predictions_pd.drop(columns=["classes_prob"], inplace=True)
    predictions_pd.to_csv(INFERENCE_PATH + MODEL_NAME + "/" + output_file, index=False)   

