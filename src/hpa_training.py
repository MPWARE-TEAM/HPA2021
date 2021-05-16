#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Make sure the following dependencies are installed.
#!pip install albumentations --upgrade
#!pip install timm
#!pip install iterative-stratification
__author__ = 'MPWARE: https://www.kaggle.com/mpware'


# In[ ]:


# Configure HOME and DATA_HOME according to your setup
HOME =  "./"
DATA_HOME = "./data/"

TRAIN_HOME =  DATA_HOME + "train/"
TRAIN_IMAGES_HOME = TRAIN_HOME + "images/"

IMAGE_SIZE = 512 # Image size for training
RESIZED_IMAGE_SIZE = 384 # For random crop
COMPOSE = None # For RGBY support

# Set to True for interactive session
PT_SCRIPT = True # True


# In[ ]:


import sys, os, random, math
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
import operator
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
import torch.nn.functional as F
import functools
from collections import OrderedDict
import torch.nn.functional as F
from torch.optim import Adam, SGD
import timm
import iterstrat


# In[ ]:


LABEL = "Label"
ID = "ID"
EID = "EID"
IMAGE_WIDTH = "ImageWidth"
IMAGE_HEIGHT = "ImageHeight"
META = "META"
TOTAL = "Total"
EXT = "ext"
DEFAULT = "default"

# 19 class labels. Some rare classes: Mitotic spindle (0.37%), Negative: (0.15%)
class_mapping = {
    0: 'Nucleoplasm', 1: 'Nuclear membrane', 2: 'Nucleoli', 3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles', 5: 'Nuclear bodies', 6: 'Endoplasmic reticulum', 7: 'Golgi apparatus', 8: 'Intermediate filaments',
    9: 'Actin filaments', 10: 'Microtubules', 11: 'Mitotic spindle', 12: 'Centrosome', 13: 'Plasma membrane', 14: 'Mitochondria',
    15: 'Aggresome', 16: 'Cytosol', 17: 'Vesicles and punctate cytosolic patterns', 18: 'Negative',
}

class_mapping_inv = {v:k for k,v in class_mapping.items()}
class_labels = [str(k) for k,v in class_mapping.items()]
class_names = [str(v) for k,v in class_mapping.items()]

LABELS_OHE_START = 3


# In[ ]:


def seed_everything(s):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# In[ ]:


def l1_loss(A_tensors, B_tensors):
    return torch.abs(A_tensors - B_tensors)

class ComboLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.from_logits = from_logits
        print("alpha:", self.alpha, "beta:", self.beta, "gamma:", self.gamma)
        self.loss_classification = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, features_single=None, y_pred_tiles=None, features_tiles=None, y_pred_tiled_flatten=None):
        
        loss_ = self.alpha * self.loss_classification(y_pred, y_true).mean()
                    
        if features_tiles is not None and self.beta > 0:
            logits_reconstruction = y_pred_tiles
            loss_tiles_class_ = self.loss_classification(logits_reconstruction, y_true).mean()
            loss_ = loss_ + self.beta * loss_tiles_class_
                                    
        if features_single is not None and features_tiles is not None and self.gamma > 0:
            loss_reconstruction_ = l1_loss(features_single, features_tiles).mean()
            loss_ = loss_ + self.gamma * loss_reconstruction_
            
        return loss_


# In[ ]:


# Main configuration
class raw_conf:

    def __init__(self, factory):
        super().__init__()
        
        self.inference = False

        self.compose = COMPOSE
        self.normalize = False if factory == "HDF5" else True
        self.norm_value = None if factory == "HDF5" else 65535.0
        
        # Dataset
        self.image_size = None if factory == "HDF5" else IMAGE_SIZE
        self.denormalize = 255
                
        # Model
        self.mtype = "siamese" # "regular"
        self.backbone = 'seresnext50_32x4d' # 'gluon_seresnext101_32x4d' # 'cspresnext50' 'regnety_064'
        self.pretrained_weights = "imagenet"
        self.INPUT_RANGE = [0, 1]
        self.IMG_MEAN = [0.485, 0.456, 0.406, 0.485] if self.compose is None else [0.485, 0.456, 0.406]
        self.IMG_STD = [0.229, 0.224, 0.225, 0.229] if self.compose is None else [0.229, 0.224, 0.225]
        self.num_classes = 19
        self.with_cam = True
        self.puzzle_pieces = 4
        self.hpa_classifier_weights = None 
        self.dropout = None
        
        # Model output
        self.post_activation = "sigmoid"
        self.output_key = "logits" if self.mtype == "regular" else "single_logits" # None
        self.output_key_extra = "features" if self.mtype == "regular" else "single_features" # None
        self.output_key_siamese = None if self.mtype == "regular" else "tiled_logits"
        self.output_key_extra_siamese = None if self.mtype == "regular" else "tiled_features"      
        
        # Loss
        self.alpha = 1.0 # Single image classification loss
        self.beta =  0.0 if self.mtype == "regular" else 1.0 # Reconstructed image classification loss
        self.gamma = 0.0 if self.mtype == "regular" else 0.5 # 0.25
        self.loss = ComboLoss(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        
        self.sampler = "prob"
        self.sampler_cap = "auto" # None
        
        self.fp16 = True
        self.finetune = False
        
        self.optimizer = "Adam" # "SGD"
        self.scheduler = None if self.finetune is True or self.optimizer != "Adam" else "ReduceLROnPlateau" # "CosineAnnealingWarmRestarts"
        self.scheduler_factor = 0.3
        self.scheduler_patience = 8
        
        self.lr = 0.0003
        self.min_lr = 0.00005
        self.beta1 = 0.9
        self.train_verbose = True
        self.valid_verbose = True

        # Train parameters
        self.L_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.map_location = self.L_DEVICE
        self.WORKERS = 0 if PT_SCRIPT is False else 8
        self.BATCH_SIZE = 36 if self.mtype == "siamese" else 48
        self.ITERATIONS_LOGS = 30
        self.CYCLES = 1
        self.EPOCHS_PER_CYCLE = 48 # 36
        self.EPOCHS = self.CYCLES * self.EPOCHS_PER_CYCLE
        self.WARMUP = 0

        self.FOLDS = 4
        self.METRIC_ = "min" # "max"

        self.pin_memory = True


# In[ ]:


# Load CSV data, drop duplicates if any
def prepare_data(filename, ext_name=None):
    train_pd = pd.read_csv(DATA_HOME + filename)
    train_pd[LABEL] = train_pd[LABEL].apply(literal_eval)
    train_pd[LABEL] = train_pd[LABEL].apply(lambda x: [int(l) for l in x])    
    if EXT not in train_pd.columns:
        train_pd.insert(2, EXT, DEFAULT)
    if ext_name is not None:
        train_pd[EXT] = ext_name
    train_pd = train_pd.drop_duplicates(subset=[ID]).reset_index(drop=True)
    assert(np.argwhere(train_pd.columns.values == EXT)[0][0] == 2)
    return train_pd


# In[ ]:


# Use PIL to support 16 bits, normalize=True to return [0-1.0] float32 image
def read_image(filename, compose=None, normalize=False, norm_value=65535.0, images_root=TRAIN_IMAGES_HOME):
    filename = images_root + filename
    filename = filename + "_red.png" if "_red.png" not in filename else filename
    mt_, pi_, nu_, er_ = filename, filename.replace('_red', '_green'), filename.replace('_red', '_blue'), filename.replace('_red', '_yellow')
    if compose is None:
        mt = np.asarray(Image.open(mt_)).astype(np.uint16)
        pi = np.asarray(Image.open(pi_)).astype(np.uint16)
        nu = np.asarray(Image.open(nu_)).astype(np.uint16)  
        er = np.asarray(Image.open(er_)).astype(np.uint16)
        ret = np.dstack((mt, pi, nu, er))
    else:
        if compose == "RGB": 
            mt = np.asarray(Image.open(mt_)).astype(np.uint16)
            pi = np.asarray(Image.open(pi_)).astype(np.uint16)
            nu = np.asarray(Image.open(nu_)).astype(np.uint16)
            ret = np.dstack((mt, pi, nu))        
        elif compose == "RYB":
            mt = np.asarray(Image.open(mt_)).astype(np.uint16)
            er = np.asarray(Image.open(er_)).astype(np.uint16)
            nu = np.asarray(Image.open(nu_)).astype(np.uint16)
            ret = np.dstack((mt, er, nu))
        elif compose == "RYGYB":
            mt = np.asarray(Image.open(mt_))
            pi = np.asarray(Image.open(pi_))
            nu = np.asarray(Image.open(nu_))
            er = np.asarray(Image.open(er_))
            ret = np.dstack(((mt + er)/2.0, (pi + er/2)/1.5, nu))
        else:
            raise Exception("Unknown compose:", compose)
    if normalize is True:
        # Some images are np.uint16 but from 0-255 range!
        if ret.max() > 255:
            ret = (ret/norm_value).astype(np.float32)
        else:
            ret = (ret/255).astype(np.float32)
    return ret

# Data available through raw PNG files 
class DataFactory:
    def __init__(self, paths, conf=None, verbose=False):
        super().__init__()
        
        self.paths = paths
        self.conf = conf
        self.verbose = verbose
        print("PNGFile factory") if self.verbose is True else None
    
    def read_image(self, uid, container=None):
        images_path = self.paths
        if container is not None and container != DEFAULT:
            images_path = images_path.replace("images", container)        
        image = read_image(uid, compose=self.conf.compose, normalize=self.conf.normalize, norm_value=self.conf.norm_value, images_root=images_path)        
        return image
    
    def cleanup(self):
        pass   

    
# Data available through HDF5 files
class HDF5DataFactory:
    def __init__(self, paths, conf=None, verbose=False):
        super().__init__()
        
        self.paths = paths
        self.hdf5_paths = None
        self.conf = conf
        self.verbose = verbose
        self.initialized = False
        print("HDF5 factory") if self.verbose is True else None

    def initialize_hdf5(self):
        if self.initialized is False:
            self.hdf5_paths = h5py.File(self.paths, 'r') if isinstance(self.paths, str) else {k: h5py.File(v, 'r') for k, v in self.paths.items()}
            self.initialized = True                
            print("initialize_hdf5", self.hdf5_paths) if self.verbose is True else None
                
    def read_image(self, uid, container=DEFAULT):
        self.initialize_hdf5()
        
        hdf5_paths_ = self.hdf5_paths if isinstance(self.hdf5_paths, str) else self.hdf5_paths.get(container)
        
        # Image is already resized, normalized 0-1.0 as float32
        image = hdf5_paths_[uid][:,:,:]
        
        if self.conf.compose is not None:
            if self.conf.compose == "RGB":
                image = image[:, :, [0,1,2]]             
            elif self.conf.compose == "RYB":
                image = image[:, :, [0,3,2]] 
            elif self.conf.compose == "G":
                image = np.dstack((image[:, :, 1], image[:, :, 1], image[:, :, 1]))         
            elif self.conf.compose == "RYGYB":
                ret = np.dstack(((image[:, :, 0] + image[:, :, 3])/2.0, (image[:, :, 1] + image[:, :, 3]/2)/1.5, image[:, :, 2]))
            else:
                raise Exception("Unknown compose:", self.conf.compose)
        
        return image
        
    def cleanup(self):
        if self.hdf5_paths is not None:
            [v.close() for k, v in self.hdf5_paths.items()] if isinstance(self.hdf5_paths, dict) else self.hdf5_paths.close()           
        print("HDF5 factory cleaned") if self.verbose is True else None


# In[ ]:


# Dataset with all images
def zero(x, y=None):
    return 0
        
class HPADataset(Dataset):
    def __init__(self, df, factory, conf, subset="train", categoricals=None, augment=None, postprocess=None, modelprepare=None, classes=None, weights=False, dump=None, verbose=False):
        super().__init__()
        
        self.df = df
        self.categoricals = categoricals
        self.subset = subset
        self.augment = augment
        self.postprocess = postprocess
        self.modelprepare = modelprepare
        self.classes = classes
        self.conf = conf
        self.factory = factory
        self.dump = dump
        self.verbose = verbose
        
        if subset == 'train':
            self.get_offset = np.random.randint
        elif subset == 'valid':
            self.get_offset = zero
        elif subset == 'ho':
            self.get_offset = zero
        elif subset == 'test':
            self.get_offset = zero           
        else:
            raise RuntimeError("Unknown subset")
        
        # Compute weights
        self.weights = self.compute_weights(self.df) if subset == "train" and weights is True else None


    def prob_from_weight(self, labels_list, weights_dict_, cap=None):
        labels_weights = np.array([weights_dict_[class_mapping[int(label_)]] for label_ in labels_list])
        prob_ = np.nanmean(labels_weights)
        if cap is not None:
            prob_ = np.clip(prob_, 0, cap) # Clip to avoid too much single rare labels, for example: 95th percentile cut, or top K
        return prob_
        
    
    def compute_weights(self, df_):
        weights_dict = {label: 1/df_[label].sum() for label in class_names}
        cap_ = self.conf.sampler_cap
        if cap_ is not None and cap_ == "auto":
            top_weights = sorted(weights_dict.items(), key=operator.itemgetter(1), reverse=True)[:3]
            print("top_weights", top_weights) if self.verbose is True else None
            cap_ = top_weights[2][1] # Cap to the top 3rd weight        
        df_dist = df_[[ID, LABEL]].copy()
        df_dist["prob"] = df_dist[LABEL].apply(lambda x: self.prob_from_weight(x, weights_dict, cap=cap_))  
        if self.verbose is True:
            print("compute_weights completed, cap:", self.conf.sampler_cap, cap_)
            for i, (k, v) in enumerate(weights_dict.items()):
                print(i, k, v)
        return df_dist[["prob"]]
    
    def cleanup(self):
        self.factory.cleanup()
    
    def __len__(self):
        return len(self.df)
    
    def read_image(self, row):        
        uid = row[ID]
        container = row[EXT]
        
        # Load image
        img = self.factory.read_image(uid, container=container)
        
        # Scale image after cropping
        if self.conf.image_size is not None and self.conf.image_size != img.shape[0]:
            img = skimage.transform.resize(img, (self.conf.image_size, self.conf.image_size), anti_aliasing=True) # Works with float image
        
        if self.conf.denormalize is not None:
            img = (self.conf.denormalize * img).astype(np.uint8)
        
        return img
    
    def get_data(self, row, categoricals):

        # Return image
        img = self.read_image(row)
                
        # Labels (OHE)
        labels = np.zeros(self.conf.num_classes, dtype=np.uint8)
        for l in row[LABEL]:
            labels[l] = 1
        
        sample = {
            'image': img,
            'label': labels,
        }

        if self.dump is not None:
            sample[ID] = row[ID]
            
        if EID in row:
            sample[META] = np.array([row[EID], int(row[IMAGE_WIDTH]), int(row[IMAGE_HEIGHT])], dtype=np.int32)            

        # Optional augmentation on RGBY image (uint8)
        if self.augment:
            tmp = self.augment(image=sample['image'])
            sample['image'] = tmp["image"] # Apply on full image
        
        # Mandatory to feed model
        if self.modelprepare: # Albumentations to normalize data
            tmp = self.modelprepare(image=sample['image'])
            sample['image'] = tmp["image"] # Apply on full image      

        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        sample = self.get_data(row, self.categoricals)
        
        return sample


# In[ ]:


# (BS, CLASSES, 12, 12) - Between 0-1.0
# Adapted from: https://github.com/OFRIN/PuzzleCAM/blob/master/core/puzzle_utils.py
def make_cam(x, epsilon=1e-5):
    x = F.relu(x) # (BS, CLASSES, 12, 12)

    b, c, h, w = x.size() # (BS, CLASSES, 12, 21)    
    flat_x = x.view(b, c, (h * w)) # (BS, CLASSES, 12x12)    
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon) # (BS, CLASSES, 12, 12)    
    

# Input  (BS, C, H, W), num_pieces = 4
# Return (BS*4, C, H//4, W//4)
# Adapted from: https://github.com/OFRIN/PuzzleCAM/blob/master/core/puzzle_utils.py
def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    h_per_patch = h // num_pieces_per_line
    w_per_patch = w // num_pieces_per_line
    
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)

# Adapted from: https://github.com/OFRIN/PuzzleCAM/blob/master/core/puzzle_utils.py
def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):

        ext_w_list = []
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])
            index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features


# In[ ]:


# Add 4 channels support
def get_4channels_conv(stem_conv2d):
    stem_conv2d_pretrained_weight = stem_conv2d.weight.clone()
    stem_conv2d_ = nn.Conv2d(4, 
                             stem_conv2d.out_channels, kernel_size=stem_conv2d.kernel_size, stride=stem_conv2d.stride, padding=stem_conv2d.padding, padding_mode=stem_conv2d.padding_mode, dilation=stem_conv2d.dilation, 
                             bias=True if stem_conv2d.bias is True else False)            
    stem_conv2d_.weight = nn.Parameter(torch.cat([stem_conv2d_pretrained_weight, nn.Parameter(torch.mean(stem_conv2d_pretrained_weight, axis=1).unsqueeze(1))], axis=1))  
    return stem_conv2d_


class HPAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.backbone = cfg.backbone
        self.with_cam = cfg.with_cam
        self.drop_rate = cfg.dropout
        
        self.preprocess_input_fn = get_preprocessing_fn(cfg) 
                
        # Unpooled/NoClassifier (features only)
        self.mfeatures = timm.create_model(self.backbone, pretrained=True, num_classes=0, global_pool='')
        
        # Add one channel more
        if cfg.compose is None:
            if "regnet" in self.backbone:
                self.mfeatures.stem.conv = get_4channels_conv(self.mfeatures.stem.conv)
            elif "csp" in self.backbone:
                self.mfeatures.stem[0].conv = get_4channels_conv(self.mfeatures.stem[0].conv)                
            elif "resnest" in self.backbone:
                self.mfeatures.conv1[0] = get_4channels_conv(self.mfeatures.conv1[0])
            elif "seresnext" in self.backbone:
                self.mfeatures.conv1 = get_4channels_conv(self.mfeatures.conv1)                
            elif "densenet" in self.backbone:
                self.mfeatures.features.conv0 = get_4channels_conv(self.mfeatures.features.conv0)  
                 
        # Classifier
        num_chs = self.mfeatures.feature_info[-1]['num_chs'] # 1296 # 2048      
        self.mclassifier = nn.Conv2d(num_chs, self.num_classes, 1, bias=False)
        # self.mclassifier = timm.models.layers.linear.Linear(num_chs, self.num_classes, bias=True)
        
        # Initialize weights
        self.initialize([self.mclassifier])
        
        print("Model %s, last channels: %d, classes: %d" % (cfg.backbone, num_chs, self.num_classes))
    
    # Pooling
    def adaptive_avgmax_pool2d(self, x, output_size=1):
        x_avg = F.adaptive_avg_pool2d(x, output_size)
        x_max = F.adaptive_max_pool2d(x, output_size)
        return 0.5 * (x_avg + x_max)

    # Average pooling 2d
    def global_average_pooling_2d(self, x, keepdims=False):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        if keepdims:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return x
    
    def gap(self, x, keepdims=False):
        return self.global_average_pooling_2d(x, keepdims=keepdims)
    
    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()       
    
    def forward(self, x):
        # ([BS, C, H, W])
        x = self.mfeatures(x) # (BS, num_chs, 12, 12)
        
        features = None
        if self.with_cam is True:
            if self.drop_rate is not None and self.drop_rate > 0.0:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)            
            features = self.mclassifier(x) # (BS, CLASSES, 12, 12)
            logits = self.gap(features) # (BS, CLASSES)            
        else:            
            x = self.gap(x, keepdims=True) # (BS, num_chs, 1, 1)
            if self.drop_rate is not None and self.drop_rate > 0.0:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            logits = self.mclassifier(x).view(-1, self.num_classes) # (BS, CLASSES)            
        
        return {"logits": logits, "features": features} # (BS, CLASSES), (BS, CLASSES, 12, 12)


class HPASiameseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.backbone = cfg.backbone
        self.with_cam = cfg.with_cam
        self.puzzle_pieces = cfg.puzzle_pieces
        self.preprocess_input_fn = get_preprocessing_fn(cfg) 
        self.cnn1 = HPAModel(cfg)
        
        if cfg.hpa_classifier_weights is not None:
            if os.path.exists(cfg.hpa_classifier_weights):
                print("Load regular HPA weights from: %s" % cfg.hpa_classifier_weights)
                self.cnn1.load_state_dict(torch.load(cfg.hpa_classifier_weights, map_location=cfg.map_location))            
        
        print("Model %s" % (cfg.mtype))
         

    def forward_once(self, x):
        x = self.cnn1(x)
        return x # {"logits": logits, "features": features}
    
    
    def forward(self, x):
        # ([BS, C, H, W])
        bs, _, _, _ = x.shape
        
        # Full image
        x1 = self.forward_once(x)
        single_logits, single_features = x1["logits"], x1["features"]
        
        # Tiled image
        tiled_x = tile_features(x, self.puzzle_pieces) # (BS*puzzle_pieces, C, H//puzzle_pieces, W//puzzle_pieces) # 2x memory
        x2 = self.forward_once(tiled_x) # Shared weights
        tiled_logits, tiled_features = x2["logits"], x2["features"]        
        
        tiled_features = merge_features(tiled_features, self.puzzle_pieces, bs) # (BS, CLASSES, 12, 12)
        tiled_logits_reconstructed = self.cnn1.gap(tiled_features) # (BS, CLASSES)
                
        return {
            "single_logits": single_logits, "single_features": single_features,
            "tiled_logits_flatten": tiled_logits, "tiled_features": tiled_features,
            "tiled_logits": tiled_logits_reconstructed,
        }    


# In[ ]:


def build_model(cfg, device, encoder_weights=None):

    if cfg.mtype == "siamese":
        model = HPASiameseModel(cfg)
    else:
        model = HPAModel(cfg)

    # Load weights
    if (encoder_weights is not None) and ("imagenet" not in encoder_weights):
        if os.path.exists(encoder_weights):
            print("Load weights before optimizer from: %s" % encoder_weights)
            model.load_state_dict(torch.load(encoder_weights, map_location=cfg.map_location))

    model = model.to(device)

    if cfg.optimizer == "Adam":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    elif cfg.optimizer == "SGD":
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=0.9)

    # Loss
    loss = cfg.loss
    loss = loss.to(device)

    return model, loss, optimizer


# In[ ]:


def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s

# Train loop
def train_loop_fn(batches, preprocessing, model, optimizer, criterion, tmp_conf, device, stage="Train", verbose=True, scaler=None):
    model.train()
    count, train_loss = 0, 0.0
    all_predicted_probs, all_target_classes = None, None
    
    with tqdm(batches, desc=stage, file=sys.stdout, disable=not(verbose)) as iterator:
        for x, batch in enumerate(iterator, 1):
            try:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                samples_data, labels_data = batch.get("image"), batch.get("label")
                
                optimizer.zero_grad() # reset gradient

                # Model
                with torch.cuda.amp.autocast(enabled=tmp_conf.fp16):
                    # Preprocessing
                    with torch.no_grad():
                        data = preprocessing(samples_data) if preprocessing is not None else samples_data                    
                    output = model(data) # forward pass
                    
                    if tmp_conf.mtype == "siamese":
                        loss = criterion(output[tmp_conf.output_key], labels_data.float(),
                                         features_single=output[tmp_conf.output_key_extra],  
                                         y_pred_tiles=output[tmp_conf.output_key_siamese],
                                         features_tiles=output[tmp_conf.output_key_extra_siamese],
                                         y_pred_tiled_flatten=output["tiled_logits_flatten"])
                        output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output
                    else:
                        output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output
                        # Compute loss
                        loss = criterion(output, labels_data.float())

                if (tmp_conf.ITERATIONS_LOGS > 0) and (x % tmp_conf.ITERATIONS_LOGS == 0):
                    loss_value = loss.item()
                    if ~np.isnan(loss_value): train_loss += loss_value
                    else: print("Warning: NaN loss")                
                
                # backward pass
                scaler.scale(loss).backward() if scaler is not None else loss.backward()

                # Update weights
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (tmp_conf.ITERATIONS_LOGS > 0) and (x % tmp_conf.ITERATIONS_LOGS == 0):
                    # Labels predictions
                    predicted_probs = torch.sigmoid(output) if tmp_conf.post_activation == "sigmoid" else output
                    predicted_probs = predicted_probs.detach().cpu().numpy()
                    target_classes = labels_data.detach().cpu().numpy()

                    # Concatenate for all batches
                    all_predicted_probs = np.concatenate([all_predicted_probs, predicted_probs], axis=0) if all_predicted_probs is not None else predicted_probs
                    all_target_classes = np.concatenate([all_target_classes, target_classes], axis=0) if all_target_classes is not None else target_classes 

                    count += 1

                    if verbose: 
                        scores_str = {"train_%s" % m.__name__: m(all_target_classes, all_predicted_probs) for m in METRICS_PROBS}
                        scores_str["train_loss"] = (train_loss / count)
                        iterator.set_postfix_str(format_logs(scores_str))
                                    
            except Exception as ex:
                print("Training batch error:", ex)
    
    scores = {"train_%s" % m.__name__: m(all_target_classes, all_predicted_probs) for m in METRICS_PROBS}
    scores["train_loss"] = (train_loss / count)
    
    return (scores, all_target_classes, all_predicted_probs)


# In[ ]:


# Valid loop
def valid_loop_fn(batches, preprocessing, model, criterion, tmp_conf, device, stage="Valid", verbose=True):
    model.eval()
    count, valid_loss = 0, 0.0
    all_predicted_probs, all_target_classes = None, None

    with tqdm(batches, desc=stage, file=sys.stdout, disable=not(verbose)) as iterator:
        for batch in iterator:
            try:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                samples_data, labels_data = batch.get("image"), batch.get("label")

                with torch.no_grad():
                    # NN model
                    with torch.cuda.amp.autocast(enabled=tmp_conf.fp16):
                        # Preprocessing
                        data = preprocessing(samples_data) if preprocessing is not None else samples_data                        
                        output = model(data) # forward pass
                        if tmp_conf.mtype == "siamese":
                            loss = criterion(output[tmp_conf.output_key], labels_data.float(),
                                             features_single=output[tmp_conf.output_key_extra],  
                                             y_pred_tiles=output[tmp_conf.output_key_siamese],
                                             features_tiles=output[tmp_conf.output_key_extra_siamese],
                                             y_pred_tiled_flatten=output["tiled_logits_flatten"])
                            output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output
                        else:
                            output = output[tmp_conf.output_key] if tmp_conf.output_key is not None else output
                            # Compute loss
                            loss = criterion(output, labels_data.float())

                loss_value = loss.item()
                if ~np.isnan(loss_value): valid_loss += loss_value
                else: print("Warning: NaN loss")
          
                # Labels predictions                
                predicted_probs = torch.sigmoid(output) if tmp_conf.post_activation == "sigmoid" else output
                predicted_probs = predicted_probs.detach().cpu().numpy()
                target_classes = labels_data.detach().cpu().numpy()

                # Concatenate for all batches
                all_predicted_probs = np.concatenate([all_predicted_probs, predicted_probs], axis=0) if all_predicted_probs is not None else predicted_probs
                all_target_classes = np.concatenate([all_target_classes, target_classes], axis=0) if all_target_classes is not None else target_classes      

                count += 1

                if verbose: 
                    scores_str = {"valid_%s" % m.__name__: m(all_target_classes, all_predicted_probs) for m in METRICS_PROBS}
                    scores_str["valid_loss"] = (valid_loss / count)
                    iterator.set_postfix_str(format_logs(scores_str))
                
            except Exception as ex:
                print("Validation batch error:", ex)
    
    scores = {"valid_%s" % m.__name__: m(all_target_classes, all_predicted_probs) for m in METRICS_PROBS}
    scores["valid_loss"] = (valid_loss / count)    

    return (scores, all_target_classes, all_predicted_probs)


# In[ ]:


# Train one fold
def run_stage(X_train, X_valid, stage, fold, device):
    
    # Build model
    snapshot_path = "%s/fold%d/%s/snapshots" % (MODEL_PATH, fold, stage)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    cnn_model, criterion, optimizer = build_model(conf, device, 
                                                  encoder_weights=os.path.join(snapshot_path.replace(stage, PRETRAINED_STAGE), MODEL_BEST) if PRETRAINED_STAGE is not None else None)
    if RESUME == True:
        resume_path = os.path.join(snapshot_path, MODEL_BEST)
        if os.path.exists(resume_path):
            cnn_model.load_state_dict(torch.load(resume_path, map_location=conf.map_location))
            print("Resuming, model weights loaded: %s" % resume_path)
    
    factory = DataFactory_(ALL_IMAGES, conf=conf)
        
    # Datasets
    train_dataset = HPADataset(X_train, factory, conf, subset="train", augment=image_augmentation_train, modelprepare=get_preprocessing(cnn_model.preprocess_input_fn), dump=None, weights=True, verbose=True)
    valid_dataset = HPADataset(X_valid, factory, conf, subset="valid", augment=None, modelprepare=get_preprocessing(cnn_model.preprocess_input_fn), dump=None, verbose=False) if X_valid is not None else None
    
    train_sampler = WeightedRandomSampler(weights=train_dataset.weights[conf.sampler].values, replacement=True, num_samples=len(train_dataset)) if conf.sampler is not None else None
    
    print("Stage:", stage, "fold:", fold, "on:", device, "workers:", conf.WORKERS, "post_activation:", conf.post_activation, "batch size:", conf.BATCH_SIZE, "metric_:", conf.METRIC_, 
          "train dataset:", len(train_dataset), "valid dataset:", len(valid_dataset) if valid_dataset is not None else None, "num_classes:", conf.num_classes, "fp16:", conf.fp16, "aug:", image_augmentation_train, 
          "sampler:", train_sampler)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, sampler=train_sampler, num_workers=conf.WORKERS, drop_last = False, pin_memory=conf.pin_memory, shuffle=True if train_sampler is None else False)
    valid_loader = DataLoader(valid_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=conf.WORKERS, drop_last = False, pin_memory=conf.pin_memory) if X_valid is not None else None
    
    scheduler = None
    if conf.scheduler is not None:
        if conf.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf.METRIC_, factor=conf.scheduler_factor, min_lr=0.000001, patience=conf.scheduler_patience, verbose=True)            
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, conf.EPOCHS_PER_CYCLE, T_mult=1, eta_min=conf.min_lr)
    print(criterion, optimizer, scheduler)

    metric = METRIC_NAME
    valid_loss_min = np.Inf
    metric_loss_criterion = np.Inf if conf.METRIC_ == "min" else -np.Inf
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=conf.fp16) if conf.fp16 is True else None
    for epoch in tqdm(range(1, conf.EPOCHS + 1)):

        lr = optimizer.param_groups[0]['lr'] if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'] if isinstance(optimizer, torch.optim.SGD) or isinstance(optimizer, torch.optim.Adam) else optimizer.get_last_lr()
        info = "[%d], lr=%.7f" % (epoch, lr)

        # Train loop
        train_scores, _, _ = train_loop_fn(train_loader, None, cnn_model, optimizer, criterion, conf, device, stage="Train%s" % info, verbose=conf.train_verbose, scaler=scaler)
        
        # Validation loop
        valid_scores, _, all_predicted_probs_ = valid_loop_fn(valid_loader, None, cnn_model, criterion, conf, device, stage="Valid%s" % info, verbose=conf.valid_verbose) if valid_loader is not None else ({"valid_%s" % metric: 0, "valid_loss": 0}, None, None)
        
        # Keep track of loss and metrics
        history.append({"epoch":epoch, "lr": lr, **train_scores, **valid_scores})

        if conf.scheduler is not None:
            scheduler.step(valid_scores["valid_%s" % metric]) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

        metric_loss = valid_scores["valid_%s" % metric]
        if (conf.METRIC_ == "min" and metric_loss < metric_loss_criterion and epoch > 1) or (conf.METRIC_ == "max" and metric_loss > metric_loss_criterion and epoch > 1):
            print("Epoch%s, Valid loss from: %.4f to %.4f, Metric improved from %.4f to %.4f, saving model ..." % (info, valid_loss_min, valid_scores["valid_loss"], metric_loss_criterion, metric_loss))
            metric_loss_criterion = metric_loss
            valid_loss_min = valid_scores["valid_loss"]
            torch.save(cnn_model.state_dict(), os.path.join(snapshot_path, MODEL_BEST))
            # Save per image OOF
            oof_pd = pd.DataFrame(all_predicted_probs_)
            oof_pd = oof_pd.set_index(X_valid[ID].values)                
            oof_pd.to_csv("%s/oof_%d.csv" % (snapshot_path, fold))
            

    factory.cleanup()
    
    if history:
        # Plot training history
        history_pd = pd.DataFrame(history[1:]).set_index("epoch")
        train_history_pd = history_pd[[c for c in history_pd.columns if "train_" in c]]
        valid_history_pd = history_pd[[c for c in history_pd.columns if "valid_" in c]]
        lr_history_pd = history_pd[[c for c in history_pd.columns if "lr" in c]]
        fig, ax = plt.subplots(1,2, figsize=(DEFAULT_FIG_WIDTH, 6))
        t_epoch = train_history_pd["train_%s" % metric].argmin() if conf.METRIC_ == "min" else train_history_pd["train_%s" % metric].argmax()
        v_epoch = valid_history_pd["valid_%s" % metric].argmin() if conf.METRIC_ == "min" else valid_history_pd["valid_%s" % metric].argmax()
        d = train_history_pd.plot(kind="line", ax=ax[0], title="Epoch: %d, Train: %.3f" % (t_epoch, train_history_pd.iloc[t_epoch,:]["train_%s" % metric]))
        d = lr_history_pd.plot(kind="line", ax=ax[0], secondary_y=True)
        d = valid_history_pd.plot(kind="line", ax=ax[1], title="Epoch: %d, Valid: %.3f" % (v_epoch, valid_history_pd.iloc[v_epoch,:]["valid_%s" % metric]))
        d = lr_history_pd.plot(kind="line", ax=ax[1], secondary_y=True)
        train_history_pd.to_csv("%s/train.csv" % snapshot_path)
        valid_history_pd.to_csv("%s/valid.csv" % snapshot_path)        
        plt.savefig("%s/train.png" % snapshot_path, bbox_inches='tight')
        plt.show() if PT_SCRIPT is False else None

    return (history)


# In[ ]:


# Mandatory transform to feed model
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def aug_custom(x, **kwargs):
    return x

def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def get_custom(**kwargs):
    return A.Lambda(name="custom", image=aug_custom, **kwargs)

def get_preprocessing_fn(cfg):
    params = {"mean": cfg.IMG_MEAN, "std": cfg.IMG_STD, "input_range": cfg.INPUT_RANGE}
    return functools.partial(preprocess_input, **params)

def get_preprocessing(preprocessing_fn):
    return A.Compose([
        A.Lambda(image=preprocessing_fn),            # Convert uint8 (0-255) in range [0-1.0] and apply Apply Z-Norm that depends on each model,
        A.Lambda(image=to_tensor),                   # Convert (H, W, C) to (C, H, W)
    ])


# In[ ]:


# Optional augmentations (Works with C=4 layers)
def image_harder_augmentation_train(p=1.0):
    return A.Compose([
        
        # Crop smaller tile randomly (uint8, float32, H, W, C)
        A.RandomCrop(RESIZED_IMAGE_SIZE, RESIZED_IMAGE_SIZE, always_apply=True, p=1.0) if RESIZED_IMAGE_SIZE != IMAGE_SIZE else A.NoOp(p=1.0),
        
        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.IAAAdditiveGaussianNoise(scale=(0.02 * 255, 0.05 * 255), p=0.5),
        ], p=0.5),
        
        # Flips/Rotations        
        A.HorizontalFlip(p=0.5),       
        A.RandomRotate90(p=1.0),        
        
        # Rotate/Distorsion
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.15),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.75),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.15),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.10),
            A.IAAAffine(shear=5.0, p=0.5),
        ], p=0.5),
        
        # Blurs
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5), # 5 to 7
            A.MotionBlur(blur_limit=(3, 5), p=0.5),
            A.MedianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5),
        
        # Stain/colors
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.25),        

  ], p=p)


# In[ ]:


# Display inputs/outputs of the model
def check_model(conf):
    print("Check model")
    m, _, _ = build_model(conf, "cpu", verbose=True)
    tmp_factory = DataFactory_(ALL_IMAGES, conf=conf, verbose=True)
    tmp_dataset = HPADataset(train_pd, tmp_factory, conf, subset="train", verbose=False, augment=image_augmentation_train, modelprepare=get_preprocessing(m.preprocess_input_fn), weights=True)
    print("tmp_dataset:", len(tmp_dataset))
    tmp_sampler = WeightedRandomSampler(weights=tmp_dataset.weights[conf.sampler].values, replacement=True, num_samples=len(tmp_dataset)) if conf.sampler is not None else None
    # tmp_sampler = torch.utils.data.RandomSampler(tmp_dataset)
    tmp_loader = DataLoader(tmp_dataset, batch_size=5, num_workers=0, drop_last = False, pin_memory=False, sampler=tmp_sampler, shuffle=False)
    for tmp_batch in tmp_loader:
        for key, value in tmp_batch.items():
            print(key, value.shape, value.dtype, "min", value.min(), "max", value.max(), "mean", value.float().mean())
        tmp_out = m(tmp_batch.get("image"))
        tmp_out = tmp_out[conf.output_key] if conf.output_key is not None else tmp_out             
        break
    print("tmp_out", tmp_out.shape, tmp_out.dtype, "min", tmp_out.min(), "max", tmp_out.max(), "mean", tmp_out.mean())
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("Params:", total_params)
    tmp_loss = conf.loss(tmp_out, tmp_batch["label"].float()) # (BS, seq_len), (BS, seq_len)
    print("loss", tmp_loss)
    tmp_probs = torch.sigmoid(tmp_out) if conf.post_activation == "sigmoid" else tmp_out
    print("tmp_probs", tmp_probs.shape, tmp_probs.dtype, "min", tmp_probs.min(), "max", tmp_probs.max(), "mean", tmp_probs.mean())
    tmp_factory.cleanup()


# In[ ]:


# Some previews with augmentation
def display_preview():
    tmp_factory = DataFactory_(ALL_IMAGES, conf=conf, verbose=True)
    tmp_dataset = HPADataset(train_pd, tmp_factory, conf, subset="train", verbose=False, augment=image_augmentation_train)
    print("tmp_dataset:", len(tmp_dataset))
    tmp_loader = DataLoader(tmp_dataset, batch_size=16, num_workers=0, drop_last = False, pin_memory=False, sampler=None, shuffle=False)
    
    ROWS = 6
    COLS = 8
    fig, ax = plt.subplots(ROWS, COLS, figsize=(20, 16))

    print("Loading images")
    i = 0
    for tmp_batch in tmp_loader:
        images = tmp_batch["image"]
        labels = tmp_batch.get("label")
        print(images.shape, labels.shape) if i == 0 else None
        for img, label in zip(images, labels):
            r = i%ROWS
            c = i//ROWS
            d = ax[r, c].imshow(img.numpy()[:,:,[0,1,2]])
            d = ax[r, c].grid(None)
            d = ax[r, c].axis('off')
            d = ax[r, c].set_title("%s" % [i for i, x in enumerate(label.cpu().numpy()) if x != 0])
            i = i + 1
            if i >= ROWS*COLS: break
        if i >= ROWS*COLS: break
    tmp_factory.cleanup()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


# In[ ]:


def check_performances(workers_ = 0):
    print("Check performances")
    augment_ = image_augmentation_train
    m, _, _ = build_model(conf, conf.L_DEVICE, verbose=True)
    tmp_factory = DataFactory_(ALL_IMAGES, conf=conf, verbose=True)
    tmp_dataset = HPADataset(train_pd, tmp_factory, conf, subset="train", verbose=False, augment=augment_, modelprepare=get_preprocessing(m.preprocess_input_fn))
    print("tmp_dataset:", len(tmp_dataset))
    tmp_loader = DataLoader(tmp_dataset, batch_size=16, num_workers=workers_, drop_last = False, pin_memory=False, sampler=None, shuffle=True)  
    i = 0
    for tmp_batch in tqdm(tmp_loader):
        images = tmp_batch["image"]
        labels = tmp_batch["label"]
        if i == 0: print(images.shape, images.dtype, images.max(), labels.shape)
        i = i + 1
    tmp_factory.cleanup()


# In[ ]:


def display_fold(train_pd_, kf_):
    # Check how well the folds are stratified.
    print("fold                                         1    2    3    4    total")
    print("======================================================================")
    for label in class_names:
        label_padded = label + " "*(43-len(label))
        dist = ": "
        for train_idx, valid_idx in kf_.split(train_pd_, train_pd_.iloc[:, LABELS_OHE_START:LABELS_OHE_START+19]):
            X_train, X_valid = train_pd_.iloc[train_idx], train_pd_.iloc[valid_idx]
            dist += "{:4d} ".format(X_valid[label].sum())
        dist += "{:4d} ".format(train_pd_[label].sum())
        print(label_padded + dist)
    label_padded = "total" + " "*(43-len("total"))


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
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--lr', default=0.0003, type=float)    
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--epochs', default=48, type=int)
    parser.add_argument('--workers', default=8 if PT_SCRIPT is True else 0, type=int)
    parser.add_argument('--resume_fold', default=0, type=int)
    parser.add_argument('--stage', default='stage1', type=str, help='stage to train')
    parser.add_argument('--pretrained_stage', default=None, type=none_or_str, help='stage to load pretrained weights from')
    parser.add_argument('--labels_file', default='train_cleaned_default_external.csv', type=str, help='CSV file with labels')
    parser.add_argument('--additional_labels_file', default=None, type=none_or_str, help='Additional CSV file with labels like train_cleaned_2018.csv')
    return parser


# In[ ]:


if __name__ == '__main__':
    
    import os, sys, random, math
    import pandas as pd
    import timeit, os, gc, psutil
    if PT_SCRIPT is False:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    import warnings
    from sklearn import metrics
    from functools import partial
    from collections import OrderedDict
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from PIL import Image
    from ast import literal_eval
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', 250)
    pd.set_option('display.max_rows', 100)    
    import skimage.io
    import skimage.transform
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    DEFAULT_FIG_WIDTH = 28
    sns.set_context("paper", font_scale=1.2)
    import argparse

    print('Python        : ' + sys.version.split('\n')[0])
    print('Numpy         : ' + np.__version__)
    print('Pandas        : ' + pd.__version__)
    print('PyTorch       : ' + torch.__version__)
    print('Albumentations: ' + A.__version__)
    print('Timm          : ' + timm.__version__)
    print('Iterstrat     : ' + iterstrat.__version__)
    
    # Parse arguments
    args = get_argsparser().parse_args() if PT_SCRIPT is True else get_argsparser().parse_args(['--seed', '2020']) #  '--additional_labels_file', 'train_cleaned_2018.csv'
    
    # Fixed seed for reproducibility
    seed = args.seed
    seed_everything(seed)
    
    # All data
    PARTS = ["external"]
    if args.additional_labels_file is not None:
        PARTS = PARTS + ["additional"]
    
    if args.factory == "HDF5":
        ALL_IMAGES = {
            DEFAULT: TRAIN_HOME + 'images_%d.hdf5' % IMAGE_SIZE,
        }
        for p in PARTS:
            ALL_IMAGES[p] = TRAIN_HOME + 'images_%s_%d.hdf5' % (p, IMAGE_SIZE)
        DataFactory_ = HDF5DataFactory
    else:
        ALL_IMAGES = TRAIN_IMAGES_HOME
        DataFactory_ = DataFactory
    print("Factory", DataFactory_, ALL_IMAGES)

    # Override basic configuration
    conf = raw_conf(args.factory)
    conf.mtype = args.mtype
    conf.backbone = args.backbone
    conf.gamma = args.gamma
    conf.lr = args.lr
    conf.BATCH_SIZE = args.batch_size
    conf.EPOCHS = args.epochs
    conf.WORKERS = args.workers
        
    print('Running on device: {}'.format(conf.L_DEVICE))

    MODEL_NAME = "%s_%s_%d_%d_%s_%s%s_v3.0" % (conf.mtype, conf.backbone, IMAGE_SIZE, RESIZED_IMAGE_SIZE if RESIZED_IMAGE_SIZE is not None else IMAGE_SIZE, COMPOSE if COMPOSE is not None else "RGBY", "fp16_" if conf.fp16 is True else "", "CV%d" % conf.FOLDS if conf.FOLDS > 0 else "FULL")
    MODEL_PATH = HOME + "models/" + MODEL_NAME
    STAGE = args.stage
    MODEL_BEST = 'model_best.pt'

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    TRAIN = True
    RESUME_FOLD = args.resume_fold
    RESUME = True
    PRETRAINED = None
    PRETRAINED_STAGE = args.pretrained_stage
    FREEZE_BACKBONE = False
    PRETRAINED_BACKBONE_STAGE = None
    USE_AUG = True
    
    # Load ID, Labels, ...
    train_pd = prepare_data(args.labels_file)
    print("Labels:", args.labels_file, "train_pd:", train_pd.shape)    
    if args.additional_labels_file is not None:
        FILTER = "additional"        
        train_extra_pd = prepare_data(args.additional_labels_file, ext_name = FILTER)
        print("Additional labels:", args.labels_file, "train_extra_pd:", train_extra_pd.shape)
        train_pd = pd.concat([train_pd, train_extra_pd], axis=0).reset_index(drop=True)
        print("Final train_pd:", train_pd.shape)
        
    # F1 metric
    def f1(y_true, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return metrics.f1_score(y_true, y_pred, average='samples')

    # mAP metric
    def mAP(y_true, y_prob):
        score = metrics.average_precision_score(y_true, y_prob, average=None)
        score = np.nan_to_num(score).mean()
        return score
    
    METRIC = f1
    METRIC_NAME = "loss"
    METRICS_PROBS = [f1]

    # Augmentations
    image_augmentation_train = image_harder_augmentation_train(1.0) if USE_AUG is True else None
    
    # Sanity check for model inputs/outputs
    CHECK_MODEL = False
    if CHECK_MODEL is True:
        check_model(conf)
    
    # Plot some inputs for the model
    DISPLAY_PREVIEW = False
    if DISPLAY_PREVIEW:
        display_preview()
    
    # Baseline for all data loading
    CHECK_PERFORMANCE = False
    if CHECK_PERFORMANCE:
        check_performances()        
            
    # Train with CV
    if args.additional_labels_file is None:
        # 89k images
        kf = MultilabelStratifiedKFold(n_splits=conf.FOLDS, shuffle=True, random_state=seed)
        # Display balance
        display_fold(train_pd, kf)
        FOLD = 0
        arguments = []
        for (train_idx, valid_idx) in kf.split(train_pd, train_pd.iloc[:, LABELS_OHE_START:LABELS_OHE_START+19]):  
            FOLD = FOLD + 1
            if FOLD < RESUME_FOLD: continue
            X_train, X_valid = train_pd.iloc[train_idx], train_pd.iloc[valid_idx]     
            print('Fold', FOLD, 'train size:', X_train.shape, 'valid size:', X_valid.shape)
            h_ = run_stage(X_train, X_valid, STAGE, FOLD, conf.L_DEVICE)        
    else: 
        # 89k images + 9k images
        # Split/concat to keep previous CV4 split to avoid leak
        default_pd = train_pd[train_pd[EXT] != FILTER].reset_index(drop=True)        
        external_pd = train_pd[train_pd[EXT] == FILTER].reset_index(drop=True)        
        kf = MultilabelStratifiedKFold(n_splits=conf.FOLDS, shuffle=True, random_state=seed)
        kf_ext = MultilabelStratifiedKFold(n_splits=conf.FOLDS, shuffle=True, random_state=seed)
        
        # Display default balance
        display_fold(default_pd, kf)
        
        # Display additional balance
        display_fold(external_pd, kf_ext)
        
        FOLD = 0
        arguments = []
        for (train_idx, valid_idx), (train_ext_idx, valid_ext_idx) in zip(kf.split(default_pd, default_pd.iloc[:, LABELS_OHE_START:LABELS_OHE_START+19]), 
                                                                          kf_ext.split(external_pd, external_pd.iloc[:, LABELS_OHE_START:LABELS_OHE_START+19])):  
            FOLD = FOLD + 1
            if FOLD < RESUME_FOLD: continue
            X_train, X_valid = default_pd.iloc[train_idx], default_pd.iloc[valid_idx]            
            print('Fold', FOLD, 'train default size:', X_train.shape, 'valid default size:', X_valid.shape)
            X_train_ext, X_valid_ext = external_pd.iloc[train_ext_idx], external_pd.iloc[valid_ext_idx]            
            print('Fold', FOLD, 'train ext size:', X_train_ext.shape, 'valid ext size:', X_valid_ext.shape)            
            X_train = pd.concat([X_train, X_train_ext], axis=0).reset_index(drop=True)
            X_valid = pd.concat([X_valid, X_valid_ext], axis=0).reset_index(drop=True)
            print('Fold', FOLD, 'train size:', X_train.shape, 'valid size:', X_valid.shape)
            h_ = run_stage(X_train, X_valid, STAGE, FOLD, conf.L_DEVICE)

