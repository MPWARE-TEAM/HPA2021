#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load RGBY images, resize to 512x512 and dump to HDF5
# dataset["image_id"] = (512, 512, 4) numpy array [0-1.0] float32
__author__ = 'MPWARE: https://www.kaggle.com/mpware'


# In[ ]:


import sys, os
import numpy as np
import skimage.transform
from PIL import Image
import shutil
import skimage.io
import h5py
import argparse
from tqdm import tqdm


# In[ ]:


print('Python       : ' + sys.version.split('\n')[0])
print('H5Py         : ' + h5py.__version__)


# In[ ]:


def get_argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--images_folder', type=str)
    parser.add_argument('--hdf5_output', type=str)
    return parser


# In[ ]:


def create_hdf5_data(hdf5_file, in_folder, size = 512):    
    
    # Use PIL to support 16 bits, normalize=True to return [0-1.0] float32 image
    def read_image(filename, normalize=True, norm_value=65535.0, images_root=""):
        filename = images_root + filename
        filename = filename + "_red.png" if "_red.png" not in filename else filename
        mt_, pi_, nu_, er_ = filename, filename.replace('_red', '_green'), filename.replace('_red', '_blue'), filename.replace('_red', '_yellow')        
        ret = None
        try:
            if os.path.exists(mt_) and os.path.exists(pi_) and os.path.exists(nu_) and os.path.exists(er_):            
                mt = np.asarray(Image.open(mt_)).astype(np.uint16)
                pi = np.asarray(Image.open(pi_)).astype(np.uint16)
                nu = np.asarray(Image.open(nu_)).astype(np.uint16)  
                er = np.asarray(Image.open(er_)).astype(np.uint16)
                ret = np.dstack((mt, pi, nu, er)) # RGBY
                if normalize is True:
                    if ret.max() > 255:
                        ret = (ret/norm_value).astype(np.float32)
                    else:
                        ret = (ret/255).astype(np.float32)
            else:
                print("Mising layer:", filename)                
        except Exception as ex:
            print("Cannot read image", filename, ex)
        return ret

    h5_file_image = h5py.File(hdf5_file, 'w')
    print("Saving to %s" % hdf5_file, "and resizing to %dx%d" % (size, size))
    
    # Image IDs to package
    filelist = [x for x in os.listdir(in_folder) if '_red.png' in x]
    filelist = [x[:-8] for x in filelist]

    for uid in tqdm(filelist):
        full_image = read_image(uid, images_root = in_folder)
        if full_image is not None:
            img = skimage.transform.resize(full_image, (size, size), anti_aliasing=True) # Works with float image
            h5_file_image.create_dataset(uid, data=img)
    
    h5_file_image.close()


# In[ ]:


# args = get_argsparser().parse_args(['--images_folder', 'D:/tmp/HPA/train/images/', '--hdf5_output', 'D:/tmp/HPA/deleteme/images_additional_512.hdf5'])
args = get_argsparser().parse_args()
image_size = args.image_size
input_folder = args.images_folder
output_file = args.hdf5_output

# Create output folder if needed
out_folder = os.path.dirname(output_file)
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


# In[ ]:


# Dump images to HDF5 
create_hdf5_data(output_file, input_folder, size=image_size)


# In[ ]:


# DATA_HOME = "./data/"
# TRAIN_HOME =  DATA_HOME + "train/"
# OUT_FOLDER = "./dumps/"


# In[ ]:


# # Dump train set to HDF5 
# create_hdf5_data(OUT_FOLDER + 'images_%d.hdf5' % IMAGE_SIZE, TRAIN_HOME + "images/", size=IMAGE_SIZE)


# In[ ]:


# # Dump public external to HDF5 
# create_hdf5_data(OUT_FOLDER + 'images_external_%d.hdf5' % IMAGE_SIZE, TRAIN_HOME + "external/", size=IMAGE_SIZE)


# In[ ]:


# # Dump HPA 2018 to HDF5 
# create_hdf5_data(OUT_FOLDER + 'images_additional_%d.hdf5' % IMAGE_SIZE, TRAIN_HOME + "additional/", size=IMAGE_SIZE)

