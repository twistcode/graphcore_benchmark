# !/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023, TWISTCODE® TECHNOLOGIES SDN BHD. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

__author__ = "Amir Fawwaz"
__copyright__ = "Copyright 2023, Twistcode AI Team"
__credits__ = ["Nurazam Malim"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Amir Fawwaz"
__email__ = "fawwwaz@twistcode.com"
__status__ = "Development"


import numpy as np
from numpy import save
import pandas as pd
import pickle
from termcolor import colored
from pathlib import Path
from tqdm import tqdm
from glob import glob
from IPython.display import clear_output, display_html
import time
import os
import pydicom
import cv2
import logging


# for logging
# let's try to move away from print
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    '''Load each of the datasets we are given.'''
    
    data_dir = Path("rsna-breast-cancer-detection")
    dir_up_ = Path(data_dir).parents[0]
    train = pd.read_csv(dir_up_ / "train.csv")
    test = pd.read_csv(dir_up_ / "test.csv")
    sample_submission = pd.read_csv(dir_up_ / 'sample_submission.csv')
    # return train, test
    return train, test, sample_submission


def data_info(csv, name="Train"):
    '''Prints basic information about the datasets we are given.'''
    '''Inspired by: https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detection-dicom-images-explore'''
    
    print(colored('==== {} ===='.format(name), 'cyan', attrs=['bold']))
    print(colored('Shape: ', 'cyan', attrs=['bold']), csv.shape)
    print(colored('NaN Values: ', 'cyan', attrs=['bold']), csv.isnull().sum().sum(), '\n')
    #print(colored('Columns: ', 'blue', attrs=['bold']), list(csv.columns))
    
    display_html(csv.head())
    if name != 'Sample Submission': print("\n")


train, test, sample_submission = load_data()
clear_output()


names = ["Train", "Test", "Sample Submission"]
for i, df in enumerate([train, test, sample_submission]): 
    data_info(df, names[i])

df1 = train[train['cancer']==0]
df2 = train[train['cancer']==1]
logger.info("Length of healthy dataframe %s", len(df1))
logger.info("Length of cancer dataframe %s", len(df2))

# convert dataframe value to list
healthy_patient_id_list = df1['patient_id'].tolist()
cancer_patient_id_list = df2['patient_id'].tolist()

# dump our list to the pickle
with open ('cancer_list.pickle', 'wb') as pick:
    pickle.dump(cancer_patient_id_list, pick)
    
with open ('healthy_list.pickle', 'wb') as pick:
    pickle.dump(healthy_patient_id_list, pick)

# now, let's get clean list (no repeat id)
healthy_patient_id_list_clean = list(dict.fromkeys(healthy_patient_id_list))
cancer_patient_id_list_clean = list(dict.fromkeys(cancer_patient_id_list))
logger.info("Length of cancer list %s", len(cancer_patient_id_list_clean))
logger.info("Length of healthy list %s", len(healthy_patient_id_list_clean))

## Now, let's divide the original training_images to 2 class folder format
healthy_paths = []
cancer_paths =  []

for f in tqdm(glob('rsna-breast-cancer-detection/cancer_paths/*')):
    id = f.split('/')[1] # this may change in linux
    cancer_paths.append(id)

for f in tqdm(glob('rsna-breast-cancer-detection/healthy_paths/*')):
    id = f.split('/')[1] # this may change in linux
    healthy_paths.append(id)

if len(cancer_patient_id_list_clean)==len(cancer_paths) and len(healthy_patient_id_list_clean)==len(healthy_paths):
    logger.warning("all looks good")
else:
    logger.warning("check balik woii!!")


healthy_paths = 'rsna-breast-cancer-detection/healthy_paths'
cancer_paths = 'rsna-breast-cancer-detection/cancer_paths/'

filelist_healthy = []
filelist_cancer = []

for root, dirs, files in os.walk(healthy_paths):
	for file in files:
        #append the file name to the list
		filelist_healthy.append(os.path.join(root,file))

for root, dirs, files in os.walk(cancer_paths):
	for file in files:
        #append the file name to the list
		filelist_cancer.append(os.path.join(root,file))
        

## Now let's prepare the dataset for Tensorflow!!!
##

start = time.time()
X_train = []
Y_train = []
        
#print all the file names
for name in filelist_healthy:
    # print(name)
    ds = pydicom.dcmread(name)
    pixel_array_numpy = ds.pixel_array
    pixel_array_numpy = cv2.resize(pixel_array_numpy, dsize=(256, 256))
    
    X_train.append(pixel_array_numpy)
    Y_train.append(0)
    # print(len(X_train))
    # plt.imshow(pixel_array_numpy, cmap='gray')
    #plt.show()
    

#print all the file names
for name_ in filelist_cancer:
    # print(name)
    ds = pydicom.dcmread(name_)
    pixel_array_numpy = ds.pixel_array
    pixel_array_numpy = cv2.resize(pixel_array_numpy, dsize=(256, 256))
    
    X_train.append(pixel_array_numpy)
    Y_train.append(1)
    # print(len(X_train))
    # plt.imshow(pixel_array_numpy, cmap='gray')
    #plt.show()

logger.info("It took {} seconds to covert dicom to numpy array".format(time.time() - start)) 

# save data to numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)
logger.info("Check X_train array %s, Y_train array %s", X_train.shape,Y_train.shape)

## save dataset to numpy array for later use
save('X_train.npy', X_train)
save('Y_train.npy', Y_train)

logger.info("Data Conversion Finish!!")