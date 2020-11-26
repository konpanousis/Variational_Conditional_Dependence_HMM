"""
Various functions to read the data.

@author: Anonymous
"""

import numpy as np
from scipy import io
import os
import sys
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET

g3d_tr_subjs = [1,3,5,7,9]
g3d_te_subjs = [2,4,6,8,10]

msra_tr_subjs = [1,3,5,7,9]
msra_te_subjs =  [2,4,6,8,10]

utd_tr_subjs = [1,3,5,7]
utd_te_subjs = [2,4,6,8]


def load_action_ready(act, dataset = 'g3d', portion = None):
    """
    Read the provided action, dataset and portion from mat files.

    :param act: the action to read, int.
    :param dataset:  the dataset, string.
    :param portion: the portion of missing values, string/

    :return: train data np.array, test data np.array
    """

    if portion is None:
        filename_train = 'data/'+str(dataset)+'/data_train.mat'
        true_labels_train = 'data/'+str(dataset)+'/true_labels_train.mat'
        true_labels_test = 'data/'+str(dataset)+'/true_labels_test.mat'
        filename_test = 'data/'+str(dataset)+'/data_test.mat'
    else:
        filename_train = 'data/' + str(dataset) + '/'+str(dataset)+'_data_train_missing_'+str(portion)+'.mat'
        true_labels_train = 'data/' + str(dataset) + '/true_labels_train.mat'
        true_labels_test = 'data/' + str(dataset) + '/true_labels_test.mat'
        filename_test = 'data/' + str(dataset) + '/'+str(dataset)+'_data_test_missing_'+str(portion)+'.mat'

    if os.path.exists(filename_train) and os.path.exists(true_labels_train):
        mat = io.loadmat(filename_train)['data_train']
        labels = io.loadmat(true_labels_train)['true_labels_train']
        train = mat[np.where(labels==act)]
    else:
        print('File', filename_train, 'does not exist in path.')
        train = np.array([])

    if os.path.exists(filename_test) and os.path.exists(true_labels_test):
        mat = io.loadmat(filename_test)['data_test']
        labels = io.loadmat(true_labels_test)['true_labels_test']
        test = mat[np.where(labels==act)]
    else:
        print('File', filename_train, 'does not exist in path.')
        test = np.array([])

    return train, test


def read_g3d_raw():
    """
    Read the g3d dataset from raw data and save to file.

    :return: none
    """
    act = []
    for subject in golf:
        cur_subj = []
        for trial in subject:
            dir = 'data/Golf/KinectOutput'+str(trial)+'/Skeleton/'
            trial_act = []

            files = sorted(os.listdir(dir), key = len)
            for filename in files :
                cur_time_frame = []
                root = ET.parse(dir+filename).getroot()

                for type_tag in root.findall('Skeleton/Joints/Joint/Position'):
                    x = float(type_tag.find('X').text)
                    y = float(type_tag.find('Y').text)
                    z = float(type_tag.find('Z').text)
                    cur_time_frame.extend([x,y,z])
                if cur_time_frame:
                    trial_act.append(cur_time_frame)

            cur_subj.append(np.array(trial_act))
        act.append(np.array(cur_subj))

    act = np.array(act)

    np.save('golf', act )
    return act
