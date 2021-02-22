
# coding: utf-8

# # Import Dependent Modules

# In[1]:


import numpy as np
import os
import cv2


# # Load Data

# In[1]:


def load_data(source, model, datatype, pathology, dir_default = '/mnt/disk1/project/public/SMhospital/Database/1_ongoing/'):
    """Import dataset according to source, purpose and pathology.
    
    Args:
      source: A string, 'sm', 'miccai', or 'sm_core'
      model: A string, 'nh' or 'nd'
      datatype: A string, 'train', 'test', 'total' or 'total_aug'
      pathology: A string, 'hemorrhagic' or 'depressed'
      dir_data: A string, 
      
    Returns:
      data_list: A tuple, (data_x, data_y)
      
    """
    cls = ['negative']
    cls.append(pathology)
    lesion = {'negative' : ['negative'], 
              'hemorrhagic': ['red_spot', 'angioectasia', 'active_bleeding'],
              'depressed': ['erosion', 'ulcer', 'stricture'],
              'protruded': ['ampulla_of_vater', 'lymphoid_follicles', 'small_bowel_tumor']}

    dir_detail = dir_default + source + model + '/' + datatype

    data_x = [None]*2
    for i in cls:
        data_x[cls.index(i)] = []
        for j in lesion[i]:
            dir_folder_temp = dir_detail + i + '/' + j
            dir_temp = os.listdir(dir_folder_temp)
            for k in dir_temp:
                temp = cv2.imread(dir_folder_temp + '/' + k)
                temp_rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                data_x[cls.index(i)].append(temp_rgb)

    data_y = [None]*2
    for i in cls:
        temp = np.zeros(shape=(len(data_x[cls.index(i)]), len(data_x)))
        temp[:, cls.index(i)] = 1
        data_y[cls.index(i)] = list(temp)
        
    return data_x, data_y


def batch(target_x, target_y, batchsize):
    """
    Args:
      target_x: A numpy
      batchsize: A integer
      
    Returns:
      batch: A tuple, (batch_x, batch_y)
    """
    batch_idx = np.random.choice(len(target_x), batchsize, replace=False)
    return target_x[batch_idx], target_y[batch_idx]
