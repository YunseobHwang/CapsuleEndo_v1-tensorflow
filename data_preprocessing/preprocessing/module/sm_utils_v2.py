
# coding: utf-8

# # Import Dependent Modules

# In[1]:


import numpy as np
import os
import cv2


# # Load Data

# In[1]:


def load_data(source, purpose, pathology, dir_default='/mnt/disk1/project/public/SMhospital/Database/1_ongoing/'):
    """Import dataset according to source, purpose and pathology.
    
    Args:
      source: A string, 'sm', 'miccai', or 'sm_core'
      purpose: A string, 'train' or 'test'
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
    
    dir_detail = dir_default + source + '/' + purpose + '/'
    
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
        
    return (data_x, data_y)


# # Equalize Data Size

# In[3]:


def equal(data_x, data_y):
    """Equalize the size of two dataset, negative and positive.
    
    Args:
      data_x: A list, images of two classes
      data_y: A list, corresponding labels
      
    Returns:
      equalized dataset: A tuple, (data_x_picked, data_y_picked)
      
    """
    def pick(data_x, data_y, num):
        num_data = len(data_x)
        picked_idx = set(np.random.choice(num_data, num, replace=False))
        
        picked_x = [data_x[i] for i in picked_idx]
        picked_y = [data_y[i] for i in picked_idx]
        
        return (picked_x, picked_y)
    
    data_size = min([len(data_y[0]), len(data_y[1])])
    data_x_picked = [None]*2
    data_y_picked = [None]*2
    
    for i in range(len(data_x)): # class
        data_x_picked[i], data_y_picked[i] = pick(data_x[i], data_y[i], data_size)
    
    return (data_x_picked, data_y_picked)


# # Split Train Data into Train (75%) / Validation (25%)

# In[4]:


def split(data_x, data_y):
    """Split the train data into train data (75%) and validation data (25%).
    
    Args:
      data_x: A list, images of two classes
      data_y: A list, corresponding labels
      
    Returns:
      Train and Validation dataset: A tuple, (train_x, train_y, valid_x, valid_y)
      
    """
    def sep(data_x, data_y):
        num_data = len(data_x)
        num_train_data = int(num_data*0.75)

        train_idx = set(np.random.choice(num_data, num_train_data, replace=False))
        valid_idx = set(np.arange(len(data_x))) - train_idx
        
        train_x = [data_x[i] for i in train_idx]
        train_y = [data_y[i] for i in train_idx]
        valid_x = [data_x[i] for i in valid_idx]
        valid_y = [data_y[i] for i in valid_idx]
        
        return (train_x, train_y, valid_x, valid_y)
    
    train_x = [None]*2
    train_y = [None]*2
    valid_x = [None]*2
    valid_y = [None]*2
    
    for i in range(len(data_x)): # class
        train_x[i], train_y[i], valid_x[i], valid_y[i] = sep(data_x[i], data_y[i])
        
    return (train_x, train_y, valid_x, valid_y)


# # Batchmaker

# In[5]:


def batch(target_x, target_y, batchsize):
    """Make batch according to purpose and batchsize from target_x and target_y.
       The output batchsize will be double of input parameter because of the number of classes.
    
    Args:
      target_x: A list, list of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch: A tuple, (batch_x, batch_y)
      
    """
    batch_idx = np.random.choice(len(target_x[0]), batchsize, replace=False)
        
    batch_x = []
    batch_y = []
        
    for i in batch_idx:
        for j in range(len(target_x)): # class
            batch_x.append(target_x[j][i])
            batch_y.append(target_y[j][i])
        
    return (np.array(batch_x, dtype = 'f4'), np.array(batch_y, dtype = 'f4'))

