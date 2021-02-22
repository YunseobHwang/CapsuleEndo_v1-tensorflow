#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.metrics import confusion_matrix
import itertools
import datetime
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders
import sys

def load_data(phase, cls, les = None, data = 'sm', data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed', 
              extract_name = False, image_ch = 'bgr'):
    """
    phase = 'train', 'test'
    cls: [les]  
      'n': ['neg']
      'h': ['redspot', 'angio', 'active'], 
      'd': ['ero', 'ulc', 'str'],
      'p': ['amp', 'lym', 'tum']}
    """
    lesions = dict(neg = 'negative', 
                   redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                   ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                   amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
    classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')

    path = os.path.join(data_dir, data, phase, classes[cls], lesions[les])
    pathlist = glob.glob(path + '/*.jpg')

    return load_image_from_path(pathlist, image_ch = image_ch, extract_name = extract_name)

def load_image_from_path(pathlist,image_ch = 'bgr', extract_name = False):
    data = []
    for i in pathlist:
        temp = cv2.imread(i)
        if image_ch == 'bgr':
            pass
        elif image_ch == 'rgb':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        elif image_ch == 'hsv':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        data.append(temp)
    if extract_name != False:
        name = []
        for i in pathlist:
            name.append(os.path.basename(i))
        return np.asarray(data), np.asarray(name)
    else:
        return np.asarray(data) 

def load_path(phase, cls, les = None, data = 'sm', data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'):
    """
    phase = 'train', 'test'
    cls: [les]  
      'n': ['neg']
      'h': ['redspot', 'angio', 'active'], 
      'd': ['ero', 'ulc', 'str'],
      'p': ['amp', 'lym', 'tum']}
    """
    lesions = dict(neg = 'negative', 
                   redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                   ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                   amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
    classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')

    path = os.path.join(data_dir, data, phase, classes[cls], lesions[les])
    pathlist = glob.glob(path + '/*.jpg')
    return np.asarray(pathlist)
    
    
def target_preprocessings(phase_a_switch = [1, 1, 1], phase_b_switch = True, mode = 'load'):
    """
    phase_a_switch = [1, 1, 1], [0, 0 ,1], [1, 1, 0].... 
    that means [flip, rotate, blur_sharp]
    """
    phase0 = ['_c']
    phase1 = {1: ['-', 'f'], 0: ['-']}
    phase2 = {1: ['-', 'r1', 'r2', 'r3'], 0: ['-']}
    phase3 = {1: ['-', 'ab', 'mb', 'eh'], 0: ['-']}
    phase4 = ['s_-30_v_30', 's_-30_v_-30', 's_30_v_-30', 's_30_v_30']
    
    if mode == 'load':
        phase_a_items = [phase1[phase_a_switch[0]], phase2[phase_a_switch[1]], phase3[phase_a_switch[2]]]
    elif mode == 'preprocessing':
        phase_a_items = [phase0, phase1[phase_a_switch[0]], phase2[phase_a_switch[1]], phase3[phase_a_switch[2]]]
    
    phase_a = []
    for i in list(product(*phase_a_items)):
        phase_a.append('_'.join(i))

    if not phase_b_switch != True:
        phase_b = []
        for i in list(product(*[phase_a, phase4])):
            phase_b.append('_'.join(i))
        return list(np.hstack([phase_a, phase_b]))
    else:
        return phase_a 

class ce_x160_load:
    def __init__(self, phase, data, pre_a, pre_b, img_ch = 'bgr', ext_name = True):
        self.phase = phase        # 'train' or 'test'
        self.data = data          # 'sm', 'sm_core', 'sm_v2', 'sm_x160', ...
        self.pre_a = pre_a        # [1, 1, 1], [0, 0 ,1], [1, 1, 0].... 
        self.pre_b = pre_b        # True or False
        self.img_ch = img_ch      # 'bgr', 'rgb', and 'hsv'
        self.ext_name = ext_name  # True or False

    def load_path(self, cls, les, data_dir = '/mnt/disk2/data/private_data/SMhospital/capsule/1 preprocessed'):
        """
        phase = 'train', 'test'
        cls: [les]  
          'n': ['neg']
          'h': ['redspot', 'angio', 'active'], 
          'd': ['ero', 'ulc', 'str'],
          'p': ['amp', 'lym', 'tum']}
        pre_a[0] must be 0
        """
        lesions = dict(neg = 'negative', 
                       redspot = 'red_spot', angio = 'angioectasia', active = 'active_bleeding', 
                       ero = 'erosion', ulcer = 'ulcer', str = 'stricture', 
                       amp = 'ampulla_of_vater', lym = 'lymphoid_follicles', tum = 'small_bowel_tumor')
        classes = dict(n = 'negative', h = 'hemorrhagic', d = 'depressed', p = 'protruded')

        path = os.path.join(data_dir, self.data, self.phase, classes[cls], lesions[les])
        pathlist = glob.glob(path + '/*.jpg')
        if self.pre_a != [1, 1, 1] and self.pre_b != True:
            path_in_phase = []
            for p in pathlist:
                name = os.path.basename(p)
                if (name.split('c_')[-1])[:-4] in target_preprocessings(self.pre_a, self.pre_b):
                    path_in_phase.append(p)   
            return np.asarray(path_in_phase)
        else:
            return np.asarray(pathlist)

    def load_image_from_path(self, pathlist,image_ch = 'bgr', extract_name = False):
        data = []
        for i in pathlist:
            temp = cv2.imread(i)
            if image_ch == 'bgr':
                pass
            elif image_ch == 'rgb':
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            elif image_ch == 'hsv':
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
            data.append(temp)
        if extract_name != False:
            name = []
            for i in pathlist:
                name.append(os.path.basename(i))
            return np.asarray(data), np.asarray(name)
        else:
            return np.asarray(data) 

    def load_data(self, cls, les):
        pathlist = self.load_path(cls, les)
        return  self.load_image_from_path(pathlist, image_ch = self.img_ch, extract_name = self.ext_name)

    
def one_hot(data, cls, n_cls = 2):
    
    one_hot = [0]*n_cls
    one_hot[cls] = 1
    
    label = np.vstack([one_hot for i in range(len(data))])

    return label

def train_valid_split(x, y = None, train_rate = 0.85):
    t_idx = np.random.choice(len(x), int(len(x)*train_rate), replace = False)
    v_idx = np.setdiff1d(np.arange(len(t_idx)), t_idx)
    if not y == None:
        return x[t_idx], y[t_idx], x[v_idx], y[v_idx]
    else:
        return x[t_idx], x[v_idx]
    
def random_minibatch(x, y, batch_size = 50):
    idx = np.random.choice(len(x), batch_size)
    return x[idx], y[idx]

def load_random_minibatch(pathlist, cls, n_cls = 2, batch_size = 50, image_ch = 'bgr'):
    idx = np.random.choice(len(pathlist), batch_size)
    batch_dir = pathlist[idx]
    batch_x = []
    for i in batch_dir:
        temp = cv2.imread(i)
        if image_ch == 'bgr':
            pass
        elif image_ch == 'rgb':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        elif image_ch == 'hsv':
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        batch_x.append(temp)
    batch_label = one_hot(batch_x, cls, n_cls)
    return np.asarray(batch_x), np.asarray(batch_label)

def shuffle(x, y):
    """
    random shuffle of two paired data -> x, y = shuffle(x, y)
    but, available of one data -> x = shuffle(x, None)
    """
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    if type(x) == type(y):
        return x[idx], y[idx] 
    else:
        return x[idx]
    
def sinusoidal_exp_LR(n_iter, f = 20, A = 5e-4, alpha = 0.001):
    """
    f: frequency, 
    A: intial value of learning rate, 
    alpha: attenuation constant of exponential
    """
    xp = np.arange(0, n_iter) 
    threshold = 0.1*A
    LR = A*(np.exp(- alpha*xp)+0.2*np.cos(f*2*np.pi/n_iter*xp)**2) + threshold 
    return LR

def damped_exp_LR(n_iter, f = 20, A = 5e-4, alpha = 0.0002, gamma = 0.1):
    """
    f: frequency, 
    A: intial value of learning rate, 
    alpha: attenuation constant of exponential, 
    gamma: attenuation constant of cosine in exp.
    """
    xp = np.arange(0, n_iter)
    threshold = 0.1*A 
    exp = np.exp(- alpha*xp)
    damped_comp = np.cos(f*2*np.pi/n_iter*xp)**2
    LR = A*exp*(1 + gamma*damped_comp) + threshold
    return LR

class training_history:
    def __init__(self, accr_train, accr_valid, loss_train, loss_valid):
        self.accr_train = accr_train
        self.accr_valid = accr_valid
        self.loss_train = loss_train
        self.loss_valid = loss_valid
    def table(self):
        print('==============================================================')
        print('[Iter] || Train_accr || Valid_accr || Train_loss || Valid_loss')
        print('==============================================================')
    def evl(self, n_iter):
        evl = '[{:*>4d}] || {:*>.2f} %    || {:*>.2f} %    || {:.8f} || {:.8f}'.format(n_iter, 
                                                                                      self.accr_train[-1]*100, self.accr_valid[-1]*100, 
                                                                                      self.loss_train[-1], self.loss_valid[-1])
        return evl
    def prt_evl(self, n_iter):
        print(self.evl(n_iter))
        print('--------------------------------------------------------------')
    def early_under(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping - Underffiting !!]\n')
    def early_over(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping - Overffiting !!]\n')
    def early(self, n_iter):
        print(self.evl(n_iter) + ' [Early stopping]\n')
    def done(self, n_iter, train_time, early_stopping):  
        global training_name
        global contents
        global filename
        global title
        
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%y%m%d%H%M')
        
        contents = (
        'Training Time : {} Min.\n'.format(train_time) +
        'Early Stopping : {}\n'.format(early_stopping) +
        'Iteration : {}\n'.format(n_iter)
        )
        print(contents)

        title = 'Training History'
    def plot(self, n_cal, save = False, path = './history', filename = 'history.png'):
        fig = plt.figure(figsize = (15,20))
        plt.suptitle('Training History', y = 0.92, fontsize = 20)

#         x_axis = range(1, len(self.accr_train)+1)
        x_axis = range(1, n_cal*len(self.accr_train)+1, n_cal)
        x_axis = np.arange(n_cal, n_cal*len(self.accr_train)+1, n_cal)

        plt.subplot(2, 1, 1)
        plt.plot(x_axis, self.accr_train, 'b-', label = 'Training Accuracy')
        plt.plot(x_axis, self.accr_valid, 'r-', label = 'Validation Accuracy')
        plt.xlabel('n_iter', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.legend(fontsize = 10)
        plt.subplot(2, 1, 2)
        plt.plot(x_axis, self.loss_train, 'b-', label = 'Training Loss')
        plt.plot(x_axis, self.loss_valid, 'r-', label = 'Validation Loss')
        plt.xlabel('n_iter', fontsize = 15)
        plt.ylabel('Loss', fontsize = 15)
    #     plt.yticks(np.arange(0, 0.25, step=0.025))
        plt.legend(fontsize = 12)
        plt.show()
        
        if save == True:
            fig.savefig(hist_path + filename)
            plt.close(fig)
            
    def email(path = './history', filename = 'history.png'):

        Login_email = 'yunseob1102@gmail.com'
        App_pw = 'dkizbkgonasswrua'
        from_email = 'yunseob1102@gmail.com'
        to_email = 'hys1102@postech.ac.kr'

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(Login_email, App_pw) 

        # msg = MIMEText(Title)
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = msg['Subject'] = title
        msg.attach(MIMEText(contents, 'plain'))
        attachment = open(hist_path + filename, 'rb')
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(p)

        s.sendmail(from_email, to_email, msg.as_string())
        s.quit()
    
    
def test_batch_idxs(data, batch_size = 75):
    """generate the serial batch of data on index-level.
       Usually, the data is too large to be evaluated at once.
    
    Args:
      data: A list or array of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch_idxs: A list, 
    """
    total_size = len(data)
    batch_idxs = []
    start = 0
    while True:
        if total_size >= start + batch_size:
            batch_idxs.append([start + i for i in range(batch_size)])
        elif total_size < start + batch_size:
            batch_idxs.append([start + i for i in range(total_size - start)])
        start += batch_size
        if total_size <= start:
            break
    return batch_idxs

def model_prob(model, image, model_type = 'binary'):
    b_idxs = test_batch_idxs(image)
    if model_type == 'ensemble':
        e_outputs, nh_outputs, nd_outputs = [], [], []
        start_time = time.time()
        for b_idx in b_idxs:
            e_softmax, nh_softmax, nd_softmax = model.get_softmax(image[b_idx])
            e_outputs.append(e_softmax), nh_outputs.append(nh_softmax), nd_outputs.append(nd_softmax)
        time_taken = time.time() - start_time
        print("{} / Inference Time: {}".format(len(image), time.strftime("%H:%M:%S", time.gmtime(time_taken))))
        return np.concatenate(e_outputs), np.concatenate(nh_outputs), np.concatenate(nd_outputs)
    else:
        outputs = []
        start_time = time.time()
        for b_idx in b_idxs:
            softmax = model.get_softmax(image[b_idx])
            outputs.append(softmax)
        time_taken = time.time() - start_time
        print("{} / Inference Time: {}".format(len(image), time.strftime("%H:%M:%S", time.gmtime(time_taken))))
        return np.concatenate(outputs)
    
def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    

def Sec_to_M_S_ms(sec):
    min_sec = time.strftime("%M:%S", time.gmtime(sec))
    ms = '{:03d}'.format(int((sec - int(sec))*1000))   
    return '.'.join([min_sec, ms])

def model_pred_and_time(model, image, model_type = 'binary'):
    b_idxs = test_batch_idxs(image)
    if model_type == 'ensemble':
        e_outputs, nh_outputs, nd_outputs = [], [], []
        start_time = time.time()
        for i, b_idx in enumerate(b_idxs):
            e_softmax, nh_softmax, nd_softmax = model.get_softmax(image[b_idx])
            e_outputs.append(e_softmax), nh_outputs.append(nh_softmax), nd_outputs.append(nd_softmax)
            printProgress(i+1, len(b_idxs), barLength = 80,
                          prefix = '# of batch: {}'.format(len(b_idxs)), 
                          suffix = 'model prediction ({})'.format(model_type[0]))
        time_taken = time.time() - start_time
        time_taken = Sec_to_M_S_ms(time_taken)
        return np.concatenate(e_outputs), np.concatenate(nh_outputs), np.concatenate(nd_outputs), time_taken
    else:
        outputs = []
        start_time = time.time()
        for i, b_idx in enumerate(b_idxs):
            softmax = model.get_softmax(image[b_idx])
            outputs.append(softmax)
            printProgress(i+1, len(b_idxs), barLength = 80,
                          prefix = '# of batch: {}'.format(len(b_idxs)), 
                          suffix = 'model prediction ({})'.format(model_type[0]))
        time_taken = time.time() - start_time
        time_taken = Sec_to_M_S_ms(time_taken)
        return np.concatenate(outputs), time_taken

class classification_metric:
    def accuracy(self, data_y, outputs, lesion):
        pred = np.argmax(outputs, axis = 1)
        true = np.argmax(data_y, axis = 1)
        acc = 100*np.mean(np.equal(true, pred))
        print("{}: {:.2f} %".format(lesion, acc))
        return acc

    def con_mat(self, data_y, outputs):
        pred = np.argmax(outputs, axis = 1)
        true = np.argmax(data_y, axis = 1)
        return confusion_matrix(true, pred)

    def plot_cm(self, cm, value_size = 15, label_size = 10, mode = 'percent'):
        plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if mode == 'percent':
                value = np.round(cm[i, j]/(np.sum(cm, 1)[i]), 3)
            if mode == 'num':
                value = cm[i, j]
            plt.text(j, i, value,
                     fontsize = value_size,
                     horizontalalignment = 'center',
                     color = 'white' if cm[i, j] > thresh else 'black')
        plt.ylabel('True label', fontsize = label_size)
        plt.xlabel('Predicted', fontsize = label_size)
        plt.xticks([0, 1], ['nor', 'abnor'], rotation=0, fontsize = label_size)
        plt.yticks([0, 1], ['nor', 'abnor'], rotation=90, fontsize = label_size)

    def cm2metric(self, confusion_matrix):
        """
        A confusion matrix for Binary classification
        2x2 array: [[TN, FN],[FP, TP]]
        """
        Acc = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
        Sen = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
        Spec = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
        NPV = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        PPV = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
        return [Acc, Sen, Spec, NPV, PPV]

    def confusion_idx(self, data_y, outputs, cls = 'positive'):
        pred = np.argmax(outputs, axis = 1)
        true = np.argmax(data_y, axis = 1)
        if cls == 'positive':
            fn_idx = np.where(true - pred == 1)[0]
            tp_idx = np.setxor1d(np.where(true == 1), fn_idx)
            return tp_idx, fn_idx
        elif cls == 'negative':
            fp_idx = np.where(true - pred == -1)[0]
            tn_idx = np.setxor1d(np.where(true == 0), fp_idx)
            return tn_idx, fp_idx

