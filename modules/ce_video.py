#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import cv2

def pre_process(image):
    image = np.array(image)
    image_pre = image[32:544, 32:544, :]
    for i in range(100):
        for j in range(100):
            if i + j > 99:
                pass
            else :
                image_pre[i, j, :] = 0
                image_pre[i, 511 - j, :] = 0
    return image_pre

def load_frame(frame_path, img_ch = 'bgr'):
    frame = cv2.imread(frame_path)
    if img_ch == 'rgb':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pre = pre_process(frame)
    return frame_pre

def find_clip_idxs(preds, n_frame):
    #pred_idx = np.where(np.argmax(preds, axis = 1) == 1)[0]

    clip_idx = []
    for idx in pred_idx:
        clip_idx.append([idx + i - 5 for i in range(11)])
    clip_idx = np.concatenate(clip_idx)
    clip_idx = np.unique(clip_idx)
    idx_start = np.where(clip_idx >= 0)
    idx_end = np.where(clip_idx < n_frame)
    return clip_idx[np.intersect1d(idx_start, idx_end)]

def merging(image, cam):
    cam_pad = np.zeros([576, 576, 3]) # zero padding
    cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    cam_pad[32:544,32:544, :] = cam_bgr
    return np.concatenate([image, cam_pad], axis = 1)

def captioning(image, softmax, model_type = 'binary'):
    if model_type == 'binary':
        cap = "Predict : {}".format(np.argmax(softmax, axis =0))
        cv2.putText(image, cap, (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cap2 = 'Probability: {0:0.2f}%'.format(100*softmax[1])
        cv2.putText(image, cap2, (760, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif model_type == 'ensemble':
        cap = "Predict : {}".format(np.argmax(softmax[:2], axis =0))
        cv2.putText(image, cap, (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cap2 = 'Hem.: {0:0.2f}%, Ulc.: {1:0.2f}%'.format(100*softmax[3], 100*softmax[5])
        cv2.putText(image, cap2, (700, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

