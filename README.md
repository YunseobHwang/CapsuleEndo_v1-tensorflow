# CapsuleEndo_v1-tensorflow

CNN-based classification and XAI-based localization

Automate lesion localization using Grad-CAM as XAI, which is also known as weakly-supervised object localization 

Hwang, Yunseob, et al. "Improved classification and localization approach to small bowel capsule endoscopy using convolutional neural network." Digestive Endoscopy (2020).

## Small Bowel Capsule Endocsopy Images by Small Bowel Lesions

![SBCE](./images/SBCE_images_by_lesion.jpg)

## Data Preprocessing and Augmentation

![SBCE_PRE](./images/preprocessing_for_SBCE_images.jpg)
![SBCE_AUG](./images/flip_and_rotation_augmentation.jpg)

## Network Fusion

- To boost sensitivity
- To enable AI (CNN) localization ability using CAM 

### Method

![NET_FUS](./images/network_fusion.jpg)
![CAM_FUS](./images/CAM_fusion.jpg)

### Results

![ROC](./images/ROC_curves.jpg){: width="100" height="100"}

![GradCAM](./images/GradCAM_comparison.jpg)

![Feature_V](./images/feature_visualization_t-SNE.jpg)
