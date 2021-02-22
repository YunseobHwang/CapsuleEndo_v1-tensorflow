# CapsuleEndo_v1-tensorflow

CNN-based classification and XAI-based localization

Automate lesion localization using Grad-CAM as XAI, which is also known as weakly-supervised object localization 

Hwang, Yunseob, et al. "Improved classification and localization approach to small bowel capsule endoscopy using convolutional neural network." Digestive Endoscopy (2020).

## Small Bowel Capsule Endocsopy Images by Small Bowel Lesions

![SBCE](./images/SBCE_images_by_lesion.jpg)

## Data Preprocessing and Augmentation

<p align="center">
     <b> Preprocessing for SBCE images</b> <br>
     <img alt="SBCE_PRE" src="./images/preprocessing_for_SBCE_images.jpg"
          width=80% />
</p>
<br>

<p align="center">
     <b> Flip and Rotation Augmentation</b> <br>
     <img alt="SBCE_AUG" src="./images/flip_and_rotation_augmentation.jpg"
          width=80% />
</p>
<br>

## Network Fusion

- To boost sensitivity
- To achieve better localization ability through a visual explanation for AI's decision >> Make it more reliable 

### Method

![NET_FUS](./images/network_fusion.jpg)
![CAM_FUS](./images/CAM_fusion.jpg)

### Results

![ROC](./images/ROC_curves.jpg)

![GradCAM](./images/GradCAM_comparison.jpg)

![Feature_V](./images/feature_visualization_t-SNE.jpg)
