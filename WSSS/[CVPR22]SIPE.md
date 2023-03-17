# [CVPR22] Self-suprvised Imag-specific Prototype Exploration for Weakly Supervised Semantic Segmentation
github: https://github.com/chenqi1126/SIPE

# 1. Introduction
기존의 CAM을 사용한 method들은 classifier를 학습시키고 학습된 weight을 각 class의 general representation (class center) 으로 간주하고 이용한다.   
4개의 image에 대해 pixel level feature를 visualize한 결과 class center는 close pixel을 high activation을 주고 disant pixel을 무시한다는 것을 확인할 수 있다.  
그러나 각 image의 cnetroid는 can be beneficial to explore more complete regions.  
Aims to tailor image-specific prototype to adaptively describe the image itself.  

Self supervised Image specific Prototype Exploration SIPE  
SIPE은 Image-specific Prototype Exploration(IPE)와 General-Specific Consistency (GSC) loss로 이루어짐. 

IPE는 먼저 utilize inter-pixel semantics to explore spatial structure cues, locating robust seed regions of each class.  
seed regions으로 extract image-specific prototypes and then produce IS-CAM by prototypical correlation.  

GSC: construct the consistency of general CAM and specific IS-CAM.  

# 3 Approach
## 3.1 Class Activation Map
class 별 forground activation map을 계산 -> Mf  
Mf기반으로 background activation map Mb계산. confusion을 줄이기 위해 background probability를 alpha(0.5)를 곱해 감소시킴.   
## 3.2 Image-specific Prototype Exploration
Image specific prototye은 feature distribution of each class를 표현하기 위해 도입. allowing to capture more complete regions.  
2 steps  
1. provides robust class-wise seed regions.
2. aggregates these seeds on a comprehensive faeature space to achieve accurate imagespecific representation.

### 3.2.1 Structure-aware Seed Locating 
CAM에서 seed를 뽑을 때 fixed threshold를 사용할 수 있으나 각기 다른 object, scenario에서 동일한 threshold를 쓰기는 어렵다.  
CAM은 discriminative region에 more activation하지만 remaining regions에도 weak attention을 준다. 즉 spatial structure of semantic objects를 제공할 potential이 있다.  

arbitrary pixel i  
semantic feature vector f^i를 query to compute semantic correlation with all pixel in that feature map 으로 사용.  
pixels with high correlation scores are more likely to belong to the same class.  
inter pixel semantic correlation S^i(j). pixel i와 j의 correlation.  
S^i: Structure map of pixel i.  

pixel i의 Sturcture map과 CAM사이의 class wise IoU 계산으로 sturcture similarity C_k^i 계산.  
i와 모든 pixel사이 유사도 vs kth CAM사이 Intersection of Union -> 유사도 계산.  
C_k^i는 structure similarity for pixel i with respect to kth class를 뜻함. 즉 pixel i와 class k의 유사도.  

pixel i는 maximal similarity를 갖는 category로 assign됨.  
R_k^i = 1 if k = argmaxC_k'^i , 0 otherwise  
모든 pxiel에 대해 수행하면 seed Region R을 구할 수 있다.  

### 3.2.2 Backgroud-aware Prototype Modeling  
> features from shallow layers contain rich low-level visual
information (e.g. color, texture), which is more suitable
to model background-related information.  

Backbone을 수정하여 Hierarchcal feature F_h 얻음  
image specific prototype P_k of foreground and background는 seed regions in hierarchical feature space의 centroid를 구하듯이 계산  
P_k = F_h의 pixel 중 R_k(class k의 seed)가 1인 pixel의 평균값  
즉 F_h의 class k에 대한 정보  

Image Specific CAM (IS-CAM)은 P_k와 F_h로 계산  
M_k^tilda = F_h의 각 pixel 값과 P_k값의 corelation(cosine similarity) 계산 후 ReLU로 음수 날림  

## 3.3 Self-supervised Learning with GSC  

classification loss는 multi-label soft margin loss between the image-level category label y and prediction y^hat (GAP of CAM)  
-> soft margin loss라는데 그냥 sigmoid CE인듯  

GSC: General-Specific Consistency Loss  
original CAM과 IS-CAM의 차이를 최소화. -> 양쪽 모두 correction 수행됨  
L1 normalization으로 계산.  
K+1개 class로 averaged. (K class + background)  
IS-CAM은 CAM이 absent object region에 더 pay attention하게 만듬.  

# 4. Experiments 
## 4.1 Experimental setting
backbone: ImageNet pretrained ResNet50 with output stride of 16  
FC layer가 classifier with output channels of 20으로 변경  

data augmentation:  
random flipping, random scaling and crop  
'Jungbeom Lee, Eunji Kim, and Sungroh Yoon. Anti adversarially manipulated attributions for weakly and semi supervised semantic segmentation'과 동일  

## 4.2 Comparison with State of the arts
pseudo label을 만들고 mIoU를 계산하지 않고 activation map (localizaiton map)그 자체의 mIoU를 계산하였다.  


