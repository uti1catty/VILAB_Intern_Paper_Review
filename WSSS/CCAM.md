# [CVPR22] C2AM: Constrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation

Contrastive learning for Class-agnostic Activation Map (C2AM) generation only using 'unlabeled image data', without the invovement of image-level supervision. 

observation:  
1. semantic information of foreground objects usually differs from their backgrounds
2. foreground objects with similar appearance or background
with similar color/texture have similar representations in
the feature space.

# 1. Introduction
Image-level supervision 문제점: classifier가 discriminative regions of target object를 찾으려 함 -> CAM이 sparse and discriminative object region에 focus하도록 limit함. 따라서 CAM을 WSOL, WSSS에 바로 적용하여 estimate complete object region을 할 수 없음  

CAM이 1 target activation을 포함하는 K class activation map을 predict하는데 반해 CCAM은 predicts only one class-agnostic activation map to indicate the forground and background regions. 

동일 image의 foreground와 background는 distance차이가 큼  
한 image의 foreground와 다른 image의 background도 dist 차이가 큼  
한 image의 background와 다른 image의 background의 color/texture가 유사하다면 distance가 small하다  
=> cross-image foreground-background contrastive loss  

foreground-background가 negative pair  
for-for / back-back이 postivie pair  
positive에서 서로 다른 image이므로 less similar semantics를 share할 수 있다. 이 문제해결을 위해 feature similarity based rnak weighting 을 구현하였다. to automatically reduce the influence of those dissimilar positive pairs.  

class-agnostic actiation map 으로부터
1. class agnostic object bounding boxes 추출 -> localization
2. background cues(신호) 추출 -> inital CAM을 refine하여 false actiation of background can be effeciently reduced  

# 2. Related Work
Unsupervised contrastive learning  
1. instance-wise contrastive learning  
postiive: samples augmented from the same instance. negative: samples augmented from teh different instance  
2. clustering based contrastive learning  
clustering algorithm is applied to generate the pseudo label for training samples, and then the supervised contrastive learning is applied.

# 3. Methodology
## 3.1 Architecture
n image Xi  
encoder h(): ResNet or VGG  
Supervised or unsupervised pretraining, (moco and detco) on ImageNet-1K can be adopted as initialization of h()  
Encoder를 통해 feature map Zi (C X W X H) 생성  (C는 channel number)  
Zi가 Disentagler로 들어감  
Disentagler의 Activation head Phi()를 통해 class-agnositc actiation map Pi (1 x H X W) 형성  
Phi()는 3X3 conv & Batch normalization  
Pi: foreground, 1-Pi: background  
Zi와 Pi/1-Pi 로 feature representation v_i^f, v_i^b (1 X C) 형성  
=> 먼저 Zi와 Pi를 flatten. Zi(C X HW), Pi (1 X HW)  
=> 이후 matmul v_i^f = Pi X Zi.T / v_i^b = (1 - Pi) X Zi.T  

-> Pi와 1-Pi는 attention에 대한 정보. 우리가 원하는 feature representation은 foreground와 background feature에 대한 정보를 담고 있는 것이다. (color, texture 등 ) 따라서 Feature Map Zi와 attention 정보 Pi를 곱함으로서 Foreground로 attention이 지정한 영역의 feature 정보를 추림  

## 3.2 Foreground-background Contrast 
(v_i^f, v_j^b)이 negative pair  
negative contrastive loss는 두 pair가 멀 수록 loss가 작아짐  
cosine similarity기반으로 둘이 가까우면 1, 90도로 멀면 0이 되어 -log(1-sim)가 가까울수록 loss를 키움.  
모든 (i, j) pair에 대해 loss를 더한것이 L_neg (같은 image내의 f,g 포함)    
## 3.3 Foreground-Foreground, Background-Background Contrst with Rank Weighting  
서로 다른 image의 F-F, B-B pair가 positive pair  
그런데 같은 F더라도 color, texture의 유사도가 작은 경우 distance가 커서 문제가 될 수 있다. 따라서 rank weighting을 수행  
우선 feature 쌍 (v_i^f, v_j^f) cosine similarity를 구하여 rank를 기준으로 weight을 계산  
w_ij^f = exp(-alpha x rank(s_ij^f))  
rank는 모든 쌍 (i, j)에 대한 similarity set 내에서 순위를 말함
-> 아마 sim value가 클 수록 (유사할수록) rank가 작은 것으로 보임  
따라서 유사할수록 weight이 커짐. (0~1)  
weighted positive loss를 F, B에 대해 각각 계산하고 더하여 L_POS 계산  
서로 다른 i,j에 대하여(i, i 배제) L_ij^f = -w_ij^f X log(s_ij^f) 로 sim이 클수록 weight이 크고 loss가 작아짐  
=> 그러면 similarity가 작더라도 weight이 작아서 loss를 0근처로 끌어 내릴텐데 distance를 가깝게 두도록 유도가 가능한가?  
==> sim이 큰 쌍만 고려하겠다는 의미인듯. sim이 작은 녀석들은 버리고 sim이 큰 녀석들의 dist만 가까이 두겠다는 것.   
===> 그런데 지금 서로 비슷한 feature를 가진 것들의 dist를 줄여야 하는데 서로 비슷한 feature를 가졌지만 v값의 차이가 클 경우 이를 가까이 둘수있게 v값을 뽑을 수 있도록 loss를 정하고 싶은 것인데 weight이 sim값을 기준으로 결정되어 비슷한 feature를 가졌지만 v값의 차이가 클경우 loss에 포함이 작게 된다. 의미가 있는가?  

## 3.5 WSSS 
backgroudnd activation map 1-P를 background cue로 사용  
CAM의 앞에 background cue를 삽입하고 channel방향 Argmax를 통해 CAM을 refine  

# 4 Experiment
Dataset: CUB-200-2011 / PASCAL VOC2012  
Evaluation: WSOL-Top1 Loc, Top5 Loc, GT-known Loc / WSSS: mIoU  

