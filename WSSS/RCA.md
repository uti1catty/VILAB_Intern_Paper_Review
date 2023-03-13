# [CVPR22] Regional Semantic Contrast and Aggregation for Weakly Supervied Semantic Segmentaiton 

github: https://github.com/maeve07/RCA.git  

# 1. Introduction
기존의 image-level label만 사용한 WSSS 방법들은 use only signle image information for object localization, neglecting inter-image contextual information.  

Image label은 individual image의 cateory를 알려줌과 함께 all dataset의 semantic structure를 unveil한다.  
'cat' dataset내에는 visually different하지만 동일한 semantically similar한 쌍이 있다.  
'cat, dog'의 different concept에서는 visually similar하지만 semantically different 하다는 정보가 있다. 

2가지 관점에서 학습 
1. Semantic contrast  
network가 embedding을 동일한 category의 memory embedding과 가깝게, 다른 category의 embedding과는 멀게 함으로서 holistoic object pattern understanding을 올린다.  
2. Semantic aggregation  
dataset level의 contextual knowledge를 수집하여 의미있는 object representation을 생성하도록 한다. non-parametric attention module(summarizes memory representation for each image independently)을 통함.  
inter-image context mining으로 capture more informative dataset-level semantics 하도록 함.  

# 3 Our approach
## 3.1 Problem statement
image I (w x h x 3)  
image level label y=[y1, y2, ...yL] class L개  

기존: image I를 F_FCN(Fully Convolutional Network)를 통과해 Feature map F 생성 (W x H x D)  
F를 class-aware convolutional layer F_CAM에 통과하여 class-aware attention map P 생성 (W x H x L)  
GAP로 prediction p 생성  
기존의 방법은 only exploit limited contextual cues in individual images.  

## 3.2 Regianl semantic Contrast and Aggregation
### 3.2.1 Pseudo-Region Representation  
dense embedding F를 P를 base로 categorical region representation으로 변경  
I에 존재하는 lth category(yl = 1)에 대하여 그 카테고리의 region level semantic information이 compact embedding vector f_l (D 1차원) 로 변경. Masked average pooling (MAP) 사용  
P_l(l category에 대한 attention)이 threshold(P_l의 mean)를 넘긴 위치를 M_l mask로 정의. 즉 M_l은 l category에 대한 attention이 수행된 영역.  
M_l과 F의 각 dim을 곱하여 average pooling 수행.  
즉 f_l[i] 는 embedding F[i] 중 category l에 대한 attention이 발생하는 위치에 있는 값들의 평균값  

### 3.2.2 Pseudo Region Memory Bank
RCA를 위한 dataset-level regional semantic information 저장을 위한 bank  
M은 L개의 dictionary를 가짐. each for one category. M = {M1, M2, ... ML}  
M_l의 각 entry는 holistic region-aware representation m_l (D 1차원) of the lth category in image I observed in the whole learning phase.  
Back propagation 단계에서 image I의 feature vector f_i가 memory representation m_l에 update  
m_l <- r * m_l + (1 - r) * f_l. (r: momentum for memory evolution)  
update는 image I에 l이 나타났으며 class l에 대한 classification score가 threshold를 넘을 때 (p_l > nu) 만 수행  

즉 M_l은 dataset level의 category별 일반적 sementatic 정보 (feature vector level)를 가진다고 볼 수 있다.  

### 3.2.3 Regional Semantic Contrast(RSC)
image I의 categorical pseudo region embedding f_l을 같은 class의 memory features m_l과 유사도를 증가시키고 (positive) 다른 class의 memroy들 m_l' (l' in M/M_l)과 유사도를 감소시키도록 (negative) loss 정의  
(eq 4)에서 마치 f_l과 positive, f_l과 negative의 similarities 사이 softmax를 수행한 후 f_l과 positive값을 loss를 위해 사용하는 모습을 보임. softmax에서 값이 커지기 위해서는 해당 쌍이 다른 쌍에 비해 크기가 커야 함. 따라서 positive 값은 커지고 negative 값은 작아지도록 loss 역할 수행.  

이러면 supervised contrastive learning같지만 label이 weak하고 noisy하기 때문에 learn robust representation이 어렵다.  
'region mixup'으로 문제 해결  
image I의 한 영역 region l마다, 다른 mini batch image의 region l- (다른 category)와 linear하게 병합.  
mixed region f_l^hat = wf_l + (1-w)f_l- (w = B(beta beta) beta distribution)  
mixup contrastive loss L_l^RM-NCE 는 2개의 similarity의 linear 병합으로 이루어짐.  

### 3.2.4 Regional Semantic aggregation 
memory bank의 large scale represesntation이 over complete을 포함하고 잇고 일부 noisy함.  
곧바로 aggregating하는 것은 computationally expensive  
따라서 각 class l마다 K-means clustering 수행으로 K개 class centroids 생성  
Q_l (K X D)  
각 centroids 모임 Q = {Q1, Q2, ... QL} (K X D X L)  
각 mini-batch image I with feature F (W X H X D)에서 affinity matrix S 계산  
S = softmax (F X Q.T) (WH X LK) (F: (WH X D), Q: (LK X D))    
Sosftmax가 각 row를 normalize  
S의 각 entry는 F의 row와 Q.T의 column의 유사도 (affinity)를 나타냄.  
즉 Feature map의 Feature들과 모든 class의 centroids와의 유사도를 계산한 결과.  

contexttual summaries F' = S X Q (WH X D)  
S의 각 row는 F의 각 row와 Q의 모든 row 사이의 affinity  
F'의 각 row는 F의 각 row와 Q의 모든 row 사이의 affinity를 weight으로 Q의 모든 row를 weighted sum한 결과.  
즉 F의 각 row와 유사한 것들을 Memory level의 정보들로부터 가져옴. -> inter-image global context  

F'을 (W X H X D)로 reshape하여 F와 concat  
F^hat = [F, F'] (W X H X 2D)  
F^hat은 F의 intra-image local context와 F'의 inter-image global context를 갖게됨.  

### 3.2.5 Class Activation Map Prediction
F^hat을 class-aware convolution layer F_CAM에 넣어서 final activation maps O 생성. (앞의 F_CAM과는 다른 structure)  
O (W X H X L)  

## 3.3 Detailed Network Architecture 
FCN: VGG16, ResNet38  
F_CAM: 1x1 conv  

최종 loss는 3개의 loss 합  
1. LRM-NCE: region mixup contrastve loss whihc is computed as the average loss of all regions appearing in I.
2. LCE(GAP(P), y): intermediate CAM prediction P
3. LCE(GAP(O), y): main cross entropy loss. 

# 4. Experiment
## 4.1 Experimental setting
classifier를 train한 후 O를 각 train image마다 생성해 foreground seed로 사용.  
compute saliency map for each image using off-the-shelf models to estimate background cue.  
Dense CRF is used as a post-processing routine to refine segmentation boundaries.  


