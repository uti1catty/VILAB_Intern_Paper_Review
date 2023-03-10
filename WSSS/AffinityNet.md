# Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation [CVPR18]

github: https://github.com/jiwoon-ahn/psa

## Introduction
CAM의 low activation score를 모두 제거 -> only confident oject and background area만 사용  
training example: pair of adjacent image coordinates on confident area 같은 class라면 1, 아니라면 0  

## Our Framework
WSSS의 접근을 2단계로 나눈다.  
1. Synthesizing pixel-level segmentation labels of training images given their iamge-level class labels.  
2. Learning a DNN for semantic segmentation with generated segmentation labels.   

3개의 DNN으로 구성.  CAMS계산 net, AffinityNet, segmentation model.  
앞의 2개의 모델은 training image의 segmetnation label을 생성하고 마지막 DNN이 실질적인 semantic segmentation을 수행  

### 3.1 Computing CAMs
CAM이 segmentation seed로 사용  
AffinityNet을 학습하기 위한 source로 사용  
Learning deep features for discriminative localization CVPR 2016[40]을 따라 계산  

### 3.2 Learning Affinity Net
>AffinityNet aims to predict class-agnostic semantic affinity btw a pair of adjacent coordinates on a training image  

#### 3.2.1 Genrating Semantic Affinity Labels
학습을 위한 sample 추출  
alpha값을 낮춰 Mbg를 증폭시킴 -> 다른 class의 명확치 않은 부분을 Mbg가 잡아먹도록 함  
각 class에 대해서 다른 모든 class와 bg보다 더 값이 높은 부분만 추출  
반대로 alpha값을 키워 Mbg를 약화시켜 confident bg 추출  
어디에도 포함되지 않은 영역은 'neutral'로 부른다.  
neutral에 포함되지 않은 (xi, yi), (xj, yj) 쌍에 대하여 Affinity label Wij*는 두 개가 같은 class 라면 1, 다른 class 라면 0을 갖는다.  
최소 하나가 neutral이라면 train과정에서 해당 쌍을 무시한다.  
Large number of pairwise affinity labels 획득  

#### 3.2.2 AffinityNet Training
training을 수행하는 동안 충분히 인접한 pair에 대해서만 고려  
1. 너무 멀리 떨어진 경우 lack of context로 affinity 예측 어려움
2. 인접한 pair만 봄으로서 computational cost를 낮춘다. 

euclidean dinstance가 r이하인 pair만 고려 (set P)  
P는 positive로 bias되어 있다. negative가 boundary에서만 발생하기 때문  
또한 bg가 object보다 커서 bg positive가 더 많음  
따라서 P를 Pbg+, pfg+, p- 로 분류하여 각각 loss를 계산후 종합  
<Loss 계산 식>  
Loss가 class agnostic하다.  

### 3.3 Revising CAMs Using AffinityNet
AffinityNet의 local semantic affinites -> transition probabiity matrix for random walk  
AffinityNet의 결과물은 affinity feature map. Semantic affinity는 수식3을 통해 계산되어 affinity matrix W를 형성한다. 이때 semantic affinity는 radius r 내의 feature들 끼리만 계산된다.  
transition matrix T는 식 11에 따라 계산된다.  
1 이상의 hyperparameter beta로 W를 hadamard power하여 불필요한 값을 제거한다.  

## 4 Network architecture
3가지 DNN의 backbone으로 ResNet38사용  
GAP, fully connected layer 제거  
마지막 3개 level을 atrous convolution으로 변경  
atrous convolution은 feature map resolution을 희생하는 대신 segmentation quality를 증가시킴  

affinityNet: 마지막 3개 level 128, 256, 512 channel 가져옴 -> 1*1 conv 각각 시행 -> concat 하여 896 channel feature map 생성 -> 1*1 conv 시행  

## 5 Experiment
backbone은 ImageNet에서 pretrained  
horizontal flip, random cropping, color jittering 사용. PASCAL VOC2012를 Adam으로 fine tuning.  
AffinityNet 말고 나머지는 random scaling 사용  
