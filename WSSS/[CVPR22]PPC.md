# [CVPR22] Weakly Supervised Semantic Sgmentation by Pixel-to-Prototype Contrast  
github : https://github.com/usr922/wseg

# 1 Introduction
2 implicit valuable priors  
1. features should retain(유지) semantic consistency across different views of an image  
2. pixels sharing the same label should have similar representations in the feature space,
and vice versa.  

Core idea: pulling pixels together to their positive prototypes and pushing them away from their negative prototypes to learn discriminaitvce dense visual representaion.  
Prototype: represesntative embedding of a category. - Estimated from piel-wise feature embedings with the top activations in the CAMs.  

cross view, intra view regularizations are considered.  


# 3 Methodology
Our method can be interpreted as a regularization term
that is adaptable to any existing WSSS framework without
changing the inference procedure.  

Loss: Cross view contrastive loss + intra view contrastive loss  

## 3.1 Preliminary
CAM 생성은 마지막 layer로 1x1 conv를 적용해 f' (C X HW) (C: class number) 의 CAM을 바로 생성  

## 3.2 Pixel-to-Prototype Contrast  
먼저 CAM에서 pseudo mask y=argmax(m)을 생성  
각 category마다 representative embedding, the prototypes P = {p_c}_{c=1~C}가 존재  
goal: learn discriminative feature emebdding for each pixel aided by contrastive learning in a projected feature space.  

먼저 pixel-wise projected feature vi (128 dim)을 projector로 생성. (projector는 1x1 conv, ReLU로 구성)  
vi와 P 사이에 contrastive loss  
F(vi;yi;P) = -log(exp(vi x p_yi / tau) / \Sum(exp(vi x pc / tau)))  
즉 이 값은 pseudo label yi의 prototype과 pixel feature vi를 곱한 값들의 softmax 값으로 이 값이 커질수록 F값은 작아짐. 곱한다는 것은 similarity를 뜻하므로 yi의 prototype과 feature vi가 유사할 수록 F값이 작아짐.  

## 3.3 Prototype Estimation
pixel wise CAM value를 confidence 값으로 사용  
top K confidence값으로 prototpe estimation  
prototype p_c는 projected pixel-wise embedding의 confidence value에 따른 weighted average로 계산  
K개의 CAM value를 골라서 CAM value와 vi를 곱한 값을 더하고 k개 CAM value더한값으로 나눔.  
이후 L2 normalization시행  
K가 작다는 것은 higher confidence를 의미.  
global context of the entire dataset을 capture하기 위해 prototype을 across the training batch에서 계산.  전체 training batch에서 highest CAM value에 해당하는 pixel을 선택.  

## 3.4 Cross view Contrast
source view S를 spatial transformation A()로 target view T를 생성.  
S와 T를 pretrained CNN backbone에 넣어 2개의 CAM을 생성  
source의 CAM과 feature map에 같은 transformation A()를 적용한다. (Target의 CAM, feature map과 유사한 조건을 맞추기 위함)  

### 3.4.1 Cross Prototype Contrast  
pixel i와 pseudo label y_i, projected feature embedding vi, 다른 view의 prototype P'에 대해서 prototype contastive loss 계산  
L^cp = F(vi;yi;P')의 모든 pixel에 대한 평균  

### 3.4.2 Cross CAM Contrast  
CAM은 pseudo label을 생성하는 역할. CAM이 동일하다면 같은 label이어야 할것.  
pixel i, prototype P, 다른 view의 pseudo label yi'에 대해서 CAM contrastive loss 계산  
L^cc = F(vi;yi';P)의 모든 pixel에 대한 평균  

현재 view와 다른 view는 source-target, target-source 모두 적용  

L^cross = L^cp + L^cc  

## 3.5 Intra-view Contrast  
### 3.5.1 Intra-view Contrast
하나의 image 안에서 contrastive loss 계산  
L^intra = F(vi;yi;P)의 모든 pixel에 대한 평균  
그런데 실험적으로 intra loss가 performance를 줄이는 결과를 가져옴.  
No precise pixel-wise annotation이 없어서 pseudo label yi가 inaccurate(정확하지 않음)으로 인해 문제 발생.  
이를 Semi-hard Prototype mining과 Hard Pixel Sampling으로 해결.  
### 3.5.2 Semi-hard Prototype mining 


