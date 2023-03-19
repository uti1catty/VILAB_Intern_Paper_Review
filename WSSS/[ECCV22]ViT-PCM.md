# [ECCV22] Max Pooling with Vision Transformers reconciles class and shape in weakly supervised semantic segmentation
github: https://github.com/deepplants/ViT-PCM

PCM: Patch Class Mapping  
BPM: baseline pseudo mask  

# 1. Introduction
기존의 CAM을 기반으로 하는 방법들은 다양한 refinement를 수행하여 multi stage로 구성됨.  
refine 방법들이 denseCRF, saliency등을 사용하는데 이는 PascalVOC에 optimize된 것. 즉 PascalVOC에 bias되어 있다.  

CAM에 의존하지 않는 새로운 pseudo mask computation method 제안.  
ViT의 locality property를 활용하여 multi label classification과 sementic segmentation 사이의 ㅎ과적인 mapping을 수행한다.  
Patch의 categorical distribution over the class of interest가 given일 때 GMP(Global Max Pooling)을 사용하여 relevance(연관성) of each patch를 구한다.  
project patch feature to class prediction (PCM) using multi-label BCE loss (MCE)  
2개의 branch를 정의하여 translation과 scaling transformation에서 equivariance를 보장한다.  

# 3. Motivations of using ViT and bypass CAM
WSSS는 multilabel image classification과 pixel-level classifcation을 mapping하는 것.  
둘은 서로 완전히 다른 space  
CNN은 inductive bias on the image features local structure가 있다. Convolution kernel이 있기 때문.  
이 inductive bias가 CAM이 특정 class prediction에 가장 크게 기여한 pixel을 indicate하게 해준다.  
생성된 Map은 'image feature와 pixel 사이의 mapping을 induce하지 않지만' 여전히 매력적이다.  

ViT는 image가 fattened patches로 split되고 encode되어 bias가 적다.  
따라서 attention과 position embedding을 통해 spatial relation을 처음부터 학습한다.  
이 학습은 내부 구조를 지정하는 각 patch에 대한 여러 basis fucntion을 생성한다.  
이 basis function은 patch가 속한 class를 implicit하게 설명하낟.  
mapping문제는 patch principal component에 의해 발생하는 implicit class representation을 푸는 것이다.  
'Explicit search method'가 이 mapping을 modeling한다.  

X: Image (h x w x 3) / C: class (K개) / patch가 pixel이라 가정  
ViT는 Image X를 f(phi)를 통해 vector value (0,1) for each category in C를 mapping.  
 basis functions specifying the patches’ internal structure, implicitly accounting for the patch classes, by a tensor Z  
Z: h x w x C  
Z를 class C방향으로 더하면 1이 되는 확률 tensor이다.  
f(theta)를 segmentation model이라고할 때 f(theta)는 original image의 patch가 some precise class in C에 belong하는지를 평가한다.  

GMP는 f(theta)와 f(phi)를 relate한다.  
Z^k의 최대값으로 선정된 Zij^k는 image에 category k가 나타났는지를 말해주는 값이다.  
GMP는 이렇게 image class prediction과 patch class prediction을 연결한다.  

# 4 The explicit search method
f의 output은 Y^hat (C^ h x w)로 pseudo mask.  
ViT는 f의 일부분.  
F는 basis functions specifying the patches internal structre를 표현.  
F (s x e) s = (n/d)^2 d: patch size  
weight matrix W (e x K)  

Z = softmax(FW) Z: (s x K)  
yk = GMP(Z^k)    
Z^k = softmax(A^k), Aj^k = Fj x W^k  
F are the encoded represetnation of pathces U  
Fj is the feature map of patch Uj  
Aj^k is the logit of patch Uj with respect to class k  

L_MCE는 gt t와 y의 BCE  

h: class  
eq 7. linear search mechanism of the proposed optimization iteratively selecting the most representative fature F_ih of each category h.  
W^h는 error value (th-yh)에 따라 F_ih 방향으로 움직이고 h가 아닌 다른 class k에 대하여 term Z_ik^h (tk-yk)/(1-yk)에 따라 F_ik 방향으로 움직인다. 
만약 (tk-yk)/(1-yk) = 1 이라면, (tk=1 즉 class k가 image에 존재) W^h는 F_ik와 반대방향으로 움직인다.  

# 5 ViT-PCM model structure  
## Augmentation  
first branch에는 input image 가 usaul로 augment   
second branch에는 translated, rotated, scaled됨.  4개의 image를 single image로 merge.  
## ViT patch encoder  
ViT가 batch of image를 받아 features F_in 반환  
## HV-BiLSTM patch conditioning  
2개의 bidirectional LSTM(BiLSTM)이 row/col 방향으로 Fin을 transformed to tensor grid  
2개가 concat되어 HV-BiLSTM을 형성하여 feature map F 형성  
>HV-BiLSTM improves information amid neighbour patches by conditioning each
patch on all other ones in horizontal (H) and vertical directions (V)  

## Patch Classifer (PC)
section 4설명처럼 BPM 생성  

## Two branches for Equivariant regularization  
ViT는 translation에 equivariant하지 않다. absolute positiaonal encoding used for self-attention 때문.  
GMP는 positional encoding에서 독립적이고 invariant to transformation이지만, BMP생성은 아님.  
2개의 branch를 사용하여 network가 equivariance properties를 학습하도록 함.  

L_ET = transformation inverse한 patch의 값 사이의 Cross Entropy Loss  

# 6 Experiments and Results
## 6.1 Setup
### 6.1.3 Reproducibility
Image는 384 384로 resize  
augment: rnadom colour jitter, random grayscale, 90도 rotation, verticla, horizontal flip  
Initially freeze backbone and ignore output feature for the [cls] token.  

