# [CVPR22] Cross Language Image Matching for Weakly Supervised Semantic Segmentation 
github: https://github.com/CVI-SZU/CLIMS

# 1 Introduction
Fixed set of object categories만 first stage training (classifier)에 사용되기 때문에 class-related background pixel(철도) 가 closely related object (기차)를 예측하는데 기여한다. 따라서 initial CAMs에 불필요한 background activation이 발생한다.  

이 문제 해결을 위해 CLIP 기반의 CLIMS를 구현함  
CLIP은 400million image-text pair를 바탕으로 학습되어 고정된 object category가 아닌 open-world setting에서 text label과 image사이 관계를 잘 학습하였다.  

CAM 생성 단계의 GAP와 FC를 제거하고 대신 conv layer를 사용  
 directly generate an activation map for each class under the supervision from CLIP model, where natrual language can be used to guide the model for the activation maps generation.  

3가지 CLIP based loss와 Regularization term 정의 (P: initial CAMs)
1. Object region and Text label Matching loss (L_OTM)  
P를 mask로 masked image 형성, masked image와 text label을 CLIP에 넣어 cosine similarity가 최대가 되도록 하는 loss.  
Foreground object area와 given text label의 유사도를 최대화  
2. Background region and Text label Matching Loss (L_BTM)  
L_OTM만 있을 경우 새의 머리만 activate 되어도 새와 유사도 높음. 이 문제 해결을 위해 background loss 적용  
(1-P)를 mask로 하여 background region 추림. masked image와 L_OTM에서와 같은 text label을 CLIP에 넣고 유사도를 최소화  
activate 되지 않았던 부위가 background로 포함되어 유사도가 높아짐. 이를 최소화 하여 해당 부위도 activate 하도록 유도  
3. Co-occuring Background Suppression loss (L_CBS)
위의 2개의 loss만 사용했을 때 background closely related to the object 영역이 activation 되는 경우 발생  
이 해결을 위하여 L_CBS 도입.  
set of class-related background text label 정의 (raiload, river 등등)  
P를 mask로 하여 Foreground object area 추출. masked image와 bg label set을 CLIP에 넣어 각각 유사도를 최소화.  
잘못된 영역이 FG로 activate 될 경우 bg label과 유사도가 높아짐. 이를 최소화 하여 해당 영역을 activate하지 않도록 유도  
4. Regularization term (L_REG)  
L_OTM, L_BTM, L_CBS만 있으면 만약 object와 object와 큰 관련이 없는 background만 있을 때 관련 없는 bg의 activation을 막을 수 없다. 해당 bg가 activatation map에 있더라도 CLIP이 object를 올바르게 판단할 것이기 때문.  
이 문제 해결을 위해서 관련 없는(irrelevant) bg를 exclude하도록 activation map P_k의 크기를 제한하는 pixel level area regularization term 도입  

# 3 Methodology
## 3.2 Cross Language Image Matching Framework
GAP가 제거되고 sigmoid를 W이후 바로 시행하여 CAM을 direct로 얻음  
Pk(h,w) = sigmoid(Wk.T Z(h, w)) (Z: (C x H x W), Pk: (K x H x W))  
위의 3가지 loss 계산을 위한 vectors 계산  
text는 'a photo of {}' 형태로 넣는다.  
class-related background text t_kl^b는 각 class k마다 L개의 text label이 정의되어 있다.  
예를 들어 k-th object가 'boat'라면 t_k^b는 {a photo of river, a photo of lake}가 되며 t_k0^b = {'a photo of river'}, t_k1^b = {'a photo of lake'}가 된다.  

# 4 Experiments
## 4.1 Experimental setup
rnadom scaling then random croppiing to 512x512  
Horizontal flipping  

SGD, cosine annealing policy to lr  
