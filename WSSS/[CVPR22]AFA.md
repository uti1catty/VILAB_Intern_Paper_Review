# [CVPR22] Learning Affinity from Attention: End to End Weakly Supervised Semantic Segmentation with Transformers
AFA: Affinity from Attention  
MHSA: Multi-head Self-attention  
PAR: Pixel Adaptive Refinement module  
github: https://github.com/rulixiang/afa

# 1 Introduction
Transformer architecutre naturally benefits the WSSS task.  
Transformer의 self-attention mechanism이 global feature relation을 잘 안다.  
그러나 MHSA가 capture한 affinity는 여전히 inaccurate하여 label revise를 위해 MHSA를 직접적으로 afinity로 사용할 수 없다.  

핵심 기여 목록 
1. End to End Transformer based framework for WSSS with image-level labels. 
2. Affinity from Attention(AFA) modeule. AFA learns reliable semantic affinity from MHSA and propagates the ppseudo labels with learned affinity  
3. Efficient Pixel-Adaptive Refinement(PAR) module. Incorporates RGB and position information of local pixels for label refinement. 

# 3 Methodology  
## 3.1 Transformer Backbone 
image 를 hxw patch로 분할  
각 patch를 flatten하고 linearly project하여 hxw token 생성  
patch token이 MLP로 project되어 QKV 형성  
self-attention matrix 계산 및 output Xi 계산  
final output Xo = FFN(X1||X2||...||Xn)  
FFN은 Layer Normalization과 MLP layers 포함  
||는 concat  
## 3.2 CAM Genration  
Transformer의 결과물 feature map F: hw x d  
activation map M^c를 형성하기 위해 Weight Matrix W를 이용하여 F를 class c에 대한 contribution에 따라 weighting  
F의 각 patch마다 계산. 각 dimension의 class c에 대한 contribution 계산  
F의 dimension 방향으로 weighted sum하여 hw x 1 짜리 값이 나옴 -> class c1에 대한 M^c1  
따라서 M^C는 hw x C 의 shape을 갖게 됨  

## 3.3 Affinity from Attention  
MHSA를 바로 affinity로 사용할 수 없다. 따라서 AFA module 형성  
Transformer block의 MHSA를 S라고 할 때 (hw x hw x n: n은 head 수)  
affinity matrix A = MLP(S + S.T)  
MHSA는 direct graph이지만 affinity는 symmetric이어야 한다. 같은 semantic을 가지면 같아야 하므로.  
따라서 transpose를 하고 더해준 후 MLP로 affinity matrix를 도출한다.  
### 3.3.1 Pseudo Affinity Label Generation 
먼저 pseudo label Y_p를 형성한다.  
2개의 background score beta_l, beta_h 정의. CAM M (h x w x C)  
Mij의 최대값이 beta_h 이상이면 해당 class를 Y_pij로 함. argmax(Mij)    
Mij의 최대값이 beta_l 이하이면 0 (background)  
사이이면 255 (ignore)  

pseudo affinity label Yaff (hw x hw) 계산법  
pixel (i, j)와 (k, l)이 같은 semantic이면 affinity를 positive로 두고 아니면 negative로 둔다  
둘 중 하나라도 ignore이면 affinity도 ignore로 설정한다.  
두 픽셀이 정해진 윈도우 내부의 경우에만 계산한다. 너무 먼 거리일 경우 ignore로 설정한다.  

### 3.3.2 Affinity Loss  
affinity matrix A와 Yaff로 loss
Yaff에서 positive인 pixel(서로 같은 semantic)들은 A에서 음수가 되어야 하고  
Yaff에서 negative인 pixel(서로 다른 semantic)들은 A에서 양수가 되어야 한다.  
sigmoid를 통해 0~1로 변환한 후 negative는 1 - sig(A), positive는 sig(A)를 Loss에 더함  
-> 반대여야 하지 않나?  

### 3.3.3 Propagation with Affinity 
Random Walk 도입  
Transition matrix T를 A로부터 계산한 후 CAM M을 propagation  

## 3.4 Pixel-Adative Refinement (PAR)
일반적으로 CRF를 많이 사용하지만 CRF is not a favorable choice in end to end framework since it remarkably slows down the training efficiency  
pixel (i, j) , (k, l) 사이의 RGB 차이, Position 차이 값을 계산 (Position은 XY coordinate)  
k_rgb^ijkl, k_pos^ijkl  
두 pixel의 차이가 작을수록 k값은 커짐  
두가지 종류의 information을 하나로 병합하기 위해 각각 softmax를 수행하고 weighted sum  
이때 (i,j)에 대해서 (k, l)을 계산하는데 (i, j)의 neighbor set을 바탕으로 softmax를 계산한다.  
k^ijkl 는 (i, j) 주변의 모든 pixel들 중에서 (k, l)이 얼마나 (i, j)와 유사한지를 나타냄  

initial CAM과 propagated CAM을 모두 PAR로 iterative 하게 refine  M (h x w x C)  
(i, j)의 모든 근처 pixel (k, l)에 대해서 M^klc를 k_ijkl로 weighted sum하여 M^ijc 계산 (refine)  

즉 각 (i, j)의 CAM은 (i, j) 주변의 pixel들의 CAM을 (i, j)와 (k, l)의 유사도를 기준으로 weighted sum을 수행하는 것  

