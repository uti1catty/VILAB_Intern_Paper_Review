# [CVPR21] Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation

github: https://github.com/allenwu97/EDAM

classification network에 activation map generation을 직접 포함하는 Embedded Discriminative Attention Mechanism (EDAM)을 소개한다.   

Discriminative Activation (DA) layer: segmentation을 위한 class-specific pixel level pseudo-labels에 사용되는 "series of normalized class-specific mask"를 생성.  

그 후 feature map과 mask가 곱해진다. 각각은 encodes the specific information of the corresponding category in the input images.  

Collaborative Multi-Attention (CMA) module이 class-specific activation maps를 받아서 각 category별 collaborative information을 추출한다.   

Inference단계에서 DA layer 결과인 activation mask를 바로 pseudo-labels for segmetantion으로 사용.  

# 1. Introduction
기존의 task들은 explore information in an $\mathit{implicit}$ manner. 
EDAM을 통해 WSSS를 위한 class-specific mask를 $\mathit{explicit}$ 하게 infer한다. 이를 위해 intra-image, inter-image homogeneity(동종성)를 explore한다.   

EDAM: DA-layer + CMA module  

1. DA layer는 fg categories와 bg를 위한 class-specific mask를 predict한다.  
2. 그 후 mask와 original feature map을 곱하여 discriminative activation map을 생성한다.  
3. CMA module은 multi-image의 activation map에 self-attention을 적용하여 fg object의 collaborative information을 explore한다.  
4. 각 iamge의 attended activation map에 average pooling을 적용하고 그 결과를 binary classifier에 넣는다. 

CMA module은 training에서만 쓰인다.  

기존의 post-processing을 위해 saliency map이 많이 사용되었으나 class-agnostic이므로 wrongly highlites the non-target object, supress the insignificant target object 문제가 많이 발생함.  
-> 새로운 post processing 방법 제안(fg pop-up, bg surpression)  

# 3. Our Approach
## 3.2 Discriminaitve Activation Layer (DA layer)
training data $\mathcal{I} = \{I_n, l_n\}_N\ \ I_n$: n-th image, $l_n \in \{0, 1\}^K$:corresponding image-level labels of $K$ categories.  

$I_n$ input, result of CNN backbone layer, feature maps $F_n \in \mathbb{R}^{C \times H \times W}$  
$F_n$ input, result of DA layer: class-specific actiation mask $M_n \in \mathbb{R}^{(K+1) \times H \times W}$, $K$ categories, 1 bg    
bg 영역이 fg영역으로 scattering over하는 것을 막기위해 bg 도입.  

activation mask가 probability of each pixel belonging to a corresponding category or bg를 indicate해야하므로 $L2\ norm\ along\ channel\ axis$ 수행.  
$$\hat{M}_n(i, j) = norm(\left| M_n(i, j)\right|)$$
$\hat{M}_n(i, j)$는 pixel-wise category distribution at position $(i, j)$  
$\hat{M}_n^k(i, j)$는 k-th value of the distribution vector.  $\in \mathbb{R}^{(1 X H X W)}$  
class-specific activation map of the image $I_n$ on category $k$: 
$$F_n^k = F_n \cdot \hat{M}_n^k\ \ F_n^k \in \mathbb{R}^{(C \times H \times W)},\ \ k \in [0, K]$$
bg는 앞으로의 procedure에 필요없어서 remove.  

## 3.3 Collaborative Multi-Attention Module
CMA module은 activation map들의 similar regions(collaborative information)을 highlight한다.  
self attention의 attention matrix가 query-key compatibility(적합성)을 indicate하는데에서 inspired -> self-attention machanism을 CMA module에 직접 적용.  
$\mathcal{F}^k = [F_1^k, F_2^k, ..., F_B^k] \in \mathbb{R}^{(B \times C \times H \times W)}$ : activation maps of $B$ images on $k$-th category.  
-> 1x1 conv 적용, channel을 $d$로 축소  
$\hat{\mathcal{F}^k} \in \mathbb{R}^{1 \times (B \times H \times W) \times d}$ => sequence of $B \times H \times W$ tokens.  

2 kind of positional encoding   
1. 각 susequence $\hat{F}_i^k \in \mathbb{R}^{(H \times W) \times d}$ 내에서 positional encoding을 위해(HW pixel의 위치) 1D positional encoding of length $H \times W$가 각 activation map에 embedded 된다.  
2. 전체 actiation map $\hat{\mathcal{F}}^k$에서 $B$개의 image를 구분하기 위한 $B$ positional encoding 이 inject된다. 이때 차원을 맞추기 위해(같은 image의 픽셀들에게는 같은 positional encoding이 들어가도록) $H\times W$회 반복되어 들어간다.  

이후 self attention module 실행. class-specific activation map에서 collaborative information 추출.  
self attention module의 output은 input activaton map과 같은 size.  
각 image의 각 category별로 output activation map에 GAP수행.  
input image가 multiple category를 가지므로 multiple Binary classification task로 간주하여 Multiple BCE loss 적용.  
$$ L_{cls} = {1 \over B \times K} \sum_{n=1}^B \sum_{k=1}^KL_{BCE}(Linear(GAP(A_n^k)), l_n^k)$$
$$ [A_1^k, A_2^k, ... A_B^k] = Self\ Attention(\hat{\mathcal{F}}^k)$$
$A_n^k \in \mathbb{R}^(C \times H\times W)$: output activation map of the image $I_n$ on $k$-th category  
$l_n^k \in [0, 1]$: gt label of image $I_n$ on $k$-th category  
각 self attention은 1개 category에 대한 class-specific activation maps를 input으로 받으므로 K independent self-attentions가 CMA module에 존재.  

## 3.4 Post-processing
saliency map을 image에서 [0, 255]사이 value로 extract하고 threshold $\theta$보다 작으면 bg로 설정.  

saliency map은 class-agnostic하고 most discriminative objects in the image를 highlight하는 경향이 있기 때문에 non-target objects를 fg로 treat할 수 있음  

maximum pixel-wise value of the corresponding region in the activation mask가 predefined threshold $\alpha$보다 작아야 salient region을 bg로 설정.  
otherwise, maximum pixel-wise value of the corresponding region in the activation mask가 predefined threshold $\beta$를 넘는다면 insignificant parts of the saliency map을 fg로 설정.  
________________
### Algorithm 1 Pseudo-labels Generation
______________
$\boldsymbol{Input}$: Normalized Activation Mask $\hat{M}$; Saliency map $S$; Category Number $K$; Threshold $\theta, \alpha, \beta$  
$\boldsymbol{Output}$: Pseudo Label Map $P$  
// Asume the background label is 0  
$P = \argmax_k (\hat(M)^k),\ \  where\ \  k \in [0, K]$  
$S_{i,j} = \begin{Bmatrix} 0,\ if\ \ S_{i,j} \lt \theta \\ 1,\ otherwise \end{Bmatrix}$  
$P_{i,j} = \begin{Bmatrix}\ 0,\ if\ S_{i,j} == 1\ \&\&\ \hat{M}_{i,j}^{P_{i,j}} \lt \alpha \\ P_{i,j},\ elif\ S_{i,j} == 0\ \&\&\ \hat{M}_{i,j}^{P_{i,j}} \gt \beta \\ P_{i,j} * S_{i,j},\ otherwise \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \end{Bmatrix}$  
__________________________
