# [CVPR21] Railroad is not a Train: Saliency as Pseudo-pixel Supervision for Weakly Supervied Segmentic Segmentation

github: https://github.com/halbielee/EPS

WSSS using Image level weak supervision은 limitation을 가짐  
1. Sparse object coverage  
2. Inaccurate object boundaries  
3. Co-occurring pixels from non-target objects  

이 문제 해결을 위해 Explicit Pseudo-pixel Supervision (EPS) 제안  
EPS는 image-level label과 saliency map의 supervision을 combine한 pixel level feedback에서 learn한다.  
1. image-level label: localization map을 통해 object identity 제공  
2. Saliency map from the off-the-shelf saliency detection model: rich boundaries 제공  

2개의 information 사이에서 fully utilize the complementary(보충) relationship을 하기 위한 joint training strategy 사용.  
이 방법으로 accurate object boundaries를 얻고 co-occurring pixel을 discard할 수 있다.  

# 1. Introduction
WSSS 3 challenges  
1. Localization map only captures a small fraction of target objects  
2. it suffers from the boundary mismatch of the objects  
3. It hardly separates co-occurring pixels from target object  

기존 approach 종류
1. expands object coverage to capture the full extent of objects by erasing pixels, ensembling score maps, using self-supervised signal.  
-> fail to determine accurate object boundaries of the target object because they have no clue to guide the object's shape  
2. improving the object boundaries of pseudo-masks  
-> Since they effectively learn object boundaries, they naturally expand pseudo-masks until boundaries.   
However, they still fail to distinguish coincident pixels of non-target objects from a target object.  
3. aims to mitigate the cooccurrence problem using extra groundtruth masks, or the saliency map.  
-> require strong pixel-level annotations, which are far from a weakly supervised learning paradigm.  
-> sensitive to the errors of the saliency map.  
-> does not cover the full extent of objects and suffers from the boundary mismatch.  
등의 문제   

이 3가지 문제를 fully utilizing both the localizaiton map (CAM) and the saliency map (off-the-shelf saliency detection model)로 해결하고자 함.  

localization map: Can distinguish different object. Does not separate their boundaries effectively.  
Saliency map: Provide rich boundary information. Does not reveal object identity.  
서로 다른 information을 함께 사용.  

### Explicit Pseudo-pixel Sueprvision (EPS)  
saliency map을 fully utilize하기 위해 classifier가 C+1 class를 predict하도록 설계. (C target class + bg)  
C localization map과 bg localization map으로 saliency map estimate.  

Saliency loss: Pixel-wise difference btw saliency map, estimated saliency map.   
Saliency loss로 model can be supervised by pseudo-pixel feedback across all classes.  

Multi-label classification loss - predict image-level label.  

Classifier가 saliency loss와 multi-label classification loss를 모두 optimize하도록 학습.  
-> bg와 fg pixel을 모두 prediction하는데 synergizing 효과.  
이것을 통해 saliency map과 pseudo-mask 모두 improve함을 확인.  

saliency loss가 boundary mismatch를 pseudo-pixel feedback으로 penalize하여 object의 accurate boundary를 학습할 수 있게 함.   
또 map을 boundary까지 expand하여 entire object를 capture하게 됨.  
saliency loss가 fg와 bg를 분리하도록 도와서 co-occurring pixel을 bg로 assign하도록 함.  

# 3. Proposed Method
## 3.1 Motivation
Key insight: fully exploit 2 complementary information.   
Object identity from localization map + Boundary information from saliency map  
saliency map을 pseudo-pixel feedback to the localizaiton map으로 사용.  

EPS가 boundary mismatch와 co-occurrence problem을 모두 tackle할 수 있는 이유  
1. boundary mismatch  
fg map from C localization map과 fg of saliency map matching  
saliency map으로부터의 feedback으로 improve  boundary of object.    
2. co-occurring pixels of non-target object  
match bg localization map & saliency map.  
co-occuring pixel은 보통 background와 overlap.  

$L_{cls}, L_{sal}$ -> saliency map과 localization map에서 서로 missing information을 보충할 수 있다.  
original saliency map from off-the-shelf model이 missing and noisy information이 있더라도 우리의 estimated saliency map은 missing object도 포함하였고 noise도 줄였다.  

## 3.2 Explicit Pseudo-pixel Supervision
Merge localizaiton maps for target labels and generate a foreground map  
FG map: $M_{fg} \in \mathbb{R}^{H \times W}$  
BG label: $M_{bg} \in \mathbb{R}^{H \times W}$  
Estimate Saliency map: $\hat{M}_s$  
$$\hat{M}_s = \lambda M_{fg} + (1-\lambda) (1-M_{bg})\ ,\ \lambda \in [0, 1]$$
$\lambda$ is hyperparameter  
saliency loss $L_{sal}$: sum of pixel-wise difference between estimated saliency map and actual saliency map  

Pre-trained model의 사용은 Weakly Supervised Learning으로 여겨진다.  
따라서 saliency map의 사용이 comman practice in WSSS에서 widely accepted하다.  
Fully supervised saliency detection model은 pixel-level annotation을 사용한다는 점에서 논쟁이 있을 수 있다.  
우리 방법론은 Unsupervised, fully supervised saliency detection model 무엇을 사용하던 모두 fully supervised를 사용한 다른 방법론을 뛰어 넘었다.   

### 3.2.1 Map selection for handling saliency bias  
naive selection rule may not be compatible with the saliency map computed by the off-the-shelf model.  
Saliency map from PFAN[51]은 some objects as salient objects를 자주 ignore한다.  
이 Systematic error는 saliency model이 statistics of different datset을 학습하기 때문에 피할 수 없다.  

이 systematic error를 tackle하기 위해 localizaiton map과 saliency map의 overlapping ratio를 사용하는 방법을 제시한다.  
i-th localization map $M_i$가 saliency map과 $\tau \%$이상 overlap 해야 fg로 assign. otherwise bg로 assign.  
모든 fg localization map은 해당 class c가 image에 존재해야 영향을 줌   
$$ M_{fg} = \sum_{i=1}^C y_i \cdot M_i \cdot \mathbb{1}[O(M_i, M_s) \gt \tau]$$
$$M_{bg} = \sum_{i=1}^C y_i \cdot M_i \cdot \mathbb{1}[O(M_i, M_s) \le \tau] + M_{C+1}$$
$y \in \mathbb{R}^C$: binary image-level label  
$O(M_i, M_S)$: Compute overlapping ratio  
$$ O(M_i, M_S) = {|B_i \cap B_s| \over |B_i|}$$
$$ for\ pixel\ p,\ B_k(p) = \begin{Bmatrix} 1\ \ if\  M_k(p) \gt 0.5 \\ 0\ \ otherwise \end{Bmatrix}$$

## 3.3 Joint Training Procedure  
saliency loss $L_{sal}$: average pixel-level distance btw $M_s$ and $\hat{M}_s$  
$M_s$: actual saliency map (PFAN model trained on DUTS dataset 사용)  
$\hat{M}_s$: estimated saliency map  
$$L_{sal} = {1 \over H \cdot W} \lVert M_s - \hat{H}_s \rVert^2 $$

classification loss $L_{cls}$: multi-label soft margin loss btw $y$ and $\hat{y} \in \mathbb{R}^C$
$y$: image-level label  
$\hat{y}$: prediction, result of GAP on localization map for each target class   
$$L_{cls} = - {1 \over C}\sum_{i=1}^Cy_i log\sigma(\hat{y}_i) + (1-y_i)log[1 - \sigma(\hat{y}_i)]$$
$\sigma()$: sigmoid  
$$L_{total} = L_{cls} + L_{sal}$$
$L_{sal}$는 bg 포함 C+1개 class에 대해서 parameter update.   
$L_{cls}$는 bg제외 C개 class에 대해서만 evaluate하여 parameter update.  

# 5. Experimental Results
## 5.1 Handling Boundary and Co-occurrence
### 5.1.1 Boundary mismatch problem
utilize SBD[17], which provides boundary annotations and the
boundary benchmark in PASCAL VOC 2011  
[32]에서 한 것과 같이 quality of the boundary는 class agnostic manner로 Laplacian edge detector로부터 edge of pseudo-masks를 계산하여 평가.  
Recall, Precision, F1 score, comparing predicted and gt boundaries.  

### 5.1.2 Co-occurrence problem
IoU for the target class <-> confusion ratio btw target class and its coincident class 비교  
confusion ratio는 얼마나 coincident class가 target class로 잘못 predict되었는가를 계산  
confusion ratio $m_{k, c} = FP_{k,c} / TP_c$  
$FP_{k,c}$: number of pixels mis-classified as the target class $c$ for the coincident class $k$  
$TP_c$: number of true-positive pixels for the target class $c$  

