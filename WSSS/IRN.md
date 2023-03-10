# Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations [CVPR19]

github: https://github.com/jiwoon-ahn/irn

## 1 Introduction
pseudo instance segmentation label 생성  
CAM을 도와주기 위해 Inter pixel Relation Network (IRN) 도입  
class-agnostic instance map & pairwise semantic affinities  
class-agnostic instance map: class label과 accurate boundary없이 rough instance segmentation mask  
pairwise semantic affinity: confidence score for class equivalence between pair of pixels  

instance augnositc CAMs + class agnositc instance map -> instance wise CAMs  
instance wise CAMs + semantic affinities -> propagate attention scores to relevant areas  
각 pixel에서 가장 높은 attention score in the instance-wise CAMs 로 instance label 선택  

## 2 Related Work
### weakly supervised instance segmentation
보통 bounding box를 weak label로 사용. boudning box가 위치와 scale을 알려주기 때문에 이를 사용하는 model은 object shape 추정에 집중  

## 3 Class activation maps
classification network로 ResNet50사용. 마지막에 downsampling layer의 strie를 2에서 1로 줄여 resolution drop을 줄였다.  

## 4 Inter-pixel Relation Network IRNet
displacement vector field / class boundary map 도출  
### 4.1 IRNet Architecture
2개의 branch가 같은 ResNet50 backbone을 공유 -> Sec3 CAM의 backbone과 동일  
FIgure 2  

### 4.2 Inter-pixel relation mining from CAMs
IRNet 학습을 위해 inter-pixel relation을 사용한다. 이를 CAM에서 추출  
CAM에서 attention score를 통해 confident fg/bg 확인  
각 confident area는 dense CRF를 통해 refine  
각 pixel의 best score class를 골라 pseudo class map M_hat을 설정  
neighboring pixels를 sampling.  
set Pfg+, Pbg+, P-로 구분  
P는 radius r 내에서 있는 모든 feature 쌍을 포함 (i, j) i,j는 i,j번 feature  
P+는 P내의 쌍 중에 같은 class map에 속하는 쌍을 포함(i, j)  
P-는 P내의 쌍 중에 다른 class map에 속하는 쌍을 포함(i, j)  

### 4.3 Loss for Displacement Field Prediction
field D (w x h x 2) 각 2D vector는 속하는 instance의 centroid를 가리킨다.
즉 변위 field로서 (i, j)의 2D vector는 (i, j) pixel이 속하는 instance의 centroid를 가리키는 방향 vector이다.   
각 instance의 centroid GT는 주어지지 않지만 같은 class의 pixel들 사이의 변위로부터 implicitly learned가 가능하다고 주장한다.  
Displacement field가 되기 위해 같은 instance내의 feature i, j에 대하여 
1. xi+D(xi) = xj+D(xj) (같은 centroid 가리킴) 
2. Sigma_x(D(x)) = 0 (centroid 정의)  

을 만족해야 한다.  

condition 1 만족  
P+내의 (i, j)에 대하여 i, j는 작은 r 내에서 추출하였으므로 같은 instance라고 가정  
#### (Q) 두개의 instance가 겹친 상황이라면 같은 instance로 편입될텐데 이는 어떻게 해결?

coordinate displacement delta_hat(i, j) = xj - xi  
difference in D delta(i, j) = D(xi) - D(xj)  
ideal의 경우 위의 2개가 identical. 따라서 두 차이를 Loss L1로 두고 이를 최소화한다.  
L_fg^D (eq 5)

condition 2 만족  
random parameter intialize로 인해 직관적으로 이미 condition 2를 만족하며 만족한 상태로 local minima를 찾는다고 볼 수 있다.  

Background의 centroid는 존재하지 않으므로 Pbg+의 delta(i, j)합을 Loss로 두고 최소화 한다.  (eq 6)  

### 4.4 Loss for Class Boundary Detection
output: B [0, 1]^(w x h)  

핵심 가정: class boundary exists somewhere between a pair of pixels with different pseudo class labels.  

(eq 7) xi, xj 사이에 line이 존재하면 B(xk)최대값이 클 것 -> affinity값인 aij는 작아져서 xi, xj는 affinity가 작아짐. line이 없다면 B(xk) 최대값이 작아 aij값이 커져서 유사도가 큼을 의미할 것.  

Class Boundary detection을 위한 input으로 CAM으로부터 pseudo class label을 생성했었음 (4.2)  
(같은 class라면 1, 다른 class 라면 0 표시 가능)   
이 결과와 aij결과를 비교하는 cross-entropy loss 사용  
(eq 8) Pfg+, Pbg+ 내의 쌍은 같은 class의 쌍 (line 51 가정) 따라서 log(aij)  
P-내의 상은 다른 class의 쌍. 따라서 log(1-aij)  

### 5.1 Generting class agnostic instance map
D에서 같은 centroid를 가리키는 것끼리 묶는다-> I  
이때 iterative하게 centroid를 향하는 D값을 변경한다. (eq 10)  

### 5.2 synthesizing isntance segmentation labels
clas별 CAM에서 I를 적용하여 instance wise CAM 생성.  
random walk로 propagation