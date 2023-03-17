# [CVPR22] Weakly Supervised Semantic Segmentation using Out-of-Distribution Data
github: https://github.com/naver-ai/w-ood

기존의 classifier가 foreground라고 confuse하는 것에 대한 Out-of-distribution (OoD) data, (foreground object class가 결여된 iamge)를 추가하여 학습  
'hard OoDs': classifier가 false-positive predictions를 하다록 하는 data. rail - train 관계 등.  

# 1. Introduction
Vision Dataset은 candidate image로 시작되어 foreground image가 정해지고 나머지가 Candidate OoD로 사용될 수 잇다.  
그러나 이를 바로 OoD로 이용할 수 없다.  
1. general OoD
images do not provide informative signals to distinguish difficult background cues from the foreground  
2. it may still contain foreground object

1번문제 해결: selecting hard OoDs whereby classifiers falsely assign high prediction scores to one of
the foreground classes  
2번문제 해결: human-in-the-loop process where images containing
foreground objects are manually pruned   
2번 문제를 해결하기 위해 effort가 필요함.  
그러나 적은 양의 hard OoD로도 성능이 높아짐. (1개만으로도 2%p)  
또 OoD를 모으는데 foreground sample을 모으는 것과 동일한 effort가 필요함.  

W-OoD : training classifier b utilizing the hard=OoDs.  
consider every hard OoD sample with a metric-learning objective:increase the distance btw the in-distribution and OoD samples in the feature space.  
-> Force the bg cues shared by the in-distribution and OoD samples(rail for train category) to be excluded from the feature-space representation.

# 3. Method
## 3.1 Collecting the Hard OoD Data
 
WSSS tsk with category labels를 생성할 때 아래 4단계로 생성.
1. define the list C of foreground classese of interest  
2. acquire unlabelled images from various sources (e.g. world wide web)
3. determine for each image whether it contains one of the foreground classes 
4. tag each image with the foreground category labels.  

candidate images obtained from step (2) but not selected in step (3). We refer to this set as the candidate OoD set. 

Candidate OoD에서 아쿠아리움의 물고기 처럼 train에 도움이 되지 않는 데이터 다수 존재.  
따라서 classifier trained on the images with foreground objects and the corresponding labels 를 이용하여 prediction score p(c)를 계산한 후 높은 순서로 rank.  
p(c)가 0.5이하인 image를 제거.  

이후 missing positive를 사람이 manually refinement 수행.  
target class가 포함된 image를 직접 수동으로 제거.  

## 3.2 Learning with Hard OoD Dataset
OoD를 'background' label하나로 할 수 있지만 그러면 diversity of OoD sample을 ignore하게 됨.  
in distribution classifier: F_in  
input x_in / x_oop  
feature: z_in / z_oop (F_in 의 feature)  
'cluster-based metric larning objective'  

Z_in 과 Z_ood를 z_in / z_ood의 set이라고 하자.  
먼저 Z_in에서 C개(class #)의 cluster 집합 P^in을 생성.  
이후 P^ood를 생성한다.  
incorrectly predicted class로 clustering을 할 수 있지만 heterogeneous한 경우가 있을 수 있다. 예를 들어 lake와 trees는 다르지만 'bird'에는 둘 모두 포함될 수 있다.  
따라서 K-means clustering on Z_ood로 P^ood를 생성한다.  

각 cluster의 center p_k.  
input x_in과 cluster Pk(P^ood의 cluster) 사이의 distance를 z(x)와 pk의 L2 norm으로 정의한다.  
x_in과 P^in의 dist가 가깝게, P^ood의 dist가 멀도록 loss를 design.  
Ld = d(x_in, Pc^in)의 총합 - d(x_in, Pk^ood)의 총합  
P^in의 class는 multi-hot binary vector of foreground classes in image x_in.  
P^ood의 class는 top-tau% closest from x_in.  

usual classification loss L_cls.  
x_in: BCE with label vector y.  
x_ood: BCE with zero-vector label y = (0, 0, ..., 0)  
L_cls = [L_BCE(F(x_in), yc) + L_BCE(F(x_ood), 0)]의 모든 class 평균  

L = L_cls + lambda * Ld  

## 3.3 Training SEgmentation Networks
학습한 Classifier F의 CAM에 IRN 사용.  

# 4. Experiments 
### 4.3.3 Analysis of Results by Class  
dining table에서 성능이 크게 떨어졌으나 이는 GT가 plate, food등의 것도 포함해서 labeling되어 있기 때문이다. 이것이 고쳐진다면 performance가 오를 것이다.  
