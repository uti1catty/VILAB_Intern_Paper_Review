# [CVPR21] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation  

project: https://cvlab.yonsei.ac.kr/projects/BANA  
github: https://github.com/cvlab-yonsei/BANA

background regions are perceptually consistent in part within an image 라는 사실을 확인함.  
And this can be leveraged to discriminate fg and bg regions inside object bounding boxes.  
-> novel pooling mehtod, 'background-aware pooling (BAP)'  
여전히 object boundary에 noise  
-> noise-aware loss (NAL) that makes the networks less susceptible to incorrect labels.  

# 1. Introduction

1. How can we generate high-quality but possibly noisy pixe-level labels from objet bounding boxes?  
=> leverage a CNN for image classification, instead of exploiting off-the-shelf mehtods.  
bg에 해당하는 feature를 버리고 fg feature만 aggrigate하도록 하고, more accurate CAM을 가져온다.  
BB내의 bg를 찾아 attention map을 만들고 이를 CAM과 함께 사용하여 pseudo-gt를 만든다.  
BAP with attention map, enabling discriminating(구분) fg and bg inside bb. 

2. How can we train CNNs for semantic segmentaiton?  
=> noise-aware loss (NAL) to train CNNs for semantic segmentation that makes the networks less susceptible(흔들리는) to incorrect labels.  
CNN feature for prediction와 classificer weights for semantic segmentation 사이의 distance를 사용하여 confidence map을 만들고 CE loss를 계산한다.  

# 3. Approach
1. train a CNN for image classification using object Bunding boxes.  
use BAP leveraging a bg prior, that is, bg regions are perceptually consistent in part within an image, allowing to extract more accurate CAMs.  
(more accurate CAMs를 추출하는 BAP를 사용하는데, 이 BAP는 bg prior, bg reigons가 image의 part로 반드시 존재한다는 prior에 영향력을 행사한다.)  
후에 각 image마다 bg를 위한 attention map을 계산한다.  
2. pseudo segmentation labels using CAMs obtained from the classifciation network together with the bg attention maps and prototypical features.  
3. train CNNs for semantic semgentaiton with pseudo gt. use NAL to lessen the influence of the noisy labels.  

## 3.1 Image classifcaiton using BAP
classification network: feature extractor & (L+1) way softmax classifier.  
feature extractor output feature map $f$.  
$B = \{B_1, B_2, ..., B_K\} $ a set of objet bounding box. $f$ size로 resized by nearest-neighbor interpolation.  
$M$: mask indicating a definte bg. (outside the BB)  
$M(p)=1$ if position $p$가 어떤 BB에도 포함되지 않을때  

### 3.1.1 Background attention map
BB내에서 fg와 bg를 구분하면 classifier가 fg object에 더 집중하도록 할 수 있다.  
그러나 BB내에는 fg와 bg가 섞여있으나 object boundary에 대한 정보가 없다.  
BB에서 fg와 bg를 구분하기 위해 이 문제를 retrieval task (회복 문제)로 바라본다.  
feature map $f$를 $N \times N$ regular grid로 나눈다.  
각 grid cell: $G(j),\ 1 \le j \le N^2$. BB내에 완전히 속하는 cell은 무시.  
각 grid cell의 feature를 queries for retrieval로 사용.  
$$ q_j = {\sum_{p \in G(j)}M(p)f(p) \over \sum_{p \in G(j)}M(p)}$$
즉 $q_j$는 $f$의 cell $G(j)$ 내의 pixel값들 중 bg에 해당하는 값들의 평균값.  

queries를 이용하여 BB내의 bg region을 retrieve.  
attention map $A$ (최종적으로 bg영역이라 판단하는 attention map)  
$$A(p) = {1 \over J }\sum_j A_j(p),\ J: number\ of\ valid\ grid\ cells$$ 
$$ A_j(p) = ReLU({f(p) \over ||f(p)||} \cdot {q_j \over ||q_j||})\ (p \in B),\ or\ 1\ (p \notin B)$$
즉 $A_j(p)$는 $p$가 bg 영역이면 1을 갖고 fg영역(BB내부)이면 bg feature $q_j$와 해당 position의 feature $f(p)$사이의 cosine similarity 값을 가진다. 이때 ReLU로 음수를 잘라내어 [0,1]값을 가짐.  
$A(p)$는 이러한 $A_j(p)$들의 평균값. 따라서 $A(p)$가 bg 영역이라 판단하는 attention map이 된다.  
### 3.1.2 BAP
$A$를 이용하여 각 BB내의 forground feature $r_i$계산  
$$r_i = {\sum_{p \in B_i} (1-A(p))f(p) \over \sum_{p \in B_i}(1-A(p))}$$
즉 $r_i$는 BB내에서 1-A에 의해 fg라고 판단될 확률을 weight으로 하는 weighted average pooling.  

### 3.1.3 Loss
fg feature $r_i$와 bg feature $q_j$를 이용하여 (L+1) way softmax classifier $w$. standard CE loss 적용.  
코드를 확인하니 $r_i$, $q_j$를 dim=0방향으로 concat하여 nn.Conv2d(1024, num_classes, 1, bias=False)를 통과.  
즉 1x1 conv로 1024->num_class로 dim 줄인 후 이 값과 bg-fg gt를 nn.CrossEntropyLoss()에 넣어 loss 계산.  

## 3.2 Pseudo label generation
먼저 DenseCRF를 사용하기 위한 unary term 계산  
$$ u_c(p) = {CAM_c(p) \over max_p(CAM_c(p))}\ if\ p \in B_c\ or\ 0\ if\ p \notin B_c $$
$$ B_c: a\ set\ of\ bounding\ boxes\ containing\ objects\ of\ the\ cass\ c$$
$$ CAM_c(p) = ReLU(f(p) \cdot w_c) $$
$w_c$ 는 $r_i$, $q_i$를 넣고 계산한 classifier의 weight.  
이 값이 각 dim의 class c에 대한 기여도를 담고 있기 때문에 이 값과 feature map $f$를 통해 CAM을 계산한다.   
즉 class c를 포함한 BB에서 CAM_c의 값을 normalize  

$$u_0(p) = A(p)$$

$u_c$들과 $u_0$을 concat하여 DenseCRF 수행-> $Y_{crf}$ 생성  
그러나 이 값은 low-level feature in the pairwise term이 incorrect될 수 있다.  

prototypical feature for each class $q_c$계산  
$$ q_c = {1 \over |Q_c|}\sum_{p \in Q_c}f(p)  $$
$Q_C$: set of locations labeled as the class c in $Y_{crf}$
즉 $q_c$는 CRF가 class c라고 판단한 영역의 feature f값의 평균-> class c의 prototype feature가 됨.  

이 값을 이용하여 feature map f의 correlation map을 계산  
$$C_c(p) = {f(p) \over ||f(p)||} \cdot {q_c \over ||q_c||}  $$
$C_c$는 feature map f의 모든 pixel에 대해서 q_c와의 유사도를 담고 있다.  
$C_c$에 argmax를 적용하면 가장 class c와 연관성 있다고 판단된 class가 선택된다.  
따라서 $C_c$에 argmax를 적용하여 pseudo segmentation label $Y_{ret}$ 를 계산한다. 

## 3.3 Semantic Segmentation with noisy labels
DeepLab. 끝에서 2번째(penultimate layer)에서 feature map $\phi$를 얻어 softmax classifier $W$를 통과해 $(L+1)$ dimension probability map $H$를 생성.  
$Y_{crf}$와 $Y_{ret}$에 같은 label인 영역을 $S$라고 하고 이 영역에 대해 CE loss 계산.  
$$ L_{ce} = - {1\over \sum_c|S_C|}\sum_c \sum_{p \in S_c}logH_c(p)
$$

~S, Y_{crf}와 Y_{ret}가 서로 다른 label을 가리키는 영역에 대해서도 S보다 less relialble 하지만 correct를 담고 있을 수 있다.

[39], [45] 가정: classifier weight은 feature space에서 각 class의 center를 represent한다. 따라서 이 weight을 representative feature for the corresponding class 로 바라볼 수 있다.   

correlation map btw CNN feature, classifier weights  
$$D_c(p) = 1 + ({\phi(p) \over ||\phi(p)||} \cdot {W_c \over ||W_c||})$$
$W_c$: classifier weight corresponding class c  
cosine similarity를 사용했으며 +1: correlation이 양수가 되도록 함.  

cofidence map $\sigma$계산  
$$\sigma(p) = ({D_{c*}(p) \over max_c(D_c(p))})^{\gamma}\ \ \ c*=Y_{crf}(p)\ \ (\gamma \ge 1): damping\ ratio  $$
confidence map은 likelihood of each label being correct  
$D_{c*}(p)$와 $max_c(D_c(p))$가 유사할 수록 label $c*$가 confident하다는 것을 의미.  
$\gamma$가 무한대이면 binary가 된다.  
confidence map을 weighting factor로 하여 CE loss 계산  

$$ L_{wce} = - {1\over \sum_c \sum_{p \in \sim S_c}\sigma(p)}\sum_c \sum_{p \in \sim S_c}\sigma(p)\ logH_c(p)$$
$$L = L_{ce} + L_{wce}$$