# [CVPR21] Distilling Dnowledge via Knowledge Review

github: https://github.com/dvlab-research/ReviewKD

Knowledge distillation: transfer knowledge from teacher network to student network.  
goal: greatly improving the performance of the student network  

previous method: Focus on proposing feature transformation and loss functions between the $\mathbf{same\ level's\ features}$ to improve the effectiveness.   
Ours: factor of connection path $\mathbf{cross\ levels}$ between teacher and student network  

1. Cross-stage connection path  
2. new review machanism  
3. nested and compact framework

# 1. Introduction
techniques for training fast and compact neural networks
1. Network pruning
2. quantization
3. knowledge distillation  

train small network (student) under the supervision of a large network (teacher)  
[9]에서 knowledge is distilled through the teacher's logit.  
student is supervised by both GT labels and teacher's logit.  

## 1.1 Our New Finding
Connection path beteen teacher and student  

How previous work deals with these paths:  
$\mathbf{ALL}$ previous methods: only use same-level information to guide student.  
이것은 intuitive and easy to construct.  
그러나 우리는 이것이 bottleneck in the whole knowledge distillation framework라는 것을 밝혔다.  

Key modification: use low-level features in the teacher netwrok to supervise deeper features for the student.   

발견 요소2: student high-level stage has the great capacity to learn useful information from the teacher's low-level features.  

## 1.2 Our knowledge Review Framework
Use multi-level information of teacher to guide one-level learning of student.  
Review machanism is to use previous features to guide the current feature.  
Student는 항상 refreshing understanding and context of "old knowledge"를 위해 무엇을 studied 했는지 check 해야한다.  

teacher의 multi-level information에서 어떻게 useful information을 extract할지, 어떻게 이것을 student에게 transfer할지가 challenge problem.  
1. Residual learning framework to make the learning process stable and efficient.  
2. Attention based fusion (ABF) module  
3. Hierarchical context loss (HCL) function  

# 3. Our Method
## 3.1 Review Machanism
input image $X$  
student network $\mathcal{S}$  
$Y_s = \mathcal{S}(X)$: output logit of student  
$\mathcal{S}$는 $(\mathcal{S}_1, \mathcal{S}_2, \ldots \mathcal{S}_n, \mathcal{S}_c)$의 different part로 구분  
$\mathcal{S_c}$: classifier, $\mathcal{S}_1, \ldots \mathcal{S}_n$: different stage seperated by downsample layer.  
$$Y_s = \mathcal{S}_c \circ \mathcal{S}_n \circ \cdots \circ \mathcal{S}_1(X)$$  
$"\circ"$: nesting of functions.  $g \circ f(x) = g(f(x))$  
$(F_s^1, ... , F_s^n)$: intermidate features  
$$F_s^i = \mathcal{S}_i \circ \cdots \circ \mathcal{S}_1(X)$$  

teacher network $\mathcal{T}$  
process는 student와 동일  

single-layer knowledge distillation:
$$ L_{SKD} = \mathcal{D}(\mathcal{M}_s^i(F_s^i),\ \mathcal{M}_t^i(F_t^i))$$
$\mathcal{M}$: transformation that transfers the feature to target representation of attention maps or factors.  
$\mathcal{D}$: distance function that measures the gap btw student, teacher  

Multi-layer knowledge distillation:
$$L_{MKD} = \sum_{i \in \mathbf{I}}\mathcal{D}(\mathcal{M}_s^i(F_s^i),\ \mathcal{M}_t^i(F_t^i))$$
$\mathbf{I}$: stores the layers of features to transfer knowledge.  

### Review Machanism
use previous features to guide the current feature  

single-layer knowledge distillation with review machanism:
$$L_{SDK\_R} = \sum_{j=1}^i\mathcal{D}(\mathcal{M}_s^{i,j}(F_s^i),\ \mathcal{M}_t^{j,i}(F_t^j))$$
student의 feature는 $F_s^i$로 고정되어 있고, $F_s^i$를 guide하기 위해 teacher의 첫 $i$ levels of features를 사용한다.  

Multi-layers knowledge distillation with review mechanism:
$$L_{MDK\_R} = \sum_{i \in \mathbf{I}} \left(  \sum_{j=1}^i\mathcal{D}(\mathcal{M}_s^{i,j}(F_s^i),\ \mathcal{M}_t^{j,i}(F_t^j)) \right)\ \ (6)$$

$$ L = L_{CE} + \lambda L_{MKD\_R}$$

## 3.2 Residual Learning framework
1. Figure 2(a)    
    $L_{SDK\_R}$을 straightforward framework로 표현.  
    $\mathcal{M}_s^{i,j}$는 simply composed of conv layers and nearest interpolation layers to transfer the $i$ th feature of student to match the size of teacher's $j$ th feature  
    teacher feature $F_t$ 는 transform 안함.  
    Student의 ith feature가 teacher의 1~i feature와 비교됨. -> teacher의 1~i feature로부터 student ith feature가 학습.  

2. Figure 2(b)  
    $L_{MDK\_R}$ with all-stage feature distilled 표현.  
    그러나 이 strategy is not optimal because of the huge information difference btw stages.  
    Also too complicated process.  
    n stage needs n(n+1)/2 pairs calculation  

3. Figure 2(c)  
    단순화를 위하여 transform of feature를 생략하여 equation (6)를 변경  
    $$L_{MDK\_R} = \sum_{i \in \mathbf{I}} \left(  \sum_{j=1}^i\mathcal{D}(F_s^i,\ F_t^j) \right)\ \ (8)$$
    $i$와 $j$의 summation 순서 변경  
    $$L_{MDK\_R} = \sum_{j=1}^n \left(  \sum_{i=j}^n\mathcal{D}(F_s^i,\ F_t^j) \right)\ \ (9)$$
    $j$가 fix되면 equation (9)는 teacher feature $F_t^j$와 student feature $F_s^j-F_s^n$ 사이의 Distance 계산  
    distance의 summation을 fused feature의 distance로 approximate  
    $$\sum_{i=j}^n\mathcal{D}(F_s^i,\ F_t^j) \approx \mathcal{D}(\mathcal{U}(F_s^j, \cdots, F_s^n), F_t^j)\ \ (10)$$  
    $\mathcal{U}$: module to fuse features  
    이 approximation이 figure 2(c)에 나타남.
4. Figure 2(d)  
    fusion calculation이 progressively(점진적) manner로 optimize 가능.  
    Fusion of $F_s^j, \cdots, F_s^n$ $\Leftrightarrow$ combination of $F_s^j$, $\mathcal{U}(F_s^{j+1}, \cdots, F_s^n)$  
    fusion operation is recursively defined as $\mathcal{U}(\cdot, \cdot)$  
    $F_s^{j+1,\ n}$ : fusion of features from $F_s^{j+1}$ to $F_s^n$  
    $$L_{MDK\_R} = \mathcal{D}(F_s^n, F_t^n) + \sum_{j=n-1}^1 \mathcal{D}(\mathcal{U}(F_s^{j+1}, \cdots, F_s^n), F_t^j)\ \ (11)$$
    loop from n-1 down to 1 to make use of $F_s^{j+1,\ n}$  
    $F_s^{n,n} = \mathcal{M}_s^{n,n}(F_s^n)$  

    Distillation process with utilizing the concept of "residual learning".  
    student의 stage 4 feature는 teacher의 stage 3 feature를 mimic(흉내내다)하기 위해 student의 stage-3와 aggregate된다.  
    따라서 student의 stage 4 feature는 residual of stage-3's feature btw student and teacher를 학습한다.  

## 3.3 ABF and HCL
ABF: Attention Based Fusion  
HCL: Hierarchical Context Loss  

1. ABF (Figure 3(a))  
    higher level feature가 lower level feature와 같은 size로 resize  
    2개의 feature가 concat되고 1x1 conv로 2개의 $H \times W$ attention maps 생성  
    각 attention map이 각 feature에 곱해진 후 더해짐  
2. HCL (Figure 3(b))  
    2개의 feature map사이 loss function으로 $L_2\ distance$ 사용  
    $L_2$ distance는 same level feature 사이 information transfer에 effective.  
    그러나 our feature는 different level info is aggregated.  
    [41]에서 Inspired, $Spatial\ Pyramid\ Pooling$을 통해 separate the transfer of knowledge into different level's context information.  

    spatial pyramid pooling으로 different levels' knowledge 추출  
    -> $L_2$ distance를 각각 적용하여 distill  



