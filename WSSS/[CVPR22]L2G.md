# [CVPR22] L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation
github: https://github.com/PengtaoJiang/L2G

# 1. Introduction 
classification model이 whole input image를 받을 때 보다 local image patch를 input으로 받을 때 discriminative region을 더 discover함을 관찰하였다.  

multiple local view에서 attention map을 생성하여 more details on undiscovered semantic region을 얻는다.  
knowledge transfer loss로 complementary(보충) attention knowledge가 gloa network에 online learning manner로 transfer된다.  

# 3 Method
## 3.2 Overall Framework
4개의 구성요소
1. Global network
2. Local network
3. attention transfer module
4. shape transfer module 

attention transfer module에서 2가지 loss 계산 
1. classification loss L_cls  
recognize semnatic objects
2. attention transfer loss L_at   
encourage the global network to imitate the local network to discover more discriminaitve regions  

shape transfer module에서는 L_st 계산  

L = L_cls + lambda * L_kt  
shape constraint가 없다면 L_kt = L_at. 있다면 L_kt = L_st  

## 3.3 Local to Glabal Attention Transfer 
image I에서 global view V_I, local view {V1, V2, ... VN} 얻음. local view는 randomly cropped from global view.   
{F1, F2, .. FN}: output of last conv layer of local network. C channel(class 개수)  
F^hat: output of last conv layer of global network. C+1 channel   
=> (background포함으로 1추가인듯) 

### 3.3.1 classification loss
classification loss는 local network에 장착  
feature maps {F1, F2, ... FN}을 global pooling layer를 통과시켜 1D feature vectors {f1, f2, ... fN} 생성.  
이후 sigmoid를 통과시켜 prediction qi 생성.  
L_cls는 각 f의 각 channel(class) 별 loss를 더한 값.  
GT는 1개의 image에서 나왔기 때문에 1개로 고정. (multi label은 가능)  

local network가 classification을 잘 하도록 유도

### 3.3.2 Attention transfer loss
global network가 local view attention을 받아들임  

local views에서 attention maps {A1^c, A2^c, ... AN^c} 생성 if c is in image-level label.  
If not, attention values in corresponding attention map will be zeroed.  

F^hat을 channel 방향으로 softmax -> G  
G^c는 각 위치가 class c로 분류될 probability  
{G1, G2, ... GN}을 G에서 {A1, A2, ... AN}와 같은 영역을 자른 것이라고 할 때 Gi와 Ai사이 mean square error 계산  
=> local view의 attention과 해당 영역에서 class c로 분류될 확률이 유사하도록 유도. 즉 local view의 detail한 부분을 activate한 것을 G도 class c로 분류하도록 유도.  

training 동안은 2개의 loss를 함께 사용하고 inference에서는 global network에서 생성된 attention map만 사용.  

### 3.3.3 Discussion
기존의 random cropping등의 data augmentation이 있지만 이 경우 cropped local patch의 정보를 global view에 쌓아올리는 것이 없다.  

## 3.4 Local to Global Shape Transfer  
object boundary가 sharp하지 않은 문제점 존재  
saliency map 도입  

attention map {Ai}를 small threshold(0.1 등) 로 binarize {Bi}. Attention map이 activate한 영역을 선택함.  
saliency model을 통해 saliency map S를 생성  
S에서 {Ai}와 같은 영역을 잘라 {Si} 생성 -> local saliency  
Bi와 Si를 elementwise multiplication하여 Gi와 mean square error 계산 -> L_st  
즉 local saliency 중 attention이 activate한 영역을 골라서 global network에게 학습하도록 전달함.  

saliency map이 있는 경우 위의 loss를 계산하고 없는 경우 기존의 L_at를 계산하여 더한다.  

# 4 Experiment  
## 4.1 Experimental setup
### 4.1.3 Data augmentation
global view: 448 * 448  
local view: 320 * 320  

## 4.2 Ablation study
### 4.2.1 Local view sampling strategy
uniform sampling vs random sampling  
9개의 uniform sampling과 9개 random sampling 결과에 큰 차이 없음  
flexibly adjust local view number N을 위해 random 선택  

### 4.2.4 L2G vs Sliding window
Global view에서 sliding window로 attention을 뽑고 그를 병합하여 local view의 정보를 가져올 수도 있다. 그러나 이 경우 original CAM보다 성능을 떨어뜨렸다.  
trained model이 global view를 base로 하기 때문에 non-discriminative object region을 찾는데 sliding window 방식이 맞지 않음을 알 수 있다.  

### 4.2.5 Classification loss in the global network
global network에 classification loss를 도입하게 되면 atttention map이 very small object region에 한정하여 생성된다.  
classifcaition loss와 attention transfer loss가 서로 반대 역할을 하는 것으로 추측함.  
classification lossㅡㄴ attention을 more discriminative하게 만든다.  

## 4.3 Comparison with SOTA
saliency map이 있으나 없으나 pseudo segmentation label의 성능이 더 좋음  
