# [ECCV22] Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation

github: https://github.com/KAIST-vilab/AEFT

## 1. Introduction
GAP (Global Average Pooling)은 feaure에서 object irrelevant region까지 포함하여 averaging를 하므로 CAMs는 작은 영역을 무시하고 object boundary에 fit하지 않는다.  
이를 해결하기 위해 GPP (Gated Pyramid Pooling)을 사용해 global context를 capture함과 동시에 fine-details를 localize하였다.  
CAM이 various bin size에 따라 average pooled 되어 spatial pyramid를 형성  
pyramid가 gating machanism에 따라 aggregated sequentially  
aggregation의 output이 image level class 예측을 위한 contribution of CAMs의 pixel level weight으로 사용된다.  
low-scale bins: global context / high-scale bins: localize fine-details  

AE method for guide CAMs to be activated on less-discriminative region  
Triplet을 도입하여 AE 변화  
anchor: original Image / Positive: erased iamge / Negative: no overlapping class image  
anchor와 negative의 distance가 이미 멀기 때문에 의도적으로 anchor에서 High confidence region을 제거하였다.  
anchor-positive: CAMs to explor less-discriminative region  
anchor-negative: prevent over-expansion problem  

>Forcing the network to make the prediction from the erased image according to the binary classification lable is the main reason for over-expansion  
이를 해결하기 위해 Triplet loss 사용  

## 3. Proposed method
### 3.2 CAMs Generation
GAP를 통하면 위치와 관계없이 모두 동등한 기여로 계산  
따라서 GAP가 image pixels and image level clss labels 사이의 잘못된 correlation을 학습시킨다.  
Resulting CAMs tend to be activated on highly correlated background regions while ignoring the small objects.  
### 3.3 Gated Pyramid Pooling (GPP) layer
low to the high scale, sequentially refine the pooled feature map with multiple gated conv layers while preserving its sign.  

Defined 'sign-preserving attention operation g  
positive prediction: model to decide 'existence of the class'
negative prediction: model to decide 'non-existence of the class'

sign-preserving attention operation g(x, y):  
x, y에 각각 ReLU 시행한 후 concat, 이후 3X3 Conv, 이후 sigmoid 수행.

rXr pooling: rXr 영역마다 average pooling 시행  
모두 input feature map f resolution이 되도록 upsampling  

positive attention alpha, negative attention beta  
이전 수행한 P_hat과 다음녀석의 g 수행 -> alpha  
이전 수행한 P_hat과 다음 P의 -붙여서 g 수행 -> beta  
alpha와 beta의 channel 1은 앞의 녀석으로부터 파생, channel 2는 뒤의 녀석을부터 파생

다음 P_hat = (이전 P_hat의 양수 X alpha_1channel + 다음 P의 양수 X lpha_2channel) / 2 - (이전 P_hat의 음수 X beta_1channel + 다음 P의 음수 X beta_2channel) / 2  
앙수만 추리는 방법: ReLU 수행 / 음수만 추리는 방법: - 붙여서 ReLU 수행  
X는 elementwise product  
즉 양수부분끼리의 attention정도에 따른 평균 과 음수부분끼리의 attention정도에 따른 평균을 병합  
equation 2~4  

f와 f로부터 파생되는 모든 P는 Kxwxh (K:class #)  

최종 prediction p는 P16_hat과 f로부터 계산  
P16_hat과 f의 양수부분끼리 elementwise 곱과 P16_hat과 f의 음수부분끼리 elementwise 곱 병합 (음수 곱에 -붙여 음수로). 이후 channel별로 average하여 sigmoid  

GPP를 통해 GAP 대비 higher localization quality CAMs 획득  

### 3.4 Adversarial Erasing Framework via Triplet(AEFT)
기존 AE: AE에서 most discriminative 영역을 삭제한 후 해당 image로 다시 model이 original image-level classification label에 따라 분류하도록 학습.  
반복학습을 통해 model이 less discriminative region에 집중하게 되고 CAMs도 함께 확장   
over exansion problem 존재  

AE의 직접적인 CAMs 학습은 generated CAMs의 quality측면에서 unstable하게 만든다고 판단  
'erase'를 학습시키는 metric learning으로 triplet loss 시행  

original image Anchor, masked image Positive, class겹침 없는 다른 image Negative  

#### 3.4.1 Masked Image 획득
CAMs로부터 forground map Afg획득: 각 class마다의 activation map의 max value 모음  
masked image Ip(i,j): Afg(i,j)가 t_H이상이면 0 (Hard masking)  
미만이면 original image I_A(i,j) X Afg(i, j) (original image에서 acivation 정도를 주며 soft masking)  
t_H: threshold - hard masking과 soft masking 구분 경계 (eq 7)  

#### 3.4.2 Adversarial Ersing via Metric Learning
Anchor, Positive, Negative를 GPP 수행한 후 P16_hat을 embedding (channel마다 평균값) 으로 변경하여 embedding상에서 비교   
embedding은 feature map을 channel마다 평균낸 class 별 정보를 담고 있는 vector  
Anchor의 feature와 Positive의 feature는 가깝게, Negative feature는 멀리 둠  

Anchor는 target class에 대한 activation이 높을 것이고 Positive는 high discriminative가 없어졌으므로 target class에 대한 activation이 낮다. 그럼에도 불구하고 distance가 작아져야 하므로 model은 target class를 위한 약한 activation 부분이 이 class의 영역임을 학습하고 해당 영역의 activation 강도를 키우도록 강요받는다.  
다음 iteration에서는 강요받은 부분까지 high discriminative로 판단하여 positive에서는 제거되어 날아가고 model은 약해진 activation을 다시 값을 키우기 위해 less discriminative부분을 찾아 activation을 키우도록 강요받는다.  
이러한 반복을 통해 CAMs 영역을 확대해 나간다.  

그러나 이렇게 될 경우 모든 영역이 잘려나가더라도 model은 추가적인 공간을 찾도록 강요받아 over-expansion문제가 발생한다. 이를 막기위해 Negative를 사용  

먼저 I_A의 GPP결과인 P16_hat에서 t_L(threshold) 영역보다 작은 영역만 선택하여 channel별 평균으로 embedding e_AL을 형성한다. e_AL은 Original Image의 low confidence 영역만 추린것을 의미하게 된다.  
이 e_AL과 e_N의 distance를 maximize시킨다.  
over-expansion이 되면 low confidence영역이 image의 objects에 대하여 더 적은 information을 갖게 되어 e_AL이 더 적은 정보를 갖게 된다. 그러면 model은 e_AL과 e_N을 구분하기가 어려워진다. 따라서 model은 이 distance가 최대가 되도록 강요받음으로서 over-expension을 누르도록 유도된다.  
-> 이 부분이 설득력이 좀 떨어지는 것으로 보임. original image의 low confidence영역은 결국 배경 영역일 것이고 다른 class들에 대한 activation info가 거의 없을 것. 그러나 Negative image는 특정 class에 대한 확실한 activation이 있기 때문에 이미 distance가 클 것이고 model이 특정 class embedding을 멀리 가져다 두기만 하면 되기 때문에 그렇게 학습이 될 수 있다. 또한 low confidence 영역과 negative image를 비교하는 것은 결국 background와 다른 class를 구분짓도록 학습하여 background에서 해당 class를 activation하지 못하도록 하는 것이 아닌가 생각됨  

최종 loss: binary cross entropy loss (class prediction <-> image level lable) + lambda1*L_attract + lambda2*L_Repel  

## 4 Experiments
### 4.1 Dataset
PASCAL VOC 2012: train(1456)으로 학습, val(1449)/test(1456)으로 evaluate  
MS-COCO 2014: train(80k)로 학습, val(40k)로 평가. COCO-Stuff dataset에서 GT segmentation label을 얻음. MS-COCO 2014는 일부 object사이 overlap 존재. 

### 4.2 Implementation detail
ResNet38 backbone, ImageNet parameter로 initialize  
data augmentation: horizontal flipping, color jittering, cropping  
sementic segmentation network: Deeplab with ResNet38 backbone  
### 4.3 Ablation study
GPP bin의 사이즈가 커질 수록 mIoU가 커짐  
GPP bin 여러개를 모두 averaging으로 했을 때 mIoU가 커짐  
GPP bin 여러개를 gated conv로 할 때 coarse->fine은 mIoU가 커짐/fine->coarse는 작아짐  
-> global context와 fine details가 small, large bin size에서 모두 잘 보존되었으며 coarse->fine 순서가 이를 잘 aggregation함  

AEFT에서 attract Loss 단일 사용, repel Loss 단일 사용 모두 mIoU의 상승이 일어남.  
repel loss가 background가 다른 원치 않은 class로 activation되는 것을 막아줌으로서 object boundary에 맞는 CAM을 생성토록 도와준 것으로 판단됨  
CAM을 직접 사용하는 것보다 GPP feature로 간접적 이용이 높은 성능을 보임  

Precision: True activation over the whole activation  
Recall: True activation over GT  


