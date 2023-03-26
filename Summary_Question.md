____
## 20230320

PascalVOC2012 train dataset을 10582개짜리를 사용하던데 이 image들은 pixel level GT가 있는 것인지? GT와 mIoU를 계산하는 것?  
A) GT는 1400개정도만 있음. 이것에 대해서만 mIoU계산. 나머지는 training에 사용.

## [CVPR18]RW
page11: output stride가 무엇을 의미?  
A) 처음 input 대비 output의 크기 비율

## [CVPR19] IRN
page 4: 지금 P+내의 pixel들은 pseudo로 같은 class에 포함되는 쌍. 그런데 같은 class이지만 다른 instance가 겹쳐있는 경우 있을 수 잇음. nearby pixxel이라면 same instance로 가정한다 했는데 이경우 다른 instance로 어떻게 구분?  
A) 구분 못함. 포기해야 하는 값인 것.  

## [CVPR22] C2AM
fg-fg, bg-bg positive fg-bg negative contrastive learning  
page 5: 지금 fg-fg에서 유사하지만 v값이 다르게 뽑혀 sim이 작을 때 이를 비슷하게 v를 뽑고자하는데 sim이 작으면 w에 의해 고려대상에서 빠짐. loss의 의미가 있는가?

## [CVPR22] ReCAM
BCE 대신 SCE 사용  
page2: 왜 multi label classification에서 class들이 dependent? 항상 같이나오는 object들이 함께 나오기 때문을 말하는가?  
A) 결국 앞에서 함께 feature가 뒤섞이기 때문에 dependent한 것이 맞다.

## [CVPR22] CLIMS
CLIP으로 image-text contrastive learning  
  
## [CVPR22] L2G
local에서 classification하고 local의 activation을 global이 흡수하는 방식  

## [CVPR22] AFA
Transformer로 featureㅃ보아서 classification하고 CAM형성. CAM에서 pseudo affinity뽑고 MHSA에서 affinity 잘 뽑도록 학습.   
page 5: classification loss 단순한 (-) 오타가 맞는지?  
A) 맞다.

## [CVPR22] RCA
Memory를 도입하여 Image set level의 global context 정보를 얻어서 classification에 더하고자 함.  
page 4: 각 class별 memory의 내용물 m_l의 값은 어떻게 생성하는가? update만 있고 생성에 대한 정보를 못찾음.  
Memory 내용물은 각각 D 차원의 1개 m만 있는 것이 맞음  

## [CVPR22] SIPE
각 image별 prototype(class 기준이 되는 feature)를 형성하여 CAM을 refine  

## [CVPR22] PPC
각 class별 prototype(CAM을 confidence로 top K개 선택)과 pixel사이의 contrastive learning(similarity softmax로 반영)  
서로 다른 view에서도 동일한 prototype, 동일한 CAM을 생성할 수 있도록 강요하여 semantic 학습  

## [CVPR22] W-OoD
relation이 강한 obejct의 구분을 추가적인 out of distribution data를 통해 학습('cluster-based metric larning objective')   

## [ECCV22] AEFT
GPP로 positive, negative를 모두 살리며 accurate CAM획득. AE와 contrastive learning으로 activation 영역을 적절 수준까지 넓힘.  

page 7: g에 attention이라는 이름을 붙인건 2개 요소의 activate부분을 가져오기 때문? 3x3 conv를 수행하는 이유?
어차피 각 2개의 layer를 따로 attention영역을 가져올 것인데 3x3 conv와 sigmoid 수행을 선택한 이유?  
A) 이 design이 가장 성능이 높았다. fixed threshold등 모두 실험해봄.  

page 9:  
positive의 EA에서 이렇게 될 경우 모든 영역이 잘려나가더라도 model은 추가적인 공간을 찾도록 강요받아 over-expansion문제가 발생한다. 이를 막기위해 Negative를 사용  
먼저 I_A의 GPP결과인 P16_hat에서 t_L(threshold) 영역보다 작은 영역만 선택하여 channel별 평균으로 embedding e_AL을 형성한다. e_AL은 Original Image의 low confidence 영역만 추린것을 의미하게 된다.  
이 e_AL과 e_N의 distance를 maximize시킨다.  
over-expansion이 되면 low confidence영역이 image의 objects에 대하여 더 적은 information을 갖게 되어 e_AL이 더 적은 정보를 갖게 된다. 그러면 model은 e_AL과 e_N을 구분하기가 어려워진다. 따라서 model은 이 distance가 최대가 되도록 강요받음으로서 over-expension을 누르도록 유도된다.  
-> original image의 low confidence영역은 결국 배경 영역일 것이고 다른 class들에 대한 activation info가 거의 없을 것. 그러나 Negative image는 특정 class에 대한 확실한 activation이 있기 때문에 이미 distance가 클 것이고 model이 특정 class embedding을 멀리 가져다 두기만 하면 되기 때문에 그렇게 학습이 될 수 있다. 또한 low confidence 영역과 negative image를 비교하는 것은 결국 background와 다른 class를 구분짓도록 학습하여 background에서 해당 class를 activation하지 못하도록 하는 것이 아닌가 생각됨
결국 이게 왜 over expension을 막아주는지?  
A) 맞는 말이지만 이렇게 돌아가지는 않음. 결국 정보가 부족해지기 때문에 둘 사이 구분이 어려워짐.  

## [ECCV22] Spatial-BCE
Loss를 각 pixel별로 계산. uncertainty를 줄이는 방향으로 fg와 bg를 반대방향으로 움직이도록 loss설정.  
0으로 수렴하지 않게 T/NT 비율 유지.  
adaptive threshold 설정.  

page5~6:  
즉 앞의 Loss 식에서 (1 - y^c), 없는 class에 대해서는 모든 pixel의 p가 작을 때 loss가 최소가 되고,  
y^c, 있는 class에 대해서는 각 pixel의 uncertatinty가 작아지도록, 즉 target t^c에서 멀어져 더 확실히 fg/bg로 나누어지도록 강요받음.  
=> Q) 그렇다면 non-target candidate의 p값은 어느 방향으로 이동하는가? 값이 더 작아지는 방향으로 이동하는가?  
=> A) Figure 3에서 Gradient를 따라 이동하게 된다면 그렇게 이동할 것.  
=> Q) 그렇다면 해당 class이지만 p값이 작게 유도되어 non-target candidate로 분류된 pixel은 어떻게 다시 올릴 것인가?  
A) 포기하는 값인 것.  

page 8:  
theta동안은 Qc값으로 tc를 추정해서 사용하는데 이후 iter동안에는 tc를 생성한다는게 어떻게 생성하고 update한다는 뜻?  

## [ECCV22] ViT-PCM
CAM에 의존하지 않는 방식. CAM이 아니라 ViT의 output을 LSTM에 넣어 patch별 relation을 더한 후 MLP를 수행한 녀석으로 Baseline pseudo mask 생성.  

## [ICCV21] OC-CSE
AE방식을 사용. pretrained classifier는 less discriminative를 activate할 능력이 있으나 안할 뿐. 따라서 AE로 CAM에서 discriminative영역을 지우고 classifier에 넣어서 해당 class를 인식하지 못하도록 학습.  
즉 CGNet의 CAM이 모든 영역을 잘 cover하도록 수행. 

Q) classifier로 다른 class를 침범하지 않도록 강요한다고 함. (loss에서 해당 class의 value가 유지되어야 하기 때문)  
그런데 어떠한 class에도 포함되지 않는 background에 대한 guide는 어떻게?  
A) 불가능하다. 실제 defense에서 단일 class의 경우 막지 못하는 것아닌가하는 질문이 들어옴.  PASCAL-VOC의 경우 50%가 multi class이고 COCO는 더 많음. multi class의 경우 image 크기가 작기 때문에 class끼리 겹친 경우가 매우 많아서 이 방법이 성능을 올림.  

______
## 20230327
## [CVPR21] AdvCAM  
Adversarial attack의 반대방향으로 adversarial climbing으로 non-discriminative 영역을 iterative하게 찾아나가며 쌓아나감. 

page 3:  
adversarial attack에서 NN을 x에 대해 미분한 것이 왜 법선방향?  

## [CVPR21] BANA
BB외부의 영역에서 bg feature를 얻고 이와 BB 내부를 비교하여 fg-bg구분. fg feature와 bg feature로 cls loss를 계산  
cls loss 계산할 때 classifier의 weight으로 CAM을 얻고 refine하여 pseudo 획득  

## [CVPR21] BBAM
object detector가 image에서 object detection 영역을 예측하는 것과 거의 유사하게 예측할 수 있는 최소한의 pixel 영역을 구하도록 학습  
해당 영역을 얻고 -> CRF로 refine -> MCG proposal로 refine 하여 pseudo GT 생성  

page 3:  
M* 식에서 argmin 식의 역할? M의 크기를 제한하는 regularization 같은데 argmin으로 어떻게 제한하는지 이해가 안감.  

## [CVPR21] EDAM
각 image마다 class-specific mask를 explicit하게 구하도록 설계. 각 Mask를 Feature에 곱하여 해당 class에 대한 feature를 구하고 B개 의 image의 feature에 대해서 self-attention 수행. GAP로 classification loss 계산. 즉 explicit Mask를 잘 구하도록 하는 것이 핵심.  

page 4:  
self attention은 input과 output의 size가 같음. input으로 1x1 conv를 마치고 dim이 1 x (B x H x W) x d 인 $\hat{\mathcal{F}}^k$ 가 들어감. 그런데 식 4에서 output의 dimension이 1x1 conv를 수행하기 전의 dim인 BxCxHxW 를 가짐. 1x1 conv로 다시 d dim을 C dim으로 바꾼 것인가?  

## [CVPR21] nsrom
image 구석의 disjoint and distant regions 에서 object를 찾기 위해 graph conv 도입.  
OA-CAM과 CAM을 추출. OA-CAM과 saliency map에서 background를 추출하고, CAM에서 activate가 되어서 object일 가능성이 있는 영역을 bg에서 제외시킴 (POM).  
이 Pseudo label로 segmentation network를 학습시켜서 initaial prediction획득. inital과 pseudo label로 non-salient region masking을 수행해서 object와 바로 근처 bg를 제외하고 전부 날림. 이걸로 다시 segmentation학습. 스스로 object를 찾도록 (정확하지 않은 bg를 주지 않아서) 학습.  

page 3:  
segemtantion network를 2회 학습하는데 이때 학습한 weight을 초기화하고 다시 학습? 아니면 그대로 두고 추가 학습?  
wrong bg를 제거하고 학습하는 것의 효과를 보려면 초기화하고 처음부터 학습하는게 맞다고 생각됨.  

## [CVPR21] EPS
CAM을 적절히 더하여 estimated saliency map을 생성하고 saliency map과 loss 계산. 이를 통해 saliency map에서 boundary에 대한 정보와 fg-bg구분에 대한 정보를 학습. classiciation loss도 함께 수행.  

page 2:  
saliency loss가 fg를 bg로부터 구분할 수 있게 돕고 co-occurring pixel을 bg로 분류할 수 있게 한다고 함.   
Saliency map이 co-occurring pixel을 전혀 fg로 구분하지 않는다고 말할 수 있기 때문인가?   

## [CVPR21] ReviewKD
Student의 k layer가 teacher의 1~k layer로부터 knowledge distillation  
해당 계산을 간단하게 하기 위하여 Residual, Attention based fusion(High level + low level방법), Hierarchical context loss (Spatial pyramid pooling으로 loss계산) 도입.  

## [CVPR22] MCTformer
class token을 class 개수만큼 도입.  
V1:  
output class-token의 average pooling으로 cls loss계산 -> 해당 class token과 class를 묶는 연결점  
class-patch attention에서 attention map 추출  
patch-patch attention을 affinity map으로 사용하여 attention map refine  

V2:  
output class-token의 average pooling으로 cls loss계산 -> 해당 class token과 class를 묶는 연결점  
output patch-token을 reshape하여 conv-GAP로 cls loss 계산  
output patch token feature map에서 PatchCAM 추출  
V1의 attention map 추출 방법대로 MCT Attention 추출  
MCT Attention과 PatchCAM을 pixelwise mul로 fusion  
patch-patch attention의 affinity로 refine  

