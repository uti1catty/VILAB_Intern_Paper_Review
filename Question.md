PascalVOC2012 train dataset을 10582개짜리를 사용하던데 이 image들은 pixel level GT가 있는 것인지? GT와 mIoU를 계산하는 것?  

## [CVPR18]RW
page11: output stride가 무엇을 의미?  
처음 input 대비 output의 크기 비율

## [CVPR19] IRN
page 4: 지금 P+내의 pixel들은 pseudo로 같은 class에 포함되는 쌍. 그런데 같은 class이지만 다른 instance가 겹쳐있는 경우 있을 수 잇음. nearby pixxel이라면 same instance로 가정한다 했는데 이경우 다른 instance로 어떻게 구분?  
구분 못함. 포기해야 하는 값인 것.  

## [CVPR22] C2AM
fg-fg, bg-bg positive fg-bg negative contrastive learning  
page 5: 지금 fg-fg에서 유사하지만 v값이 다르게 뽑혀 sim이 작을 때 이를 비슷하게 v를 뽑고자하는데 sim이 작으면 w에 의해 고려대상에서 빠짐. loss의 의미가 있는가?

## [CVPR22] ReCAM
BCE 대신 SCE 사용  
page2: 왜 multi label classification에서 class들이 dependent? 항상 같이나오는 object들이 함께 나오기 때문을 말하는가?  
결국 앞에서 함께 feature가 뒤섞이기 때문에 dependent한 것이 맞다.

## [CVPR22] CLIMS
CLIP으로 image-text contrastive learning  
  
## [CVPR22] L2G
local에서 classification하고 local의 activation을 global이 흡수하는 방식  

## [CVPR22] AFA
Transformer로 featureㅃ보아서 classification하고 CAM형성. CAM에서 pseudo affinity뽑고 MHSA에서 affinity 잘 뽑도록 학습.   
page 5: classification loss 단순한 (-) 오타가 맞는지?  
맞다.

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
이 design이 가장 성능이 높았다. fixed threshold등 모두 실험해봄.  

page 9:  
positive의 EA에서 이렇게 될 경우 모든 영역이 잘려나가더라도 model은 추가적인 공간을 찾도록 강요받아 over-expansion문제가 발생한다. 이를 막기위해 Negative를 사용  
먼저 I_A의 GPP결과인 P16_hat에서 t_L(threshold) 영역보다 작은 영역만 선택하여 channel별 평균으로 embedding e_AL을 형성한다. e_AL은 Original Image의 low confidence 영역만 추린것을 의미하게 된다.  
이 e_AL과 e_N의 distance를 maximize시킨다.  
over-expansion이 되면 low confidence영역이 image의 objects에 대하여 더 적은 information을 갖게 되어 e_AL이 더 적은 정보를 갖게 된다. 그러면 model은 e_AL과 e_N을 구분하기가 어려워진다. 따라서 model은 이 distance가 최대가 되도록 강요받음으로서 over-expension을 누르도록 유도된다.  
-> original image의 low confidence영역은 결국 배경 영역일 것이고 다른 class들에 대한 activation info가 거의 없을 것. 그러나 Negative image는 특정 class에 대한 확실한 activation이 있기 때문에 이미 distance가 클 것이고 model이 특정 class embedding을 멀리 가져다 두기만 하면 되기 때문에 그렇게 학습이 될 수 있다. 또한 low confidence 영역과 negative image를 비교하는 것은 결국 background와 다른 class를 구분짓도록 학습하여 background에서 해당 class를 activation하지 못하도록 하는 것이 아닌가 생각됨
결국 이게 왜 over expension을 막아주는지?  
맞는 말이지만 이렇게 돌아가지는 않음. 결국 정보가 부족해지기 때문에 둘 사이 구분이 어려워짐.  

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
포기하는 값인 것.  

page 8:  
theta동안은 Qc값으로 tc를 추정해서 사용하는데 이후 iter동안에는 tc를 생성한다는게 어떻게 생성하고 update한다는 뜻?  

## [ECCV22] ViT-PCM
CAM에 의존하지 않는 방식. CAM이 아니라 ViT의 output을 LSTM에 넣어 patch별 relation을 더한 후 MLP를 수행한 녀석으로 Baseline pseudo mask 생성.  

## [ICCV21] OC-CSE
AE방식을 사용. pretrained classifier는 less discriminative를 activate할 능력이 있으나 안할 뿐. 따라서 AE로 CAM에서 discriminative영역을 지우고 classifier에 넣어서 해당 class를 인식하지 못하도록 학습.  
즉 CGNet의 CAM이 모든 영역을 잘 cover하도록 수행. 

Q) classifier로 다른 class를 침범하지 않도록 강요한다고 함. (loss에서 해당 class의 value가 유지되어야 하기 때문)  
그런데 어떠한 class에도 포함되지 않는 background에 대한 guide는 어떻게?  
불가능하다. 실제 defense에서 단일 class의 경우 막지 못하는 것아닌가하는 질문이 들어옴.  PASCAL-VOC의 경우 50%가 multi class이고 COCO는 더 많음. multi class의 경우 image 크기가 작기 때문에 class끼리 겹친 경우가 매우 많아서 이 방법이 성능을 올림.  
