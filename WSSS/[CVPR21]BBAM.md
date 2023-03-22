# [CVPR21] BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instnace Segmentation 

github: https://github.com/jbeomlee93/BBAM

# 1. Introduction
pixel level method of lcalizing a taret object inside its BB using a trained object detector.  
object detection을 수행할 때 detector가 focus하는 image region을 highlight하는 attribution map을 사용.  
BBAM: Bounding Box Attribution Map: object detector가 거의 같은 결과물을 내도록 하는 최소한의 영역을 indication한다.  
이 localization을 semantic/instance segmentation learning에 pseudo GT로 사용할 수 있다.  

# 3. Method
## 3.1 Revisiting Object Detectors
보통 1 stage, 2 stage 접근법이 있는데 region propsal-box refinement의 2 stage 방식에 집중한다. (Fast R-CNN 등)  
Region Proposal Network(RPN)는 BB형태로 candidate object proposla을 생성하는데 proposa이 class-agnositc이고 noisy하며 redundant(장황한)하다. 따라서 refinement step이 필요하다. classification, BB regression이 수행된다.  
RPN에서 나온 proposal boxes는 different size를 갖지만 RoI pooling이 feature map corresponding to each proposal을 predefined fixed size로 변경한다.  
pooled feature map이 classification head와 bounding box regression head를 통과한다.  

### Classifciation head
각 proposal의 class c에 대한 probability $p^c$를 계산하고 most likely class $c* = \argmax_c p^c$를 assign한다.  
### Bounding box regression head
noisy proposal을 object에 fit한다.   
offsets $t^c = (t_x^c, t_y^c, t_w^c, t_h^c)\ for\ each\ class\ c \in \{1,2,...,C\}$ 을 계산하여 fit에 이용한다.  
final localization은 각 coordinate를 offset $t^{c*}$를 이용해 shifting한다.  
## 3.2 Bounding Box Attribution Map (BBAM)
image $I$와 corresponding bounding box annotation은 주어짐.  
RPN으로부터 나왔거나 주어진 set of object proposals $O=\{o_k\}_{k=1}^K$ (K: number of proposal)도 있다.  
각 proposal $o_k$에 대해 box head $f^{box}$, cls head $f^{cls}$가 각각 box offset $t_k = f^{box}(I, o_k)$, cls probability $p_k=f^{cls}(I, o_k)$를 계산함.  

BBAM은 detector가 object detection을 위해 필요한 중요한 image의 영역을 identify한다.  
smallest mask $M: \Omega \to [0,1]$  
$\Omega$: original image와 prediction이 ㄱ거의 같아지는 subset of the image를 capture하는 set of pixels.    

Mask는 subset of image를 perturbation function으로 다음과 같이 specify한다.  
$$ \Phi(I, M) = I \circ M + \mu \circ (1 - M)$$
$\circ$: pixel wise multiplication, $\mu$: per-channel mean of the training data with the same size as $M$  
$\Phi$는 image에서 핵심부위만 그대로 두고 나머지는 per channel mean으로 채운 값.  
각 proposal $o$에 대하여 best mask $M*$는 다음 함수를 최적화하도록 gradient descent w.r.t $M$으로 계산된다.  
$$M* = argmin_{M \in [0,1]^\Omega} \lambda ||M||_1 + L_{perturb}$$
$$ L_{perturb} = 1_{box}||t^c - f^{box}(\Phi(I, M), o)||_1 + 1_{cls}||p^c - f^{cls}(\Phi(I, M), o)||_1$$  
1은 logical variables that have a value of 0 or 1 to control which head is used to produce localizations.  
즉 $L_{perturb}$는 핵심부위만 있는 image를 classifier가 예측한 값과 실제 값의 차이를 모두 더한 값.  
Q=> 위 식의 앞의 값은 regularization term인듯?  

original image와 같은 크기의 Mask를 사용하면 adversarial effect에 취약: 작은 변화에도 prediction이 크게 변함.  
이를 mask downsampled by stride $s$를 통해 여러 pixel이 $M$의 하나의 element에 의해 움직이도록 하여 해결.  
$M \in R^{\ulcorner w/s \urcorner \times \ulcorner h/s \urcorner}$ for image $I \in R^{w \times h}$ perturbation function $\Phi(I, M) = I \circ \hat{M} + \mu \circ (1 - \hat{M})$, $\hat{M} \in R^{w \times h}\ (Upsampled\ M)$  

fixed size of perturbation unit은 RoI-pooled features에서 perturbation의 different size를 야기.  
따라서 apdative stride $s(a)$ ($a$는 image와 object detector가 예측한 BB의 영역의 비율) 사용.    
small object에 small stride, large object에 large stride 사용.  

## 3.3 Generating Pseudo Ground Truth
먼저 각 GT box에 대하여 randomly jittering each coordinate of the box by up to $\pm 30\%$ 로 set of object proposals $O$를 생성한다.  
proposal이 $f^{cls}, f^{box}$로 보내져서 $f^{cls}$가 class를 올바르게 판단하고 $f^{box}$가 예측한 box의 IoU가 80%이상이면 해당 proposal을 positive proposal $O^{+} \in O$ 로 넣는다.  

$$L_{perturb} =  E_{o \in O^{+}} [1_{box}||t^c - f^{box}(\Phi(I, M), o)||_1 + 1_{cls}||p^c - f^{cls}(\Phi(I, M), o)||_1]$$ 
1함수는 모두 1로 설정.  

BBAM에 모든 pixel이 포함되지 않기 때문에 이를 CRF를 이용하여 refine한다.   
BBAM의 pixel을 threshold $\theta$ 이상 값을 추려 pseudo instance level GT mask를 생성한다.  
이 mask를 $\Tau$라고 하면 threshold $\theta$ 가 $\Tau$의 크기를 결정한다. 각 BBAM마다 fg에 해당하는 pixel의 비율은 다르므로 fixed threshold를 사용할 수 없다.  
2개의 threshold $\theta_{fg}, \theta_{bg}$를 고려하고 해당 사이의 값은 training segmentation network단계에서 무시한다.  

### Refine with MCG proposals  
MCG는 unsupervised mask proposal generator   
먼저 mask $\Tau$와 가장 IoU가 높은 mask proposal을 선택한다.  
해당 proposal이 target의 일부만을 포함할 수 있으므로 mask $\Tau$내에 완전히 포함되는 proposal을 모두 선택한다.  
refined mask $\Tau_r$  
$$ \Tau_r = \bigcup_{i \in S}m_i\ \ where\ S = \{i|m_i \subset \Tau \} \bigcup \{\argmax_i IoU(m_i, \Tau)\}$$



