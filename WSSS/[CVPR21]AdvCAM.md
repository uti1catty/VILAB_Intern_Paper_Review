# [CVPR21] Anti-Adversarially Manipulated Attributions for Weakly and Semi-Supervised Semantic Segmentation

github: https://github.com/jbeomlee93/AdvCAM

AdvCAM은 classification score를 높이도록 manipulate된 image의 attribution map.  
manipulation is realized in an anti-adversarial manner, images along pixel gradients in the opposite direction from those used in an adversarial attack.  

pixel의 값을 계속 update해주어 iterative하게 non-discriminative한 영역을 activate하도록 확장해나가는 방식.  

# 1. Introduction
만약 discriminative한 영역을 지운 image가 dicision boundary를 넘어서 다른 class로 인식되면 erroneous attribution map이 생성된다.  

adversarial attack은 small perturbation of an image that pushes it across the dicision boundary를 찾는 방법이다.  
우리의 방법은 이 adversarial attack의 반대 방식으로, perturbation that pushes the manipulated image away from the decision boundary를 찾는다.  
이 manipulation은 target class의 classification score를 높이는 pixel gradient를 따라 image가 교란되는 adversarial climbing에 의해 실현된다.   
점차 non discriminative region이지만 해당 class에 포함되는 영역을 포함해 CAM이 더 많은 영역을 identify하도록 한다.   

ascending he gradient는 classification score를 상승하는 것을 보장하지만,
repetitive ascending은 bg나 다른 object의 region같은 관련없는 area를 함께 activate하거나, 특정 영역의 attribution score를 매우 높이도록 할 수 있다.  
이 문제를 해결하기 위해 regularization term을 포함하였다. 이 regularization은 다른 class의 score를 supress하고 이미 높은 score를 받은 attribution score를 제한한다.  

# 3. Propsed Method
## 3.1 Preliminaries (Adversarial Attack)
AA는 small pixel level perturbation that cna change the output from DNN을 찾는 것을 목표로 한다.  
NN(x) != NN(x + n)을 만족하는 n을 찾는다.  
n을 구성하는 대표적인 방법은 NN의 decision boundary에 대한 법선 벡터를 고려하는 것. x에 대한 NN의 gradient를 찾음으로서 실현될 수 있다.   
$$x' = x - \zeta \nabla_xNN(x)$$
$\zeta$는 extent of the chage to the image이다.  

## 3.2 AdvCAM
## 3.2.1 Adversarial Climbing
anti-adversarial technique that manipulates the image so as to increase the classifciation score of that image.  
iterative adversarial step t
$$x^t = x^{t-1} + \zeta\nabla_{x^{t-1}}y_c^{t-1}$$

Localization map $\Alpha$ 생성. 각 iteration t에서 manipulated 된 image의 CAM들의 병합으로 iteration의 결과물을 encasulte한 결과.  
$$ \Alpha = {\Sigma_{t=0}^TCAM(x^t) \over max\Sigma_{t=0}^TCAM(x^t)}$$
전체 pixel중의 max값으로 나누어 주어서 최대값이 1이되도록 설정.  

### 3.2.2 How can Adversarial Climbing Improve CAMs?
2가지 질문에 응답.  
1. non-discriminative features가 enhance될수 있는가?
2. enhaced feature가 사람의 관점에서 class-relevant한가?

A1. discriminative region $R_D = \{i|CAM(x^0)_i \ge 0.5\}$와 Non discriminative region $R_ND = \{i|CAM(x^0)_i \lt 0.5\}$ 을 정의.  
pixel amplification $s_t^i = CAM(x^t)_i / CAM(x^0)_t $ 를 정의할 때 step t에 따라 $R_D, R_ND$ 모두 증가한다. 이때 $R_ND$의 $s_t^i$가 더 크게 증가한다.  
즉 non-discriminative region이 더 많ㅇ amplify된다.   

A2. Input에 대하여 sharply curved loss landscape가 NN을 AA로부터 취약하게 만든다고 한다. 
이후 연구자들이 loss landscape의 curvature를 줄이거나 loss가 behave linearly하도록 하면 NN의 robustness를 향상시킬수 있음을 보였다.  
robust in this sense한 System이 인간의 인식과 더 잘 일치한다고 나타났다.  
Adversarial Climbing의 영향으로 발생한 loss surface의 carvature는 매우 작다.  
따라서 우리는 인간의 관점에서 class와 관련된 속성을 증가시킨다고 기대할 수 있다.  

## 3.3 Regularization
loss surface obtained by adversarila climbing is resonably flat하지만 그럼에도 too much repetitive는 다른 object의 region을 activate하거나 이미 high score인 region을 increase할 수 있다.   
이미 high인 것을 increase하면 normalized score가 낮아지며, 새로운 region을 추가로 activate하는것을 막는다.  
이를 
1. supressing the logit values associated with other classes, 
2. restricting hihg attributions on discriminative regions of the target object  

로 해결한다.  

2번 문제는 Mask를 통해 해결한다.
$$M = 1(CAM(t^{t-1}) > \tau) (1(): indicator function)$$

regularization을 apply한 식은 다음과 같다.
$$ x^t = x^{t-1} + \zeta\nabla_{x^{t-1}}L$$
$$L = y_c^{t-1} - \Sigma_{k\in C\backslash c}y_k^{t-1} - \lambda||M\odot |CAM(x^{t-1} - CAM(x^0))|||_1 (\odot: element-wise\_ mul)$$
뒤의 Mask를 사용하는 term에서 $x^{t-1}$의 CAM에서 $x^{0}$의 CAM을 빼서 discriminative 역만 추출한 것. 이 값들이 커지면 Loss가 커지는 것. 즉 해당 값들이 $x^{0}$의 CAM값을 유지하도록 강요.   

Regularization은 보통 Loss에 더해주게 되는데 지금 L값은 Loss가 아니라 predict value. Loss계산은 GT에서 predict를 빼주면서 계산하게 되니 부호를 반대로 하여 predict값에서 regularization term을 빼주어 같은 효과를 가져오도록 함.  


