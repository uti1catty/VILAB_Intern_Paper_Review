# [ICCV21] Unlocking the Potential of Ordinary Classifier: Class-specific Adversarial Erasing Framework for Weakly Supervised Semantic Segmentation

github: https://github.com/KAIST-vilab/OC-CSE


CAM이 오직 가장 discriminative region에 집중함  
adversarial erasing으로 less discriminative region을 탐색 가능토록 함  
Ordinary classifier가 이미 less discriminative region을 활성화 할 수 있는 능력이 있음을 discriminative region을 지움으로서 실험적으로 확인  
1. adopts the ordinary classifier to notify the regions to be erased 
2. generates a class-specific mask for erasing
by randomly sampling a single specific class to be erased
(target class) among the existing classes on the image for
obtaining more precise CAMs.  

ordinary classifier의 가이드를 통해 cam generation network (CGNet)이 다른 class를 침범하지 않도록 강요받으며 CAM을 생성  

1. Randomly sample a single class to be erased (target class)
2. CGNet에서 생성한 CAM에서 target class 영역 선택 후 image에 masking
3. masking한 image를 ordinary classifer에 넣어서 각 class의 prediction score 계산
4. classifier의 결과물에서 taget class의 score가 낮아지도록 CGNet 학습 -> CGNet은 최대한 정확한 영역을 잡아내도록 CAM을 생성하도록 학습됨  

# 4. Proposed Method
## 4.1 CAMs Generation
Global Average Pooling, fully connected layer 로 CAM을 생성하지 않음  
1X1 conv layer which has #of classes (nc) output channels followed by GAP  
1X1 conv를 진행한 후 channel이 ck가 된다. 각 channel은 class ck의 CAM이 된다.  
CAMs에 GAP를 하게 되면 ck길이의 1D vector가 나오는데 이것이 prediction result가 되어 class c일 확률을 의미한다.  (IPAD 그림)  

CAM A^ck를 erasing을 위한 back propagable mask로 쓰기 위해 ReLU를 시행하고 그 값의 최대 값으로 나누어 0~1로 normalize 시킨다.  

## 4.2 Propsed Framework
Image Ii가 CGNet으로 들어가 CAMs Ai와 prediction pi를 생성  
random으로 target class ck 선택하여 Image Ii Masking  
fixed ordinary classifier에 masked image Ii_hat이 들어가 prediction pi_hat 생성  
CGNet으로부터 image에서 target class가 더이상 없도록 masking하고 remain class는 여전히 남아있도록 하길 바람  
2개의 classification loss를 합하여 최종 loss로 함. (binary cross entropy loss 사용)  

## 4.4 CAM Refinement
more accurate pixel-level label 생성을 위해 AffinityNet 논문의 방법대로 따라가 CAM Refinement를 진행하였다.  
AffinityNet을 학습하기 위한 fg / bg는 CRF를 our refined CAMs에 적용하여 얻었다.  

# 5. Experiments
## 5.2 Implementation details
CGNet과 ordinary classifier의 backbone으로 ResNet38 사용  
둘 모두 ImageNet weights으로 initialized  
ordinary classifer가 PASAL VOC 2012 train dataset으로 standard classification loss로 pretrained  
MS-COCO로 실험할 때는 ordinary classifier를 MS-COCO로 같은 방식으로 pretrain  

Data augmantation: random resizing / horizontal flipping / color jittering / random corpping  


