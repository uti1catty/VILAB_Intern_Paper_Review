# [CVPR22] Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation 
github: https://githb.com/zhaozhengVhen/ReCAM  

# 1. Introduction
WSSS를 위해 classification model에서 CAM을 추출  
마지막 multi class classification prediction을 수행할 때 2가지 방법  
1. sigmoid 후 loss (BCE) -> 각 class 별로 0~1로 만들고 GT와 비교  
class 사이에 class-exclusive 없음  
2. softmax 후 loss (SCE) -> 모든 class를 weight에 따라 상관관계로 합 1이 되도록 한 후 GT와 비교  
class 사이에 class-exclusive 있음

label이 1개일 경우 BCE SCE에서 val accuracy 차이는 없으나 CAM의 mIoU를 계산한 결과 SCE가 훨씬 뛰어난 성능을 보임  
different class 사이를 model이 confuse하게 하는 false positive pixel을 줄임  
class specific feature를 탐색하도록 유도하여 false negative pixel을 줄임  

그러나 multi label classification 문제에서는 different class의 probability가 independent하지 않다. 따라서 바로 SCE를 사용할 수 없음   
([47] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label
learning algorithms)

따라서 SCE를 additional loss to Reactivate the model로 사용 and generate ReCAM.  

# 3. Preliminaries
## 3.1 CAM
multi-label classifcation model with GAP followed by predition layer(FC layer). Prediction loss: BCE  
각 class마다 BCE를 구하고 더하여 loss 계산  
CAM은 weight과 feature를 곱하여 class k에 대한 CAM을 얻는다. (ReLU로 normalize 시행)  

## Semantic Segmentation 
Loss: class prediction z에 softmax -> GT y와 pixel, class 별로 CE계산, 합  

# 4 Class Re Activation Maps (ReCAM)
## 4.1 ReCAM Pipeline  
classifier에서 feature map f(x) (W X H X C) 획득  
FC layer-1과 sigmoid BCE로 multi label loss 계산  
FC layer-1의 weight으로 CAM 추출  
class별 CAM을 soft mask로 feature map과 elementwise multiplication -> K개의 masked feature map 생성 f1, f2, ...  
각 masked feature map을 FC layer-2에 넣어 각각 prediction 계산  
각 prediction을 softmax치고 CE계산  
K개 class에 대해 각각 sigle-label SCE를 계산하여 합산으로 L_sce 계산  
L_ReCAM = L_bce + lambda X L_sce  

reactivation이 끝난 후 feature map f(x) 에서 ReCAM 추출.  
추출 방식은 CAM 추출방식과 동일. 단 FC가 2개이므로 weight w 선택 옵션이 4 종류 존재  
w / w' / w + w' / w X w'  

# 5 Experiments
ReCAM은 큰 computational cost를 먹지 않는다.  
BCE base method에 SCE base인 ReCAM을 plug and play하여 성능을 올릴 수 있다.  
