# [CVPR22] Multi-class Token Transformer for Weakly Supervisedd Semantic Segmentation

github: https://github.com/xulianuwa/MCTformer

tansformer-based framework to learn class-specific oject localization maps as pseudo labels for WSSS.  
stansdard Vision transformer의 one-class token의 attended regions가 class-agnostic localization map을 형성하는데 역할을 한다는 것에서 Inspired.  
Multiple class token으로 class-specific attention을 효율적으로 capture하도록 함.  

MCTformer: Multi-class Token Transformer: multiple class tokens로 class tokens와 patch tokens 사이의 interaction을 학습.  

class-to-pathc attention corresponding to different class token에서 class-discriminative object localization map 생성.  

patch-to-patch transformer attention에서 patch-level pairwise affinity 추출 -> refine the localization map  

# 1. Introduction
ViT: 1 extra class token to aggregate information from teh entire sequence of the patch token.  

DINO[3]: semantic scene layout can be discovered from the attention maps of the class token.  
transformer atteiont의 different heads가 different semantic regions of image를 attend할 수 있다고 증명되었다. 그러나 어떻게 head를 correct semantic class로 associate(연결)할지가 unclear함.  
즉 these attention maps are still class-agnostic.  

using 1 class token makes the accurate localization of different objects on a signle challenge.  

2 main reasons.  
1. 1 class token design essentially inevitably captures context information from other object categories and the backgorund.  
-> class-specific and generic representations form different object classes를 1개의 class token으로 고려하여 학습하기 때문에 non-discriminative하고 noisy하다.  
2. Model uses the only one-class token to learn interactions with patch tokens for a number of distinct object classes in a dataset.  
-> model apacity가 targeted discriminative localization performance를 달성하기에 충분히 알맞지 않다.  

Tackle this Issue.  
multiple class token which will be responsible for learning representation for different object classes.  
MCTformer: multiple class-specific tokens가 있어 class-specific transformer attention을 이용.  

그러나 단순히 class개수를 ViT에서 증가시키는 것은 이 class token들이 여전히 specific meaning을 갖지 않기 때문에 맞지 않다.  
각 token이 specific object class의 high-level discriminative representation을 학습하게 하기 위해 "class-aware training strategy for multiple class token"을 도입.  
-> output class tokens from the transformer encoder에 embedding dimension 방향으로 Average pooling을 적용하여 class score를 생성 -> GT class label에 의해 directly supervised.  
-> 각 class token과 class label 사이의 strong connection 생성.  
-> class-to-patch attention of different classes를 class-specific localization maps로 directly 사용 가능.  

patch-to-patch attention을 training의 byproduct(부산물)로서 추가적인 computation엇이 patch-level pairwise affinity로 사용 가능.  

또한 when applied on patch tokens, transformer framework가 CAM method 를 fully complements한다. (class-token과 patch-token based representation을 동시에 classify)  

# 3. Multi-class Token Transformer
## 3.1 Overview
Input RGB image split to non-ovelapping patches, transformed into a sequence of patch tokens.  ($N \times N$ patches -> $N^2$ tokens)  
Multiple class token이 patch token과 concat되고 embedding position information (PE)와 더해져 input token of transformer encoder 생성.  

Transformer block $L$개 통과  

output class token from the last layer에 Average Pooling 적용 -> class score 생성. (MLP 대신)  
이 class score와 GT class label 사이 classification loss.  
-> each class token - classs label 사이 strong connection  

test time에서 class-to-patch attention in the transformer에서 class-specific localization 추출.  

각 transformer layer에서 학습한 서로 다른 정보를 이용하기 위해 aggregate the attention maps from multiple layers.  

patch-to-patch attention에서 aptch-level pairwise affinity 추출하여 refine class-to-patch attention.  

## 3.2 Class-Specific Transformer Attention Learning
### 3.2.1 Multi-class token structure design.  
input image -> split into $N \times N$ patches  
$T_p \in \mathbb{R}^{M \times D}$ :transformed sequence of patch tokens   
$D$: embedding dimension  
$M = N^2$   
$T_{cls} \in \mathbb{R}^{C \times D}$: $C$ class tokens  
$C$: number of classes  
$Concat(T_p, T_{cls}) + PE$ concat($C$ class tokens, patch tokens) + position embeddings.  
$T_{in} \in \mathbb{R}^{(C+M)\times D}$: input tokens of transformer encoder.  
Transformer encoder: $L$ consecutive encoding layer  
each layer: Multi-Head Attention (MHA) module + MLP + 2 LayerNorm layers (before each MHA and MLP).   
### 3.2.2 Class-specific multi-class token attention  
Use standard self-attention layer to capture long-range dependencies btw tokens.  
First, normalize input token sequence and transform it to triplet of $Q,K,V$ through linear layer.  
$Q \in \mathbb{R}^{(C+M) \times D}$, $K \in \mathbb{R}^{(C+M) \times D}$, $V \in \mathbb{R}^{(C+M) \times D}$  
Scaled Dot-Product Attention mechanism -> queries, keys attention value.  
Output token: weighted sum of all tokens using the attention values at weights.  
$$Attention (Q, K, V) = softmax(QK^T/\sqrt{D})V\ \ (1)$$
softmax: row 별로 계산. row token에 대하여 col tokens들의 유사도 값 계산.  
즉 output token의 row token은 attention matrix의 row값을 weight으로 모든 $V$의 token들을 weighted sum 한 결과.  
attention matrix의 row값은 row token에 대하여 col tokens들의 유사도 값 계산한 결과이므로 output의 row는 $Q$의 row token과 $K$의 모든 token들의 유사도 값을 weight으로 $V$의 모든 token을 weighted sum한 것.  

$A_{t2t} = softmax(QK^T/\sqrt{D}) \in \mathbb{R}^{(C+M)\times (C+M)}$  : token-to-token attention map  
$A_{c2p} = A_{t2t}[1:C,C+1:C+M] \in \mathbb{R}^{C \times M}$: class attentions to patches; class-to-patch attention  
Each row represents attention scores of a specific class to all patches.  

original spatial positions of all patches와 attention vector로 $C$ class-relevant localization map 생성   
각 transformer encoding layer마다 class-relevant localization map 추출.  
Higher layers가 더 high-level discriminative representation 학습.  
Earlier layers는 general and low-level visual information capture.  

Fuse class-to-patch attentions from the last $K$ transformer encoding layers. -> Explore a good trade-off btw precision and recall on generated object localizaiton maps.  
$$ \hat{A}_{mct} = {1 \over K} \sum_l^K\hat{A}_{mct}^l\ \ (2)$$
$\hat{A}_{mct}^l$: class-specific transformer attention extracted from $l^{th}$ transformer encoding layer of MCTformer-V1.   
Fused map $\hat{A}_{mct}$ normalized by min-max normalization along 2 spatial dimensions.  
$A_{mct} \in \mathbb{R}^{C \times N \times N}$: Generated final class-specific object localization maps
$$A_{mct} = min\_max\_Norm(\hat{A}_{mct})$$

### 3.2.3 Class-specific attention refinement
$A_{p2p} = A_{t2t}[C+1:M,C+1:C+M] \in \mathbb{R}^{M \times M}$: patch-to-patch attention  
4D tensor로 reshape. $\hat{A}_{p2p} \in \mathbb{R}^{N \times N \times N \times N}$  
이 affinity로 class-specific transformer attention refinement.  
$$ A_{mct\_ref}(c, i, j) = \sum_k^N \sum_l^N \hat{A}_{p2p}(i, j, k, l) \cdot A_{mct}(c,k,l)\ \ (3)$$  
$A_{mct\_ref} \in \mathbb{R}^{C \times N \times N}$: refined class-specific localization map  
***
## 식 (3)에 대한 설명
$(i,j)$를 고정하고 생각하자.  
$\hat{A}_{p2p}(i, j, k, l)$는 특정 patch $(i,j)$에 대해서 다른 모든 patch와의 유사도(affinity)를 의미함. softmax로 인하여 모든 값의 합은 1    
affinity: $\mathbb{R}^{N \times N}$  
이 값을 $A_{mct}(c,k,l)$과 곱함. 즉 모든 patch의 class-specific attention에 affinity를 곱함.  
이후 $\sum_k^N \sum_l^N$으로 모든 patch의 값을 더함.  
즉 patch $(i,j)$에 대해서 다른 모든 patch와의 affinity를 weight으로 class-specific attention을 weighted sum하는 것.  
이를 통해 특정 class $c$에서 patch $(i,j)$가 다른 활성화된 patch $(k,l)$과 유사하다면 이 patch도 weighted sum에 의해 값이 커져 활성화되게 됨. -> False negative filtering  
만약 patch $(i,j)$가 class $c$에서 활성화 되어 있지만 활성화가 되지 않은 patch $(k,l)$과 유사도가 높다면 이 patch도 weighted sum에 의해 값이 작아져 비활성화 됨. -> False Positive filtering  
***

### 3.2.4 Class-aware training
우리는 multiple class token $T_{cls} \in \mathbb{R}^{C \times D}$를 사용하고 different class token이 different class-discriminative information을 학습하기를 원함.  
Output class token에 Average pooling을 적용하여 class score 생성  
$$ y(c) = {1 \over D}\sum_j^DT_{cls}(c, j)\ \ (4)$$
$y \in \mathbb{R}^C$: prediction   
$c \in 1,2,\cdots, C$  
$T_{cls}(c,j)$: $j^{th}$ feature of $c^{th}$ class token  
$GT$ - $y$ 사이 multi label soft margin loss  
이를 통해 each class token이 class-specific information을 cpature할 수 있도록 strong and direct class-aware supervision 제공  
___
### 참고: multi labels soft margin loss
$$L_{cls} = - {1 \over C}\sum_{c=1}^Cy_clog\sigma(p_c) + (1-y_c)log[1 - \sigma(p_c)]$$
$p_c$: prediction of the network for c-th class  
$\sigma()$: sigmoid  
$C$: foreground class number  
$y_c$: image level label for c-th class. 
___

## 3.3 Complementarity to Patch-Token CAM
MCTformer-V2: CAM module과 MCTformer-V1 통합  
$T_{out} \in \mathbb{R}^{(C+M) \times D}$: output tokens from transformer encoder  
$T_{out\_cls} \in \mathbb{R}^{C \times D}$: output class token  
$T_{out\_pat} \in \mathbb{R}^{M \times D}$: output patch token  
$T_{out\_pat}$를 $\mathbb{R}^{N \times N \times D}$로 reshape  
Conv layer with $C$ output channel 통과  
$F_{out\_pat} \in \mathbb{R}^{N \times N \times C}$: produced 2D feature map  
Global Average Pooling (GAP) layer로 class prediction으로 transform  
MCTformer-V1처럼 class token에서도 class score 계산  
Total loss: sum of 2 multi-label soft margin losses  
$$L_{total} = L_{cls-class} + L_{cls-patch}\ \ (5)$$

### 3.3.1 Combining PatchCAM and class-specific transfromer attention
Test time  
patch token-based CAM(PatchCAM)을 last conv layer에서 추출  
$A_{pCAM} \in \mathbb{R}^{N \times N \times C}$: PatchCAM  
$$A_{pCAM} = min\_max\_Norm(F_{out\_pat})$$  

PatchCAM과 class-specific transformer attention map을 combine하여 fused object localization map $A$ 생성  
$$ A = A_{pCAM} \circ A_{mct}\ \ (6)$$
$\circ$: element-wise multiplication (Hadamard product) 

$A_{pCAM}$과 $A_{mct}$모두 min-max Norm으로 0~1값을 가짐.  
따라서 element-wise multiplication을 수행하면  
1. 두 방법이 모두 active로 동의하는 것이 active로 남음.  
2. 한 방법만 deactive라고 판단하는 값은 값이 작아짐. 
3. 두 방법이 모두 deactive라고 판단한 값은 매우 작아짐.  

### 3.3.2 Class-specific object localization map refinement  
MCTformer-V1에서 한 것 처럼 (Eq.3) MCTformer-V2의 patch-to-patch attention map을 추출해 patch-level pairwise affinity로 사용.  
$$ A_{ref}(c, i, j) = \sum_k^N \sum_l^N \hat{A}_{p2p}(i, j, k, l) \cdot A(c,k,l)\ \ (7)$$ 

class token과 patch token에서의 class prediction의 classification loss를 통해 strong consistency btw 2 types of tokens가 improve the model learning을 enfoce할 수 있다.  
### Intuition
1. This consistency constraint can be regarded as an auxiliary supervision to guide the learning of more effective patch representations.
2. the strong pairwise interaction (i.e. message passing) between the patch tokens and the multiple class tokens can also lead to more representative patch tokens, thus producing more class-discriminative PatchCAM maps, compared to only using one class token as in TS-CAM [14].

# 4. Experiments
## 4.1 Experimental Setting
### Dataset
PASCAL VOC 2012  
MS COCO  

### Evaluation metric
mIoU  
semantic segmentaiton on val set  
### Implementation details
MCTformer를 DeiT-S Backbone pretrained on ImageNet으로 구현  
Initialize the proposed multiple class token을 위해 Pre-trained class token in DeiT-S 사용.  
Data augmentation과 default training parameter는 [14,36]을 따름  
Training image는 $256 \times 256$으로 reshape된 후 $224 \times 224$로 crop.  
Semantic Segmentation을 위해 ResNet38 based Deeplab V1 사용.  
Test time에 multi-scale testing과 CRFs를 with hyper-parameters suggested in [6]로 post-processing을 위해 사용.  

## 4.2 Comparison with SOTA
### PASCAL VOC
[4,22,32,39,49]를 따라 PSA[1]를 object localization maps(seed)에 적용하여 pseudo semantic segementation GT labels (Mask) 얻음.  

## 4.3 Ablation Studies
### 4.3.2 Complementarity of PatchCAM and the proposed class-specific transformer attention 
Figure 5e: class-specific transformer attention can effectively localize objects while with "low responses and noises".  
Figure 5f: PatchCAM maps show "high responses on object regions, while they also have more BG pixels around the objects activated.  
Fusion -> clearly improved localizaiton maps with reduced BG noise.  
