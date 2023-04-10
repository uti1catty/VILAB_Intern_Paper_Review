# [CVPR22] DearKD: Data-Efficient Early Knowledge Distillation for Vision Transformers

No code yet

2 stage  
First: distills the inductive biases from the early intermediate layers of a CNN  
Second: Gives the transformer full play by training without distillation  

# 1. Introduction
CNN strong inductive biases by 2 constraints  
1. locality
2. wieght sharing mechanisms in the convolution operation  

DeiT 2 drawbacks  
1. Some works [11, 51]에 따르면 inserting convolutions to early stage of the network가 best performance를 가져옴.  
그러나 DeiT는 CNN의 classificaiton logit에서만 distill하여 'early (shallow)' transformer layer가 inductive bias를 capture하기 어려움.  
2. Distillation throughout the training implicitly hinders transformers from learning their own inductive bias and stronger representations.  

Distill from both classification logit and intermediate layers of the CNN.  
Intermediate: Explicit learning signals for intermediate transformer layers.  

Multi-Head Convolutional-Attention (MHCA)  

Aligner module: CNN features transformer tokens alignment  


