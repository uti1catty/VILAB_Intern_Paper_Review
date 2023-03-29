## OC CSE
### 1. 
model_cse.py의 117 line.   
현재 VOC12ClsDataset Dataset의 __getitem__의 return은 name (확장자 제외), img, label  
train.py에서 enumerate(train_data_loader)로 (name, img, label) pack을 불러와서 model.unpack(pack) 수행  
117 line에서 self.name = pack[0][0] 인데 왜 [0][0]? pack[0]이 모든 batch의 image name을 받을 수 있지 않나?


model_cse.py 239 line  
denorm()을 하는데 norm_tensor가 불린적이 없는데 denorm을 하는이유?
=> dataset 생성시 class normalize를 부르는데 여기서 normalize 시킴. 이를 denorm하는것.


model_cse.py infer_msf() 233-236 line
왜 image에 존재하는 object에 대해서만 cam을 저장?   
이후에 evaluation.py do_python_eval()의 compare() task='cam'에서 이 cam들 값과 threshold값만 갖고 argmax해서 pseudo label을 만드는데   
이러면 실질적으로 cam이 예측한 최대 값이 image에 있는 object가 아니라면 cam은 잘못 예측했는데 올바르게 예측했다고 값이 나올 수 있다고 생각.  
또한 image에 있는 object에 대해서만 pseudo label을 만드는게 일반적인 것인지 궁금. 


## MCTformer
engine.py line 199  
MCTformer에서도 cam을 image에 존재하는 object에 대해서만 저장  
evaluation.py의 do_python_eval() line 33~36  
저장한 cam dict를 불러와서 그 값들에 대해서만 argmax로 pseudo label 생성  
일반적으로 이렇게 계산하는 것인가?


vision_transformer.py
self.norm = nn.LayerNorm(embed_dim)  
forward_features() line 244  
block을 모두 통과한 후 x = self.norm(x) 즉 LayerNorm을 통과  
그 후 cls token만 가져감  
이러면 LayerNorm 즉 해당 token의 embed_dim 방향으로 normal distribution normailzation 수행한 결과?  
cls token의 평균 = 0, sigma = 1 ?   
이 값을 mlp last layer에 넣는게 맞는지? 


models.py  
반면에 MCTformer는 마지막 block을 모두 통과한 후 cls tokens, patch tokens를 그대로 가져옴  
그 후 cls token을 line 105에서 x_cls_logits = x_cls.mean(-1)을 통해서 embed_dim 방향으로 average pooling 수행  
이 수행한 값을 engine.py line 35에서 그대로 prediction으로 사용  
-> 값 자체는 0~1 사이 값이 아닌 임의의 값을 지니게 되는 것이 맞는가?


main.py line 167 168  
gen_attn 값을 변경하며 따로 생성  
datasets.py build_dataset() gen_attn값을 넣어서 Dataset 생성  

각 dataset class (ex VOC12Dataset)에서 gen_attn의 경우 train or gen_attn이면 'tran_aug_id.txt를 사용  
어차피 둘다 train_aug를 사용하는데 굳이 dataset_train_과 data_loader_train_을 생성해서 사용하는 이유?  
=> batch size가 data_loader_train_은 1인데 이 차이를 두기 위해서  
data_loader_train_은 gen_attention_maps에서만 사용되는 dataset. run.sh를 보면 generate class-specific localization maps에서 VOC12MS 사용을 하고 있음. 해당 dataset은 flipping, scaling으로 8장 image list 반환.
 

engine.py line 188  
unsqueeze로 dim 추가가 필요한 이유를 모르겠음. 뭔가 dimension을 맞추는 것 같은데   
line 200에서 sum_cam[b, cls_ind, :]로 바로 batch를 접근


infer_aff.py line 125  
paper page5에서는 row-wise normalization이라고 하는데 col-wise norm이 아닌지?  
col 마다 합이 1이 되도록 나눠줌.  
=> line 134에서 cam_vec 과 trans_mat이 곱해지는 순서가 paper page5와 반대. col-wise가 맞음