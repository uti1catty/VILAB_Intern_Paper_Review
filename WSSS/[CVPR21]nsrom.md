# [CVPR21] Non-Salient Object Mining for Weakly Supervised Semantic Segmentation 

github: https://github.com/NUST-Machine-Intelligence-Laboratory/nsrom

Graph based global reasoning unit to strengthen the classificaton network's albility to capture global relations among disjoint(분리된) and distant(멀리) regions.  
-> help network activate the object features outside the salient area.  

Exert the segmentation network's self-correction ability  
1. Potential object mining module: reduce the false-negative rate in pseudo labels.  
2. Non-salient region masking module: (for complex images) to generate masked pseudo labels. Help discover the objects in the non-salient region.  

# 1. Introduction 
기존의 바법들은 salient region을 위한 response map 확장에 집중한다. 그리고 salency map에서 bg를 추출한다.   
우리는 non-saient region object mining을 시도한다.  
non-salient region은 일반적으로 corner나 edge근처에 산개해있다.

CNN기반의 traditional classifcaition Netwrok는 local relation에 집중하여 disjoint and distant regions 사이의 global relation을 capture하는데 inefficient하다.  
따라서 $\mathbf{Graph\_based\ Global\ Reasoning\ Unit}$ to strenghthen classifciation network's capabiity in activating the object features outside the salient region.  

기존의 방법들은 activated region을 확장하는 방법을 사용하여 inevitably object area를 bg로 넘는다.  
이에 saliency map을 bg clue로 제공하는데 saliency map이 conspicuous region근처의 pixel label을 잘 correct하지만 outside the salient area의 object label을 remove한다.  
연구진은 CAM이 sparse 하고 incomplete하여 accurate boundary를 갖고 있지 않지만 "useful clues for the objects in the non-salient region을 제공한다"는 사실을 확인했다.  
따라서 $\mathbf{Potential\ Object\ Mining\ Module}$을 이용하여 naive CAM에서 activate되는 outside the conspicuous region의 object를 discover한다. -> reduce false-negative rate of pseudo label을 목표로 함.  

Segmentation model  
dataset에서 complex image (2개 이상의 category)는 object가 outside of salient area에 있기 쉽다.  
따라서 $\mathbf{Non\_salient\ Region\ Masking\ Module}$ for complex image로 masked pseudo label을 생성한다.  

# 3. The Proposed Approach
기존의 방법들은 Salient area의 pseudo label을 refine하는데 집중.  
우리는 non-salient region에서 more object를 discover.  
1. Graph-based global reasoning unit in classification network.  
help to activate the object features outside the salient region  
2. Potential object mining module (POM)  
3. Non-salient region masking module (NSRM)  
Improve the quality of pseudo labels for non-salient region object mining.  

## 3.1 CAM Generation 
CAMs(class attention maps)를 생성하기 위한 classification network  
"Capture global relations among disjoint and distant region 능력 강화" -> graph-based global reasoning unit before final classifier.  

Features from encoder $X \in \mathbb{R}^{L \times K},\ K: feature\ dimension,\ L=H \times W\ locations$  
Project from coordinate space to a latent interaction space  
projection function: $V = f(X) \in \mathbb{R}^{N \times K}$   
$$v_i = b_iX = \sum_jb_{ij}x_j$$
$$B=[b_1, ..., b_N]\in\mathbb{R}^{N \times L}:\ Learnable\ Projection\ weight$$
$$N: Number\ of\ the\ features(nodes)\ in\ interaction\ space$$

Graph convolution to capture the relations between features in new space  
$$Z = ((I - A_g)V)W_g\ \in \mathbb{R}^{N \times K}$$
$$A_g: N \times N\ adjacency\ matrix\ learned\ by\ gradient\ descent\ during\ training$$
$$W_g: state\ update\ function$$
reverse projection $Y = g(Z) \in \mathbb{R}^{L \times K}$ 적용.  
$$ y_i = d_iZ = \sum_jb_{ij}z_j$$
$$ D = [d_1, ..., d_N] = B^T$$

$Y$ 값을 classifier에 넣고 GAP로 $p_c$ 구함

multi label soft margin loss  
$$L_{cls} = - {1 \over C}\sum_{c=1}^Cy_clog\sigma(p_c) + (1-y_c)log[1 - \sigma(p_c)]$$
$p_c$: prediction of the network for c-th class  
$\sigma()$: sigmoid  
$C$: foreground class number  
$y_c$: image level label for c-th class.  

final classifier에서 CAMs 얻음. (여기에선 1x1 conv with C channel을 pixel-wise classifier로 사용)  
OAA를 따라서 online accumulated class attention maps (OA-CAMs)도 함께 생성  
OA-CAMs: have more entire regions / Strengthen the lower attention values of target object regions with their integral attention model.  

## 3.2 Potential Object Mining
work of OAA가 OA-CAMs를 통해 object cues를 추출하고, saliency map에서 background cues를 추출.  
value of each OA-CAM을 비교하여 class label of each pixel assign.  
saliency map이 shape information을 제공해주므로 initial label은 Background extraction(BE)이후에 clear object boundaries를 갖지만 outside the conspicuous area에서 many object parts를 miss한다.  

recall: TP/(TP + FN), precision: TP/(TP + FP)
OA-CAMs: high recall, low precision  
CAMs: low recall, high precision  

Potntial object mining (POM) module to discover the object region activated in the CAM.  
Mine potential object with a class adaptive threshold $T_c$ for class $c$  
$$T_c = \begin{Bmatrix} MED(v)\ \ if\  \exist(i,j),\ s.t.\ l_{ij}=c \\ TQ(v)\ \ otherwise \end{Bmatrix}$$
$MED$: median of input   
$TQ$: top quartile of input  
$v$: set of attention values of pixels in the CAM, whose locations $p$ are selected  
$$p =   \begin{Bmatrix} \{(i, j)|l_{ij}=c\}\ \ if\  \exist(i,j),\ s.t.\ l_{ij}=c \\ \{(i, j)|a_{ij} \gt T_{bg}\}\ \ otherwise \end{Bmatrix}$$
$a_{ij}$: attention value in CAM at the position $(i, j)$  
$l_{ij}$: value in the inital label at position $(i, j)$, pseudo label of pixel.  
즉 만약 inital label이 class $c$를 포함하고 있다면 class c로 label된 pixel들의 attention 값의 Median 값을 $T_c$로 선정.  
inital label이 class $c$를 포함하지 않는다면 CAM attention 값이 $T_{bg}$ 값을 넘는 모든 pixel을 선정하여 해당 pixel들의 attention 값의 Top quartile 값을 $T_c$로 선정.  

inital label을 다음과 같이 변경  
$$l_{ij} = \begin{Bmatrix} 255\ \ if\  \exist c,\ (i,j),\ s.t.\ l_{ij}=0,\ a_{ij}^c \gt T_c \\ l_{ij}\ \ otherwise \end{Bmatrix}$$
$a^c$: CAM for class $c$  
즉 bg pixel (labeled as 0) in initial label 인 position (i, j)인데 어떤 class $c$에 대하여 CAM의 값이 $T_c$를 넘는 $c$가 있다면 255로 label하고 training에서 무시한다. (이 영역이 potential object region)  
이단계에서는 이 potential object region을 label해주지 않는데 이는 avoid introducing wrong object label을 위해서이다. 
여기서 false-negative rate of pseudo labels를 줄여 misleading information으로부터 생성되는 gradient를 discard하도록 돕는다.  

## 3.3 Non-Salient Region Masking
POM이 pseudo label을 more ignored pixel과 함께 enrich한다.  
Segmentation netwrok가 training동안 이 potential object region에 대해 correct label을 predict하도록 allow한다.  

complex image는 salient area 외부에 object를 가질 확률이 높다.  
이 경우 salient area에 object label만 갖는 pseudo label만으로는 segementation network가 objects outside the salient region을 detect하게 하기 어렵다.  

Non-salient Region Masking (NSRM) module  
pseudo label로 학습시킨 segmentation network의 initial Prediction, Pseudo label 이 필요하다.  
$\mathcal{Main\ Assumtion}$: object labels within the salient region are correct with high probability.  
1. object region in eh inital prediction을 guidance of our pseudo label로 expand한다.  
2. expanded prediction map에서 object mask를 extract한다.  
3. object mask를 dilation operation으로 expand한다. -> object 바로 외부의 bg를 포함시키기 위함.
4. expanded prediction map에 masking operation을 적용하여 Masked Pseudo Label을 얻는다.  

3번에 의해 masked pseudo label에서 object의 바로 외부의 bg는 bg로 남게 되어 object의 boundary에 대한 정보를 담는다. 그 밖은 모두 blank(label 정의 X)로 되어 Segmentation Network가 training하는 동안 스스로 유추만 하도록 하여 non salient region의 object를 detect할 수 있도록 유도한다.  

