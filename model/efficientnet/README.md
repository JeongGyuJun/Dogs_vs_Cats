# classification_efficientnet
시작 하기 전, 이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용하였다. 실험 결과로 (Train, Validation)Accuracy, Loss, feature map, Grad CAM을 확인할 수 있다. ImageNet에서 84.4% top-1 / 97.1% top-5 정확도를 달성함.

#### - 데이터는 kaggle에서 제공하는 Cast-vs-Dogs 데이터를 이용하여 2만 5천장에서 1만 7천개를 train_data, 4천개를 validation_data, test_data 4000으로 나누어 실험을 진행하였다.

### EfficientNet
모델 스케일링을 체계적으로 연구하여 네트워크 Depth, Width, Resolution의 세심한 균형이 더 나은 성능으로 이어질 수 있음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81704346-a01ea180-94a8-11ea-808d-557faca9e9e8.png" width="75%"></p>

따라서 복합 계수를 사용하여 모든 차원의 Width/Depth/Resolution을 균일하게 확장하는 새로운 스케일링 방법을 제안함.

위 3가지 요소를 확장하여 b0 ~b7의 결과를 보여줌. 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81704373-a876dc80-94a8-11ea-881a-cd28bab5c6af.png" width="50%"></p>

위 그래프 결과 다른 모델과 비교 하였을 때 더 적은 매개변수로 높은 정확도를 얻을 수 있음을 보여줌.
그리고 확장한 결과 dataset에 최적화되어 학습하는 것을 볼 수 있음.

직관적으로 복합 스케일링 방법은 입력 이미지가 더 큰 경우 수용 필드를 증가시키기 위해 더 많은 레이어가 필요하고 세밀한 패턴을 캡처하기 우해서 더 많은 채널이 필요하기 때문에 효율적으로 접근할 수 있었음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81704417-b6c4f880-94a8-11ea-9423-6db75c945d60.png" width="50%"></p>

위 표는 efficientnet_b0 전반적인 기본 구성을 보여주며 여기서 Depth, Width, Resolution을 균일하게 확장하여 스케일링을 하였음. 여기서 efficientnet b0~b7까지 논문에서 제시함.

### optimizer(SGD, Adam)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83730724-6fcfba80-a684-11ea-8468-7e867e0e98e4.png" width="50%"></p>

learning rate : 0.001 로 계속 학습 진행했으며(비교가 목적) 여기서 주요 관점은 optimizer에서 대표적으로 자주 사용하는 SGD와 스텝 방향과 사이즈에 대한 부분을 개선한 Adam을 사용하여 학습에 대한 속도와 모델의 성능을 이끌 수 있는 것을 비교하여 확인하고자 했다.

#### 설정
모델 빌드의 경우 옵티마이저로 SGD(learning rate = 0.001, momentum = 0.9)함수와 Adam(learning rate = 0.001)을 사용했으며 손실 함수는 binary_crossentropy로 설정했으며 하이퍼 파라미터는 250epoch, (image_size 224, 224, 3), step(train, validation 90, 20),batch_size(train, validation 20)으로 설정했다.

그리고 효율적으로 모델 성능을 이끌고자 추가적으로 Data Augment을 다음과 같이 사용함.

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255
    
#### 학습 결과   

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84604648-70f9b680-aed2-11ea-806c-6485d80dce5e.png" width="65%"></p>

<table border="1">
<th>Train Model Optimizer</th>
<th>Max Train Accuracy</th>
<th>Min Train Loss</th>
<th>Max Validation Accuracy</th>
<th>Min Val Loss</th>    
<tr>
<td>SGD Optimizer</td>
<td>0.9456</td>
<td>0.1300</td>
<td>0.9550</td>
<td>0.1331</td>    
</tr>    
<tr>
<td>Adam Optimizer</td>
<td>0.9562</td>
<td>0.1095</td> 
<td>0.9725</td>    
<td>0.1012</td>    
</tr>    
</table>    

위와 같이 optimizer를 제외한 나머지 설정을 똑같이 하였을 때, SGD의 결과는 accuracy 94.65%, loss 0.1321가 나왔으며 Adam은 accuracy 95.90%, loss 0.1053이 나왔다. 근소하지만 그래도 Adam이 최적값에 더 빠르게 수렴하면서 더 높은 성능을 얻을 수 있었다.   

#### - feature map
Feature map을 원할하게 보기 위한 연산을 적용함으로써  MBConv Block에서 반환되는 것에 대한 레이어층의 피쳐 맵을 가져와 시각화 함.

#### SGD

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81701238-b296dc00-94a4-11ea-8b1e-346d5b5dc1ca.png" width="75%"></p>

#### Adam

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81700850-30a6b300-94a4-11ea-84ec-837488d476d3.png" width="75%"></p>

맨눈으로 구별하기는 어려웠다. 그래도 이 데이터에서는 두개의 optimizer를 사용했을 때 많은 특징을 찾을 수 있었다.

#### - Grad CAM:Generalized version of CAM(cat, dog)
간단히 말해서, Grad CAM은 얼굴 위치 추적기라고 부르며 Grad CAM을 원할하게 보기 위한 연산을 적용함으로써  모델이 이미지에 대해서 어느 위치를 보며 예측을 한 것인지 레이어층 별로 알 수 있다.

#### SGD

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81701314-cb06f680-94a4-11ea-9971-b0d18f61a465.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81701360-db1ed600-94a4-11ea-9915-0ed6bc760365.png" width="50%"></p>

#### Adam

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81700964-53d16280-94a4-11ea-9f0a-e9182f2ccf92.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81701018-66e43280-94a4-11ea-9b51-aeee87bb9dfb.png" width="50%"></p>


위의 Grad CAM을 보면 SGD보다 Adam의 결과를 보면 전반적으로 특징을 보다 더 뚜렷하게 찾아서 보여주는 것을 확인할 수 있었다.
그리고 SGD의 경우는 predict과 grad cam의 이해하기 어려운 결과를 보기도 하였다. 
