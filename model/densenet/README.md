# classification_densenet
시작 하기 전, 이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용하였다. 실험 결과로 (Train, Validation)Accuracy, Loss, feature map, Grad CAM을 확인할 수 있다.

#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하여 2만 5천장에서 1만 7천개를 train_data, 4천개를 validation_data, test_data 4000으로 나누어 실험을 진행하였음.

## DenseNet(Densely Connected Convolutional Networks)
DenseNEt은 각 Layer와 후속 Layer 사이에 각각 하나씩 직접 L(L+1) 연결을 가지고 있다. 각 Layer는 모든 이전 Layer의 feature map을 입력으로 사용한다. 이 구성은 소멸 단계 문제를 완하하고, 전파를 강화하며, 재사용에 용이하며, 매개변수 수를 크게 감소시킨다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75467606-44776880-59cf-11ea-987c-26462f352d1a.png" width="75%"></p>

이전 ResNet에서도 유사한 방법이 사용되었다.
하지만 ResNet은 feature map을 추가하는 방식인 반면 DenseNet은 각각의 feature map을 연결하기 위한 구성을 지니고 있다.

#### Growth Rate
각 feature map이 연결되어 있기 때문에 feature map의 채널 수가 많으면 채널을 각 채널별로 계속 연결할 수 있어 더 많은 채널로 이어질 수 있다. 그래서 DenseNet에서는 각 계층의 feature map에 매우 적은 수의 채널을 사용하는데, 이를 각 계층의 피쳐 맵에 대한 채널 수(k)라고 한다. 이러한 방식으로 다음 밑에서 Transition Layer을 소개하고자 한다.

### Bottleneck Layer
DenseNet은 ResNet과는 다른 Bottleneck Layer 구조를 가지고 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75468567-c0be7b80-59d0-11ea-93c9-ad5869a5bae8.png" width="75%"></p>

3 x 3 Convolution, 1 x 1 Convolution을 볼 수 있다. 여기서 입력 feature map의 채널 수만큼 만드는 것이 아니라, growth_rate만큼 feature map을 만들어 계산 비용을 줄일 수 있다. 이는 효율적으로 매개변수를 줄일 수 있다.

### Transition Layer
단순하게 말하면, feature map의 수를 줄이는 기능이다.

### Composite function

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75469140-d54f4380-59d1-11ea-8317-273f6dcacf39.png" width="75%"></p>

다음은 pre-activation 구조로 BatchNormalization - ReLU function - Convolution Sequence를 사용함.

### Avg pooling vs Max pooling
Pooling은 CNN에서 매우 중요한 역할을 한다. Subsampling을 사용하여 위치나 이동에 강한 특성을 가진 feature map의 크기를 줄일 수 있다.

Average Pooling 방법은 많은 ReLU를 활성 함수로 사용한다. 이로 인해 0이 많이 발생하며, 평균 작동에 의한 강한 자극이 감소한다. 또한 강한 자극과 약한 자극의 평균을 취하면 상쇄되는 상황이 발생할 수 있다.

반면에 Max Pooling 방법은 학습 데이터에 지나치게 적합할 수 있다. 그리고 추가로 Stochastic Pooling 개념도도 추가적으로 알고 있으면 좋을 것 같다.

### 설정
DenseNet 121 Layer을 사용했으며 Pooling에 대해서 비교하고자 하였다.

Max, Avg pooling 모델 빌드의 경우 옵티마이저로 SGD(learning rate = 0.001, momentum = 0.9)함수를 사용했으며 손실 함수는 binary_crossentropy로 설정했으며 하이퍼 파라미터는 250epoch, (image_size 224, 224, 3), step(train, validation 90, 20),batch_size(train, validation 20)으로 설정했다.


그리고 효율적으로 모델 성능을 이끌고자 추가적으로 Data Augment을 다음과 같이 사용함.

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255

### 학습 결과

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84610548-e32fc280-aef5-11ea-9135-17aaa74ad66b.png" width="65%"></p>

<table border="1">
<th>Train Model Pooling</th>
<th>Max Train Accuracy</th>
<th>Min Train Loss</th>
<th>Max Validation Accuracy</th>
<th>Min Val Loss</th>    
<tr>
<td>Avg Pooling</td>
<td>0.9075</td>
<td>0.2193</td>
<td>0.9000</td>
<td>0.2536</td>    
</tr>    
<tr>
<td>Max Pooling</td>
<td>0.92625</td>
<td>0.1779</td> 
<td>0.9325</td>    
<td>0.1964</td>    
</tr>    
</table>    

결과적으로는 Avg Pooling은 Accuracy 86.87%, Loss 0.28962, Max Pooling은 Accuracy 90.85%, Loss 0.22649가 나왔다.

결과를 확인하면 확실히 Max Pooling은 데이터에 빠르게 적합하는 것을 볼 수 있으며, 그 반면에 Avg Pooling은 평균 연산에 의해 강한 자극이 감소되어 조금 더 늦게 적합하는 것을 확인이 가능하다.

위와 같은 특성들을 이용하여 결과를 보완하고자 나중에 Ensemble을 진행할 때 많은 도움이 될듯 하다. 왜냐하면 지나치게 적합하는 것과 강한 자극이 감소하여 상쇄되는 상황이 발생할 수 있는 두 가지의 특성을 적절하게 조화로우면 각각의 장점이 살아날 것이라고 생각한다.

#### - feature map(conv_2d layer)
Convolution layer에 해당하는 레이어층의 피쳐 맵을 가져와 시각화 함.

##### avg pooling

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75474243-f74cc400-59d9-11ea-9d09-43faa465a9af.png" width="100%"></p>

##### max pooling

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75474318-1f3c2780-59da-11ea-9948-00dcb768fc68.png" width="100%"></p>

맨눈으로 자세히 보이지는 않지만 실습한 ipynb 파일을 들어가면 조금 더 원할하게 확인이 가능할 것이다. 

#### - Grad CAM:Generalized version of CAM(cat, dog)
간단히 말해서, Grad CAM은 얼굴 위치 추적기라고 부르며 모델이 이미지에 대해서 어느 위치를 보며 예측을 한 것인지 레이어층 별로 알 수 있다.

##### avg pooling

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81388881-eba51880-9153-11ea-8d31-d632d2c846c5.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81388946-02e40600-9154-11ea-8de9-2b6b09ebc347.png" width="50%"></p>

##### max pooling

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81389128-4c345580-9154-11ea-9392-6a529c4b871b.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81389167-5eae8f00-9154-11ea-9f93-185b898be8d8.png" width="50%"></p>

전반적으로 간단한 이미지에 대해서는 비슷하지만 가끔 예외 경우들이 있었다. 우선적으로 이전 VggNet, ResNet모델에서 잘 못 예측한 이미지에 대해서 지금 모델에서는 올바르게 예측하는 경우도 있었다. 위  Dog Image Grad CAM을 확인하면 이전 이미지에서 잘못 예측했던 것을 이 모델에서는 올바르게 바라보고 있는 것을 확인할 수 있다. 이것은 모델마다 특징이 있어 바라보는 것이 다를 수 있다는 것을 확인이 가능하며, 추가로 사람의 신체 중 한 곳을 바라보거나 특징을 정확히 찾지 못하여 고양이로 잘 못 예측하는 경우들이 있었다.

