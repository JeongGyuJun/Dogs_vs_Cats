# classification_vggnet
시작 하기 전, 이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용하였다. 실험 결과로 (Train, Validation)Accuracy, Loss, feature map, Grad CAM을 확인할 수 있다.

#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하여 2만 5천장에서 1만 7천개를 train_data, 4천개를 validation_data, test_data 4000으로 나누어 실험을 하였음.

### VGG16
VggNet은 2014년 ILSVRC에서 2위를 차지했지만 훨씬 단순한 구조로 이해와 변형이 용이하다는 장점이 있다.

특징으로는 VGGNet은 작은 필터 크기의 컨볼루션 연산이 있으며, 단점으로는 많은 파라미터가 존재한다는 것을 알 수 있다.
파라미터가 많은 것은 Gradient Vanishing, Over Fitting 등의 문제가 발생할 가능성이 높다는 것을 의미한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74590059-48f66580-504e-11ea-9952-0828a186eb60.png" width="50%"></p>

VggNet 16Layer의 기본 구성을 위 그림에서 확인할 수 있다.

### high, low accuracy
이 실험에서는 높고, 낮은 정확도에 대해서 비교하고자 실험을 진행했다. 여기서 주로 보는 관점은 정확도의 수치를 기준으로 모델이 어디를 어떻게 바라보며 예측하는 것에 대해서 보고 싶었다.

### 설정
실험에서 추가로 각 블록에 Kernel_initializeer = 'he_normal'을 추가하였으며 Vanishing/Exploding을 완하하기 위해 Convolution과 Max Polling 사이에 Batch Normalization을 추가 하였다. 그리고 출력층에 대해서 Multi Class가 아니여서 softmax가 아닌 binary class에 적합한 sigmoid함수를 사용했다.

모델 빌드의 경우 옵티마이저로 SGD(learning rate = 0.001, momentum = 0.9)함수를 사용했으며 손실 함수는 binary_crossentropy로 설정했으며 하이퍼 파라미터는 250, 1epoch, (image_size 224, 224, 3), step(train, validation 80, 20),batch_size(train, validation 20)으로 설정했다.

그리고 효율적으로 모델 성능을 이끌고자 추가적으로 Data Augment을 다음과 같이 사용함.

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255

추가로 비교 실험으로한 모델은 정확도를 낮추기 위한 모델은 개와 고양이 데이터 수를 3장씩하여 라벨으 거꾸로 설정해서 모델 학습을 했다.

### 학습 결과

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84385924-5ce05b80-ac2b-11ea-9b0f-687dcf635aa3.png" width="80%"></p>

###### Accuracy 50.35%, Loss 0.6931의 history는 epoch이 낮음으로 별도로 표시하지 않았음.

<table border="1">
<th>Train Model Result</th>
<th>Max Train Accuracy</th>
<th>Min Train Loss</th>
<th>Max Validation Accuracy</th>
<th>Min Val Loss</th>    
<tr>
<td>Accuracy 95.05%, Loss 0.1351</td>
<td>0.9500</td>
<td>0.1458</td>
<td>0.9700</td>
<td>0.0965</td>    
</tr>    
<tr>
<td>Accuracy 50.35%, Loss 0.6931</td>
<td>0.2500</td>
<td>0.6933</td> 
<td>0.5000</td>    
<td>0.6932</td>    
</tr>    
</table>    

기존 데이터 분류한 모델 학습으로는 Accuracy 95.05%, Loss 0.1351 나왔으며 정확도를 낮추기 위한 모델 학습으로는 Accuracy 50.35%, Loss 0.6931 나오는 것을 볼 수 있었으며 정확도를 기준으로의 차이점은 아래 feature map, Grad CAM에서 자세히 확인이 가능했다.

### feature map
Feature map을 원할하게 보기 위한 연산을 적용함으로써 각 레이어층의 피쳐 맵을 가져와 시각화 함.

##### accuracy 95.05%, loss 13.51% feature map

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74591625-d345c600-505c-11ea-8f0f-56a6d300223e.png" width="75%"></p>

##### accuracy 50.35%, loss 69.31% feature map

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83293298-9033eb00-a226-11ea-84f1-1d8cbdac0874.png" width="75%"></p>

높은 정확도와 낮은 정확도의 특징 맵을 비교하였다.

정확히 맨눈으로 보면 높은 정확도는 이미지의 특징을 잘 찾아 대략 모든 피쳐 맵이 콘트라스트한 이미지가 그려지거나 대부분 출력되는 것을 확인할 수 있으며, 정확도가 낮은 것은 스무딩한 이미지가 그려지거나 가끔가다가 안보이느 피쳐 맵도 있었다.

##### Grad CAM:Generalized version of CAM(cat, dog)
간단히 말해서, Grad CAM은 얼굴 위치 추적기라고 부르며 Grad CAM을 원할하게 보기 위한 연산을 적용함으로써  모델이 이미지에 대해서 어느 위치를 보며 예측을 한 것인지 레이어층 별로 알 수 있다.

##### Accuracy 95.05%

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81381921-7da72400-9148-11ea-9a46-d68470ebaa59.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81381990-9a435c00-9148-11ea-9164-6e47b25f6816.png" width="50%"></p>

##### Accuracy 50.35%

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83292912-e81e2200-a225-11ea-8f63-72f75ba4a778.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83292933-ee140300-a225-11ea-8b20-212d28b9b80a.png" width="50%"></p>

상대적으로 높은 정확도 모델에서 올바르게 바라보는 것은 주로 코와 귀 얼굴 중심으로 보고 있었으며, 그렇지 못한 것은 다른 곳을 보는 경우, 사람과 고양이가 같이 있는 경우 등 여러 상황이 있어서 잘못 예측하는 경우도 있었다.반면에 낮은 정확도 모델에서는 특정 한 곳을 바라보는 것이 아닌 전체적으로 바라보며 이것은 특징을 찾지 못하여 모델이 올바르게 작동하지 않는 것으로 생각했다.

결과적으로 데이터 바탕으로 조건이 만족하며 충분하고 올바른 모델 학습이 이뤄진다면 좋은 성능을 이끌 수 있었으며, 그렇지 못한 것은 제대로 동작하지 않아 잘못 판단하는 경우가 많았다.
