## image_convert
문제가 될 경우 삭제하겠습니다.

시작 하기 전, 이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용함. 

#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하여 2만 5천장에서 1만 7천개를 train_data, 4천개를 validation_data, test_data 4000으로 나누어 실험을 진행하였음.

### 실험 모델
VggNet(Layer 16, epoch 250, optimizer SGD(lr=0.001, momentum=0.9), batch_size 20, step(train, batch) 80, 20, loss function binary crossentropy, dense layer activation sigmoid)

Data Augmentation

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255

### 모델 학습 결과(성능)
Accuracy 95.05%, Loss 0.1351

####  데이터
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980644-611e2780-a952-11ea-9fdb-dc2bc5e42924.png" width="100%"></p>

위 데이터를 가지고 여러 인위적인 변화(노이즈)를 주어 모델의 예측 성능과 Grad-CAM을 확인하고자 함.

#### 실험
이 실험에서는 이미지 데이터에 영향을 받을 수 있는 여러 상황들이 있지만 그 중에서도 빈번하게 실생활에서도 영향을 받을 수 있으며 또는 인위적으로 데이터에 잡음 ,노이즈 등 변화를 주는 것으로 학습된 모델의 성능이 어떻게 달라지는지 확인하고자 함.

##### predict score x가 0.5000 이상이면 개의 클래스로 예측한 것이며 0.5000 미만이면 고양이의 클래스로 예측한 것이다.

그래서 다음과 같이 대표적으로 사용하는 영상처리 기법 중 일부를 정하여 실험을 진행함. 

    - pixel update(add, sub), line, gauss, salt & pepper, poisson, speckle, cellphone(purpose DSP), data format(predict scores)
    
#### Pixel Update
Color 각 RGB에 맞게 원하는 목적에 따라 작업을 변경하여 미세한 차이와 급격한 차이에 대해서 확인하고자 함.

##### - pixel_add
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980655-73986100-a952-11ea-8497-64fa468f1909.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980813-a0994380-a953-11ea-8725-3f443d1976bf.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980817-a858e800-a953-11ea-86e5-60b81c2ead99.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980819-af7ff600-a953-11ea-907b-74bc827779ee.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 덧셈 연산의 변화를 주어 확인하였을 때 약간의 변화는 크게 영향을 끼치지는 않지만 50pixel 이상부터는 확실히 모델의 성능이 떨어지는 것을 확인할 수 있다. 그런데 고양이의 경우는 크게 예측의 변화가 있지는 않지만 개의 경우는 모델 예측의 성능이 크게 떨어지는 것을 확인할 수 있다. 위의 이미지를 보면 사람 손을 보고 오히려 잘못된 예측을 하는 경우까지 발생했다.

##### - pixel_sub
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980861-fc63cc80-a953-11ea-9a5f-0c97bd3c0fd8.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980866-02f24400-a954-11ea-817e-9077db0d62e1.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980870-08e82500-a954-11ea-9685-2f748e294628.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83980874-0ede0600-a954-11ea-8eac-a3f735ed07fa.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 뺄셈 연산의 변화를 주어 확인하였을 때 약간의 변화는 크게 영향을 끼치지는 않지만 50pixel 이상부터는 덧셈과 다르게 둘 다 많은 변화가 있어 확실히 모델의 성능이 떨어지는 것을 확인할 수 있다. 그래도 고양이보다는 개가 확실히 더 떨어지는 것을 볼 수 있다.

##### 전체적으로 미세한 변화는 모델 예측과 그에 따른 Grad-CAM에 영향을 크게 미치지는 않았지만 그래도 약간의 변화들을 확인할 수 있었으며 급격한 변화로 100pixel 이상부터 모델 예측 성능에서 원 이미지보다 크게 벗어나는 것을 확인이 가능 했다.

### Noise
이미지에 여러 인위적인 노이즈를 추가함으로써 강도에 따른 변화를 확인하고자 함.

#### - Gauss
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981037-413c3300-a955-11ea-82ee-e502e18d708c.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981039-47caaa80-a955-11ea-8f5a-a8f1a9982d8c.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981046-4e592200-a955-11ea-8023-fbb97654a19a.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981050-53b66c80-a955-11ea-92a7-35d7b8588db5.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 Gauss Noise의 변화를 주어 확인하였을 때 고양이에서는 크게 모델 예측 성능이 떨어지지는 않았지만 개의 경우는 확연하게 떨어지는 것을 확인할 수 있다. Grad CAM을 통해서 고양이, 개 둘 다 모델이 바라보는 곳을 확인한 결과 이미지의 전체적으로 바라보는 것을 볼 수 있었다. 그렇지만 고양이의 경우 크게 영향을 받지 않는 것에 대한 궁금증을 여전히 안고 갈 수 밖에 없었다.

#### - Salt & Pepper
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981088-be67a800-a955-11ea-94d3-18d48ebde9ee.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981092-c4f61f80-a955-11ea-9fbb-aae13de91825.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981093-ca536a00-a955-11ea-8662-cb785defcf33.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981096-d2130e80-a955-11ea-9e3e-0401bafb6c5f.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 Salt & Pepper Noise의 변화를 주어 확인하였을 때 0.7비율로 소금과 후추의 노이즈를 주었을 때 모델 예측의 변화가 크게 있었으며 이번의 경우는 이전과 다르게 반대로 고양이의 모델 예측 성능이 크게 떨어지며 개의 경우는 크게 변화가 없었다.

#### - Poisson
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981119-19999a80-a956-11ea-808f-eca395ec8c66.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981122-1e5e4e80-a956-11ea-9e31-b4b9fdaeece5.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981125-24ecc600-a956-11ea-9e4e-f4f40d002d77.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981127-29b17a00-a956-11ea-892f-2bf55ac49fdb.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 Poisson Noise의 변화를 주어 확인하였을 때   -   보류

#### - Speckle
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981135-36ce6900-a956-11ea-82f6-569748af3ced.png" width="100%"></p>

기본적인 픽셀 값에서 추가로 Speckle Noise의 변화를 주어 확인하였을 때 개의 경우는 확연히 떨어지는 것을 볼 수 있었으며 고양이의 경우 크게 변화가 없었다. 이전 Salt & Pepper Noise와 비슷한 상황이라고 생각이 들었다.

##### 사람 눈에는 잘 보이지 않는 데이터에 작은 변화가 있을 때도 모델이 예측하는 predict score 달라지는 것을 볼 수 있다.

#### - CellPhone(galaxy 9+, iphon6, gpro) - DSP
위 실험에서 연구하고자 하는 것을 말하기 전에 일단 관련된 주제에 대해서 간단한 소개를 하자면 CPU와 다향한 HW를 하나의 Chipset에 집적하는 것을 SoC (system on chip)라고  부른다. 특히, 이런 SoC중에 시스템의 중심에서 전체를 Control하고 특별한 지시를 수행하는 제일 중요한 SoC를 DSP(Digital Signal processor)라고 한다. 이 DSP는 카메라의 센서에서 받아 들인 영상정보를 분석해서 저장하거나 Display에 띄우는 역할을 하는 것으로 카메라 영상 정보와 밀접한 관련이 있다. 그래서 각 다른 기기에서 같은 상황의 이미지를 촬영하여 데이터를 분석하고자 한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981142-4352c180-a956-11ea-9997-8ecc3bc079b7.png" width="100%"></p>

그 결과 위의 이미지와 이 모델에서의 예측 점수를 확인할 수 있다. 이는 각각의 기기에서 카메라의 센서들이 받아 들인 영상정보가 미세하게 다름으로써 발생하는 것이라고 생각이 되며 이것이 환경과 구성하는 모델 등 상대적으로 영향을 받을 수도 있다고 생각할 수 있다.

#### - Data format
화질, 압축, 용량 등 영상 데이터 영상을 저장하는 방식이 다 다르며 각 포맷과 확장자마다 장점과 단점을 가지고 있다. 그래서 이 영상 데이터를 다음과 같이 포맷과 확장자를 다르게 저장함으로써 모델 예측 성능이 어떻게 달라지는 것을 확인하고자 한다. 여기서는 기존에 사용한 jpg, 추가로 png, bmp, tiff 확장자를 확인할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83981158-5e253600-a956-11ea-9a18-78d6e07b678f.png" width="100%"></p>

결과적으로 기본적인 영상에서는 다른 것을 확인할 수 없었으며 이는 영상 데이터의 복잡성(이미지 크기, 픽셀 etc..)이 간단하기 때문이라고 생각이 든다.

##### 이전 Grad CAM에서더 확인할 수 있었듯이 확실히 고양이보다는 개를 잘 못 예측한 경우가 많았던 것을 확인할 수 있었다. 그만큼 여기서도 확인할 수 있었던게 노이즈나 잡음 등에 민감하게 영향을 받을 수 있단느 것을 확인할 수 있었다.

##### 전반적으로 데이터에 자연 또는 인위적으로 변화를 주어 확인한 결과 모델이 학습한 클래스가 가지고 있는 최적화 가중치 값과 사이의 간격이 멀어진다고 생각이 들며 전반적으로 개의 경우 크게 영향이 미쳤다. 이는 이전에 데이터 분석 파트를 먼저 연구한 결과 개의 경우 특정 한 곳을 바라보고 예측하는 것이 아닌 얼굴 전체(코, 눈, 귀, 입)를 보고 예측한다고 볼 수 있었다. 그와 반대로 고양이는 크게 영향을 받지 않았다. 가끔 Salt & Pepper Noise와 같이 예외 경우가 발생하기도 했다.

