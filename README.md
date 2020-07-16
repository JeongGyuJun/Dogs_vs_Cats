## kaggle_dogs-vs-cats
본 데이터는 Kaggle 공개 데이터를 이용함.

이미지에 개 또는 고양이가 포함되어 있는지 여부를 분류하는 알고리즘으로 이것을 사람이 구별하기는 쉬울 수 있지만, 컴퓨터에서는 조금 어려울 것임.

### 데이터 정보

<p align="center">Dogs-vs-Cats Class</p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83288445-51019c00-a21e-11ea-8556-b35142464dc6.png" width="50%"></p>

    데이터 포맷 : jpg, jpeg
    총 25,000장, Train : 17,000장, Validation : 4,000장, Test : 4,000장
    해상도 : 고유 이미지 사이즈가 다름.
    class : Dog, Cat

다음 아래와 같이 전체적인 데이터를 한 눈에 확인할 수 있음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83288474-5f4fb800-a21e-11ea-8219-ad1218bcd73b.png" width="75%"></p>

#### 데이터 보완
데이터 레이블을 변경하지 않고 픽셀 변화시키는 CNN의 성능 향상을 위하여 Tensorflow에서 제공해주는 Data Augmentation을 사용함.

다음과 같이 데이터에 적용함.

    Train data : rotation_range = 40, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, rescale = 1./255
    Validation data : rescale = 1./255
    Test data : rescale = 1./255

### 실험 소개 및 수행 내역
###### 자세한 내용은 각 항목의 제목을 클릭하면 결과를 확인이 가능함.

단순한 모델 학습이 아닌 각 모델마다 비교하고자 하는 것을 하나 정하여서 하고자 함.

그래서 다음과 아래와 같이 순서대로 비교하여 진행했으며 실험 내용을 확인하고 싶으면 밑에 리스트 항목을 클릭하면 이어서 볼 수 있다. 


1. [VggNet(Accuracy - Featrue Map)](https://github.com/JeongGyuJun/classification_vggnet)


2. [ResNet(Layer - 50, 101 Layer)](https://github.com/JeongGyuJun/classification_resnet)


3. [DenseNet(Pooling - Max, Avg)](https://github.com/JeongGyuJun/classification_densenet)


4. [EfficientNet(Optimizer - SGD, Adam)](https://github.com/JeongGyuJun/classification_efficientnet/blob/master/README.md)

5. [Data Analysis(Grad CAM 이용)](https://github.com/JeongGyuJun/Cats-vs-Dogs-Data-grad-cam)

6. [Image Convert(Pixel Update, Noise, Formatting)](https://github.com/JeongGyuJun/image_convert)
