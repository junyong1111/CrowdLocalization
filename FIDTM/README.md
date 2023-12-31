# Focal Inverse Distance Transform Map

[Focal Inverse Distance Transform Maps for Crowd Localization](https://arxiv.org/abs/2102.07925)

- **Focal Inverse Distance Transform Map IEEE**
    
    [2102.07925.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f91d9666-0f54-44e2-931e-b8dfd099862c/2102.07925.pdf)
    

**FIDTM**

 군중 분석의 중요한 측면인 **군중 위치 파악**에 초점을 맞춥니다. 이들은 밀집된 군중 장면에서 개인의 위치를 정확하게 찾기 위해 초점 **역거리 변환(FIDT) 맵**이라는 새로운 접근 방식을 제안한다. 밀집도 맵에 비해 FIDT 맵은 밀집된 지역에서 겹치지 않고 개별 위치를 더 잘 표현하며, 제안된 방법은 여러 군중 데이터 세트에서 최첨단 성능을 달성한다.

### 1. INTRODUCTION

 군중 분석에는  Crowd Detection,  Crowd Counting, Crowd Localization 등 다양한 작업이 포함된다. Crowd Detection은 바운딩 박스를 사용하여 개인을 식별하는 것을 목표로 하며, Crowd Counting는 포인트 레벨 주석을 기반으로 장면의 총 인원 수를 추정한다. 이 논문에서는 포인트 레벨 주석을 기반으로 각 사람의 머리 위치를 예측하는 Crowd Localization에 중점을 두어  Crowd Detection및  Crowd Counting에 비해 더 까다로운 작업을 수행한다. 밀집된 군중 속에서 각 사람마다 바운딩 박스에 주석을 다는 것은 비용이 많이 들고 힘든 작업이다. 논 문은문집집도 의 한계를 해결하기 위해 밀집된 지역에서 겹치지 않고 개별 위치를 정확하게 설명하는 새로운 초점 역거리 변환(FIDT) 맵을 제안한다.

<img width="363" alt="1" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/c3c3cddf-5b4b-4553-9edd-06ffdc7507ad">

그림 1. FIDT 맵의 장점
출처 :  [https://arxiv.org/abs/2012.04164](https://arxiv.org/abs/2102.07925)

 그림1 은 군중 위치 파악을 위한 초점 역거리 변환(FIDT) 맵의 장점을 보여준다. 이 이미지에서는 오클루전이 심하고 배경이 어수선하여 포인트 레벨 주석만으로는 개인의 위치를 파악하기 어렵다. 가장 가까운 이웃 거리 정보로 표현되는 FIDT 맵을 사용하면 밀집된 지역에서도 가까운 머리를 구분할 수 있어 가우시안 블롭으로 구성된 밀도 맵에 비해 더 정확한 위치 파악이 가능하다.

 군중이 밀집된 장면에서 개인의 위치를 예측하는 군중 위치 파악의 현재 회귀 기반 방법은 밀집도 맵을 사용하지만 밀집된 지역에서 가우시안 블롭이 겹치기 때문에 정확한 개인 위치와 크기를 제공하는 데 어려움을 겪는다. 이러한 한계는 보행자 추적 및 군중 분석과 같은 고수준 애플리케이션을 사용하는 데 방해가 된다. 이 문제를 해결하기 위해 PSDDN 및 LSC-CNN과 같은 일부 방법은 가장 가까운 이웃 헤드 거리를 활용하여 의사 지상 실측 바운딩 박스를 초기화하지만, 이러한 접근 방식은 여전히 복잡한 감지 프레임워크에 의존하며 헤드 크기가 부정확하게 표현되어 성능이 저하될 수 있다.

 군중 위치 파악의 일부 방법은 사람이 많은 장면에서 개인의 위치를 정확하게 찾기 위해 바이너리형 맵이나 세분화형 맵과 같은 다양한 유형의 맵을 설계하는 데 중점을 둔다. 그러나 이러한 방법은 연결된 구성 요소가 서로 연결되어 여러 개의 머리를 하나로 잘못 예측할 수 있는 밀집된 지역에서는 종종 문제에 직면한다. 이 논문에서는 로컬 최대값을 활용하여 개별 머리 위치를 나타내는 초점 역거리 변환(FIDT) 맵이라는 대안적 접근 방식을 제안하여 매우 밀집된 군중에서도 정확한 정보를 제공한다. 

 기존의 밀도 맵과 달리 FIDT 맵은 밀집된 군중 속에서도 주변 머리와 겹치지 않고 각 사람에 대한 정확한 위치 정보를 제공한다. FIDT 맵은 머리 중심에 가까운 픽셀에 더 높은 응답을 할당하여 맵에서 로컬 최대값의 수를 기준으로 개인을 계산할 수 있다. FIDT 맵에서 각 로컬 최대값은 개별 사람을 나타내며, 상세한 로컬 구조 정보가 있으면 이러한 로컬 최대값을 정확하게 찾는 데 도움이 된다. 예측된 FIDT 맵과 지상 실측 맵 간의 유사성을 개선하는 한 가지 방법은 SSIM 손실을 사용하는 것이다. 그러나 FIDT 맵의 배경 픽셀 값에는 구조 정보가 부족하기 때문에 기존의 SSIM 손실은 잘못된 로컬 최대값으로 이어질 수 있다. 이 문제를 해결하기 위해 독립 SSIM(I-SSIM) 손실이 도입되어 로컬 맥시마의 구조 정보를 캡처하는 모델의 능력을 향상시키고 배경 영역에서 잘못된 탐지를 줄인다.

 이 논문에서 저자는 초점 역거리 변환(FIDT) 맵에서 로컬 최대값을 추출하기 위해 **LMDS(로컬 최대값 감지 전략)**라는 후처리 기법을 제안한다. 이 단계는 밀집된 군중 장면에서 머리를 정확하게 위치 파악하는 데 매우 중요하다. 제안된 방법은 샘플이 부정확하고 군중이 매우 밀집된 까다로운 시나리오에서도 군중 측위에서 최첨단 성능을 달성한다.

**정리**

많은 사람들이 붐비는 장소의 사진이 있고 목표는 군중 속에서 각 사람의 정확한 위치를 정확하게 찾는 것이라 가정하자.

기존에는 밀도 맵이라고 하는 흐릿한 지도를 사용하여 장면에서 사람들이 어디에 있는지 표현했다. 그러나 이러한 지도는 서로 겹치고 정확한 위치를 표시하지 않아 정확도가 떨어졌다. 이를 개선하기 위해 이 논문에서는 밀도 맵 대신 **FIDT(초점 역거리 변환) 맵**이라는 새로운 유형의 맵을 사용할 것을 제안한다. 이러한 FIDT 맵은 겹치는 지역 없이 각 사람이 어디에 있는지 더 정확한 정보를 제공한다.

지도에 점들이 서로 겹치거나 섞이지 않고 각 개인의 위치를 나타내는 점들이 있다고 생각하면 된다.

모델이 이러한 개별 지점에 집중하고 배경 영역이나 다른 피크에 의해 혼동되지 않도록 하기 위해 I-SSIM 손실 함수를 도입하며 이 손실 함수는 모델이 로컬 디테일을 더 잘 학습하고 개인의 머리를 나타내는 특정 포인트만 인식하는 데 도움이 된다.

또한 이러한 정확한 FIDT 맵을 기반으로 군중 속 각 사람의 예측된 머리 중심을 효과적으로 찾아내는 LMDS(Local-Maxima-Detection-Strategy)라는 전략을 설계한다.

### 2 RELATED WORK

 **RELATED WORK**  섹션에서는 군중 수 계산 및 군중 위치 파악을 위한 다양한 방법에 대해 설명한다. 군중 수 계산을 위해 밀도 맵과 경계 상자 주석을 사용하는 방법에 대해 언급하지만, 정확한 머리 위치를 제공하는 데는 한계가 있음을 강조한다. 또한 로컬 최대값을 활용하여 각 사람에 대한 정확한 위치 정보를 제공하는 것을 목표로 하는 군중 위치 측정을 위해 제안된 초점 역거리 변환(FIDT) 맵을 소개한다. 또한 로컬 최대값과 배경 영역을 처리하는 모델의 능력을 향상시키기 위한 독립 SSIM(I-SSIM) 손실과 로컬 최대값 감지 전략(LMDS)의 사용에 대해서도 설명한다.

 현재의 군중 분석 방법은 주로 컨볼루션 신경망(CNN)으로 생성된 밀도 맵을 사용하여 이미지의 사람 수를 계산하는 데 중점을 둔다. 하지만 이러한 방법에는 개별 인물의 정확한 위치 파악이 부족하다. 이를 해결하기 위해 최근 거리 변환 지도와 같은 기법을 사용하여 각 사람의 머리 위치를 예측하는 접근 방식이 시도되고 있다. 본 논문에서 제안하는 방법은 로컬 최대값을 사용하여 머리 위치를 나타내는 FIDT 맵이라는 새로운 레이블을 도입하고, 로컬 영역의 구조 정보에 초점을 맞춘 손실 함수를 활용하여 머리 중심 검출을 개선한다.

<img width="768" alt="2" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/caa0b0f0-05eb-4f68-8775-66620c63a7a7">

그림 2. 파이프라인
출처 :  [https://arxiv.org/abs/2012.04164](https://arxiv.org/abs/2102.07925)

그림 2에 설명된 방법의 파이프라인에서는 훈련 단계에서 평균 제곱 오차(MSE) 손실과 제안된 이미지 구조 유사성(I-SSIM) 손실이 사용된다. 테스트 단계에서는 로컬 최대값 검출 전략(LMDS)을 사용하여 각 사람의 위치를 결정할 수 있으며, 로컬 최대값의 수를 세어 최종 카운트를 얻는다. 또한 더 나은 시각화를 위해 크기 추정 단계를 통해 경계 상자를 얻을 수 있다.

### 3 METHODOLOGY

그림 2에 표시된 바와 같이, 해당 논문에서 제시하는 방법에서는 훈련 단계에서 회귀분석을 사용하여 예측된 FIDT(초점 역거리 변환) 맵을 생성한다. 예측 결과와 실측 결과 사이의 차이를 측정하기 위해 두 가지 손실 함수인 MSE(평균 제곱 오차)와 I-SSIM(구조적 유사성 지수 측정)을 사용한다. 테스트 단계에서는 예측된 FIDT 맵을 생성하고 로컬 최대값을 계산하여 개체 수를 결정할 수 있는 제안된 로컬 최대값 감지 전략(LMDS)을 사용하여 위치 맵을 얻는다. 또한, 더 나은 시각화를 위해 간단한 KNN(K-Nearest Neighbors) 전략을 사용하여 바운딩 박스를 얻을 수 있다.

 이미지에서 머리 위치를 찾는 데 사용되는 초점 역거리 변환(FIDT) 맵의 공식화에 대해 설명하면 다음과 같다. FIDT 맵은 유클리드 거리 변환 맵을 반전하고 추가 조정을 적용하여 머리 중심과 배경에서 멀어지는 응답 감쇠를 개선하여 생성된다. 이 맵은 밀도 맵에 비해 개별 위치를 더 정확하게 표현하고 전경 영역에 집중하는 데 도움이 된다.

 로컬라이제이션 프레임워크에서 회귀 분석은 이미지의 개별 위치를 나타내는 초점 역거리 변환(FIDT) 맵을 추정하는 데 사용된다. FIDT 맵은 LMDS(로컬 최대값 감지 전략) 알고리즘을 사용하여 예측된 FIDT 맵에서 로컬 최대값을 감지하여 얻는다. 이 전략에는 최대 풀링과 적응형 임계값을 적용하여 오탐을 걸러내고 이미지에서 사람의 좌표를 얻는 것이 포함된다.

**정리**

이 논문에서 저자들은 군중 위치 파악을 위한 방법을 제안한다.  논문에서는 회귀분석(머신러닝 모델의 일종)을 사용하여 초점 역거리 변환(FIDT) 맵이라는 예측 맵을 생성한다. 이 맵은 사람이 붐비는 장면에서 겹치지 않고 개인의 위치를 정확하게 묘사한다.

**학습 과정**에서 두 가지 유형의 손실 함수를 사용하여 예측 결과와 실측 데이터 간의 차이를 측정한다. 

1. 평균 제곱 오차(MSE) 손실
2. 이미지 구조 유사성(I-SSIM) 손실

 이러한 손실은 모델이 로컬 구조 정보를 학습하고 FIDT 맵에서 중요한 지점을 더 잘 인식하도록 훈련하는 데 도움이 된다.

**테스트** 중에는 로컬-최대-탐지-전략(LMDS)이라는 전략을 사용한다. 이 전략은 FIDT 맵에서 각 개인의 중심점을 효과적으로 추출하는 데 도움이 된다. 또한 더 나은 시각화를 위해 간단한 KNN 전략을 사용하여 바운딩 박스를 생성할 수 있다.

전반적으로 이 방법은 이전 방법에서 일반적으로 사용되는 밀도 맵 대신 FIDT 맵을 사용하여 군중 위치 파악 정확도를 향상시킨다. 제안된 접근 방식은 군중 밀도가 서로 다른 여러 데이터 세트에서 최첨단 성능을 달성하고 음수 또는 극도로 밀집된 장면에서도 견고함을 보여준다.

 여러 사람이 붐비는 장면이 있다고 가정하고 목표는 장면에서 각 개인의 위치를 정확하게 찾는 것이 가정해보자.

기존 방법에서는 밀도 맵을 사용하여 군중 분포를 표현한다. 이러한 밀도 맵은 사람들이 밀집된 영역을 나타내는 흐릿한 가우시안 블롭으로 구성된다. 그러나 흐릿하고 겹치는 블롭으로 인해 매우 밀집된 장면에서 개인의 정확한 위치를 찾기가 어렵다.

이제 이 논문에서 제안한 초점 역거리 변환(FIDT) 맵 접근 방식을 고려해 보자  밀도 맵을 사용하는 대신 FIDT 맵은 군중 위치 파악을 위해 특별히 설계된 회귀 모델에 의해 생성된다.

FIDT 맵은 기존 밀집도 맵에 비해 밀집 지역이 겹치지 않으면서 개별 위치에 대한 보다 정확한 정보를 제공한다. 이는 일반적인 밀집도 맵에서처럼 흐릿한 얼룩으로 퍼져나가는 것이 아니라 각 사람의 위치에 해당하는 정확한 지점에 더 높은 값을 할당함으로써 달성된다.

학습 과정에서 예측 결과와 실제 결과의 차이를 측정하기 위해 실측 데이터와 함께 평균제곱오차(MSE) 손실과 이미지 구조 유사성(I-SSIM) 손실이라는 두 가지 손실 함수가 활용된다. 이를 통해 FIDT 맵에 존재하는 로컬 구조 정보에 대해 모델을 효과적으로 훈련할 수 있다.

테스트 또는 추론 단계에서는 다른 방법으로는 어려움을 겪을 수 있는 고밀도 군중이나 부정적인 장면에서도 개인의 위치를 정확하게 나타내는 중심점을 추출하는 로컬 최대값 감지 전략(LMDS)이 예측된 FIDT 맵에 사용된다.

테스트 단계에서 이러한 로컬라이즈된 개인을 더 잘 시각화하기 위해 LMDS 기법을 통해 얻은 추출된 중심점 주위에 간단한 KNN 전략을 사용하여 경계 상자를 생성할 수 있다.

전반적으로 이 방법은 기존의 블러 기반 밀도 대신 초점 역거리 변환(FITD) 맵을 생성하는 것과 같은 새로운 기법을 사용하여 보다 정확한 군중 측위 결과를 제공하는 동시에 FITD 맵에서 정확한 위치 중심을 추출하는 LMDS와 같은 효과적인 전략을 통합함으로써 이전 접근 방식을 개선한다.

<img width="742" alt="3" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/ce732cf1-b280-48c9-84f9-471936587008">

그림 3 FIDT 시각화
출처 : https://arxiv.org/abs/2102.07925