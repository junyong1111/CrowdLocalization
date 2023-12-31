# CrowdLocalization
Learning Independent Instance Maps for Crowd Localization

## LIIM for Crowd Localization
[Learning Independent Instance Maps for Crowd Localization](https://arxiv.org/abs/2012.04164)

LIIM 

군중 분석에는 군중이 밀집된 장면에서 각 사람의 머리 위치를 정확하게 찾는 것은 중요하다. 기존 방법으로는 특히 매우 밀집되어 있고 다양한 규모의 군중에서 정확한 위치를 예측하는 데 한계가 있다. 이 연구에서 연구진은 군중을 겹치지 않는 구성 요소로 분할하여 보다 정확한 위치 파악을 가능하게 하는 **Independent Instance Map segment (IIM)이라는 프레임워크를 제안한다**. 또한 다양한 밀도 영역에서 세분화 품질을 개선하기 위해 차별화 가능한 바이너리제이션 모듈(BM)을 도입한다. 실험 결과는 제안된 방법이 기존 접근법을 능가하고 F1 측정값을 크게 개선하는 등 효과적임을 입증한다.

### 1. INTRODUCTION

 **INTRODUCTION** 섹션에서는 Crowd 분석에서 Crowd localization의 중요성과 Crowd counting에 비해 더 정확한 결과를 제공할 수 있는 잠재력에 대해 설명한다. 군중 위치 파악을 위한 세 가지 유형의 방법, 즉 탐지 기반 방법, 휴리스틱 알고리즘, 포인트 감독 방법에 대해 언급한다. 저자는 군중을 겹치지 않는 연결된 구성 요소로 분할하여 밀집된 군중 장면에서 개인의 위치와 수를 구하는 **Independent Instance Map sement(IIM)**이라는 프레임워크를 제안하여 기존 방법을 개선한다.

 Crowd localization은 군중 분석에서 군중 내 각 개인의 위치를 예측하는 것을 목표로 하는 작업이다. 이미지 수준의 군중 수 계산에 비해 Crowd localization 파악은 더 높은 수준의 군중 분석 작업에 더 정확한 결과를 제공합니다. 현재 Crowd localization에는 객체 감지 모델을 사용하는 **감지 기반 방법**, 밀도 또는 세분화 맵을 분석하는 **휴리스틱 알고리즘**, 포인트 레벨 라벨 또는 의사 상자 라벨을 활용하는 **포인트 감독 방법** 등 세 가지 주요 접근 방식이 있습니다.

- **감지 기반 방법**
    - 객체 감지 모델을 사용하여 머리의 위치를 예측
    - 극도로 혼잡한 장면에서는 어려움이 존재
- **휴리스틱 알고리즘**
    - 밀도 또는 세분화 맵을 기반으로 사람의 위치 예측
    - 예측이 여러 번 이루어지거나 위치가 부정확한 경우가 많음
    - 작고 흐릿하며 밀도가 높은 물체를 분할하는 데 어려움 존재
- **포인트 감독 방법**
    - 포인트 레벨 레이블을 활용하거나 의사 상자 레이블을 생성하여 군중 장면에서 머리의 위치를 학습
    - 머리 크기를 정확하게 반영하고 큰 범위의 변화를 처리하는 데 한계 존재

<img width="351" alt="1" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/8485ff5e-3588-4902-af3a-fc28b0ae4a60">


그림 1. 군중 위치 측정을 위한 네 가지 유형의 실측 레이블과 제안된 독립 인스턴스 맵을 비교  
출처 :  https://arxiv.org/abs/2012.04164

이 논문에서는 군중 장면의 각 머리가 독립적인 연결된 컴포넌트에 해당하고 그 크기가 머리 영역을 반영하는 연결된 컴포넌트 분할을 활용하는 휴리스틱 접근법을 제안한다. '독립 인스턴스 맵(IIM)'이라고 불리는 결과 맵은 밀집된 군중에서 헤드 위치 파악의 정확도를 향상시킨다. 이 방법은 신뢰도 맵을 이진 분할 맵으로 변환하는 임계값을 학습하는 차등 이진화 레이어를 도입한다. 이를 통해 작은 물체를 세분화하는 정확도를 개선하고 더 나은 로컬라이제이션 결과를 얻을 수 있다.

군중 사진이 있고 이미지에서 각 사람의 머리 위치를 찾는 것이 과제라고 가정해 보자 이 작업을 위한 기존의 방법으로는 대략적인 추정이나 예측만 할 수 있다.

그러나 저자들은 서로 다른 객체(이 경우 머리 크기가 다른 사람들)에 따라 네트워크의 신뢰도가 달라진다는 사실을 발견했다. 예를 들어, 머리가 큰 사람은 중간 크기의 사람에 비해 낮은 신뢰도로 예측될 수 있다.

이 문제를 해결하고 모든 크기의 머리에 대해 더 정확한 위치 파악 결과를 캡처하기 위해 포인트 투 포인트 임계값이라는 개념을 도입했다. 이러한 임계값은 픽셀 수준 바이너리제이션 모듈(PBM)이라는 모듈에 의해 결정된다.

이미지에 픽셀 값을 기준으로 여러 개의 카테고리가 있다고 생각하면 됩니다. 하나의 카테고레는 큰 크기의 헤드를, 다른 카테고리는 작은 크기의 헤드를 위한 것이다. PBM은 이러한 헤드 크기의 공간 분포를 고려하여 각 픽셀 이진화 프로세스에 대해 서로 다른 임계값을 결정한다.

연구팀은 이러한 적응형 임계값과 PBM을 통해 군중을 독립적으로 연결된 구성 요소로 분리하는 **독립 인스턴스 맵 분할(IIM)** 방법을 사용하여 학습 데이터에서 점만 레이블로 사용하는 경우에도 더 나은 객체 로컬라이제이션 정확도를 달성할 수 있었습니다.

요약하자면, 이들은 군중을 별도의 구성 요소로 분리하는 IIM 방법과 함께 PBM에서 제공하는 적응형 임계값 기법을 사용하여 다양한 규모의 객체에 대한 다양한 신뢰도를 고려함으로써 군중 측위 정확도를 개선했다.

### 2 RELATED WORK

 **RELATED WORK** 섹션에서는 Crowd Counting 및 Crowd Localization에 대한 이전 연구에 대해 설명한다. 여기에는 특징 추출을 위한 CNN 기반 방법, 밀도 맵 감독, 포인트 레벨 레이블, 감지 기반 기법 등 다양한 접근 방식이 언급되어 있다. 제안된 방법인 **독립 인스턴스 맵 분할(IIM)**은 Crowd Localization를 위한 간단하고 효과적인 접근 방식으로 소개되며, 도트 라벨링을 사용하더라도 기존 모델보다 성능이 뛰어난다.

**2.1 Crowd Counting**

 Crowd Counting는 **군중 속의 사람 수를 정확하게 파악하는 분야이다.**  군중 장면에서 각 사람의 정확한 머리 위치를 파악하는 Crowd Localization과 밀접한 관련이 있다. 최근 몇 년 동안 합성곱 신경망(CNN)을 사용하는 딥러닝 기술은 기존의 수작업 방식에 비해 이미지에서 유용한 특징을 추출하는 데 큰 가능성을 보여주었다. 

 일부 CNN 기반 접근 방식은 픽셀 수준 또는 패치 수준 해상도에서 밀도 맵을 추정할 수 있는 네트워크 아키텍처 또는 특정 모듈을 설계하는 데 중점을 둔다. 이러한 밀도 맵은 이미지의 여러 영역이 얼마나 혼잡한지에 대한 정보를 제공한다. 다른 방법에서는 이미지 내에서 개별 헤드의 존재와 위치를 나타내는 포인트 레벨 레이블을 사용하여 카운팅 모델을 감독한다. **그러나** 이러한 **기존 방법은** 전체 이미지에 대한 전체 개수만 예측하거나 이미지 내의 국소 밀도에 대한 대략적인 추정치만 제공하는 경우가 많다. **따라서 혼잡한 장면에서 각 개별 머리의 정확한 위치를 정확하게 파악하기 어렵다.**

**2.2 Crowd Localization**

Crowd Localization란 사람이 많은 장면에서 **사람들의 머리 위치를 찾는 것**이다. 연구자들이 사용한 세 가지 주요 접근 방식이 있다

1. 감지 기반 방법
- 바운딩 박스를 사용하여 머리 위치를 찾는다. 이러한 방법은 수작업으로 만든 특징에 의해 제한되거나 더 나은 성능을 위해 CNN을 활용할 수 있다. 그러나 일반적인 객체 감지기는 밀집된 군중에서 어려움을 겪을 수 있으므로 연구자들은 작은 객체를 감지하고 혼잡한 장면에서 성능을 개선하기 위한 특정 네트워크를 개발했다.
1. 휴리스틱 방법
    - 고해상도 밀도 및 세분화 맵을 활용하여 혼잡한 장면에서 헤드 센터를 찾는다. Idrees 등이 제안한 방법, Liu 등이 제안한 방법, Gao 등이 제안한 방법, Xu 등이 제안한 방법, Liang 등이 제안한 방법, Abousamra 등이 제안한 방법 등 다양한 후처리 기법을 사용하여 피크 포인트를 식별하고 가우시안 전제, 회귀 거리 맵, 헤드 영역 분할을 사용하여 중심을 복구할 수 있다. 그러나 이러한 알고리즘 중 일부는 단일 대규모 헤드에 대해 여러 예측을 생성할 수 있으며, Abousamra 등은 정확한 측위를 위해 지속성 손실을 도입하여 토폴로지 제약 조건을 적용한다.
2. 포인트 감독 방법
- 박스 레벨 주석 대신 포인트 레이블을 사용하여 혼잡한 장면에서 머리를 정확하게 찾기 위해 모델을 감독하는 방법이다. 여러 손실 함수를 결합하여 개별 객체를 분할하고, 포인트 주석에서 의사 상자 수준 레이블을 생성하고, 위치와 크기를 동시에 예측하고, 자체 학습 전략을 사용하여 객체의 중심과 크기를 추정하는 등 다양한 접근 방식이 제안되었다. 이러한 기술은 박스 수준 주석의 부족함을 극복하고 군중 위치 파악의 정확도를 향상시키는 것을 목표로 한다.

군중 분석 분야에는 **Crowd Counting**과 **Crowd Localization**이라는 두 가지 주요 작업이 있다. **Crowd Counting**는 주어진 이미지 또는 비디오에서 총 인원 수를 추정하는 것을 목표로 하며, **Crowd Localization**은 사람이 많은 장면 내에서 각 사람의 위치를 정확하게 찾는 데 중점을 둔다. 

 수천 명의 관객이 무대 근처에 밀집해 있는 음악 페스티벌에서 항공사진을 촬영했다고 가정하면. **Crowd Counting**은 정확한 위치를 알지 못한 채 이 사진만 보고 전체적으로 몇 명이 참석했는지 추정하는 반면, **Crowd Localization**은 이미지에서 각 사람의 정확한 머리 위치를 파악하는 것을 목표로 한다.

- **Crowd Counting**는 이벤트 참석자 수를 추정하거나 공공장소의 인구 밀도를 모니터링하는 등 총 참석자 수를 파악하는 것이 중요한 시나리오에서 중요하다. 이는 리소스 할당, 이벤트 계획, 군중 관리에 유용한 정보를 제공한다.
- **Crowd Localization**는 혼잡한 현장 내에서 각 개인의 위치를 정확하게 찾는 데 중점을 둔다. 이 작업은 특정 개인을 식별하거나 움직임을 추적하는 것이 필수적인 감시 시스템이나 보안 모니터링과 같은 애플리케이션에서 특히 유용할 수 있다.

두 작업 모두 중요하지만, 오클루전(사람들이 서로를 가리는 경우), 스케일 변화(카메라로부터의 거리에 따라 사람들의 크기가 다르게 보이는 경우), 개인이 겹치는 밀집된 군중 등의 요인으로 인해 정확한 **Crowd Localization이** 단순 계산보다 더 어려울 수 있다.

요약하면, 두 작업 모두 애플리케이션의 상황과 요구 사항에 따라 고유한 중요성이 있지만, 정확한 **Crowd Localization**은 **Crowd Counting**으로 제공하는 전체 인원 수 추정과 비교하여 군중이 많은 장면 내에서 개별 행동에 대한 더 자세한 인사이트를 제공할 수 있다.

### 3 METHODOLOGY

**METHODOLOGY**  섹션에서는 연구 논문에서 사용된 접근 방식에 대해 설명한다. 군중 지역의 신뢰도를 예측하고, 신뢰도 맵을 이진 맵으로 변환하는 이진화 모듈을 적용하고, 세분화 맵에서 연결된 구성 요소를 감지하여 헤드 센터를 식별하는 등 군중 측위를 위한 독립 인스턴스 맵 세분화(IIM) 프레임워크가 소개된다. 이 프레임워크는 밀집된 장면에서 군중 로컬라이제이션의 정확도를 향상시키는 것을 목표로 한다.

<img width="736" alt="2" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/033dc6ea-a812-4d3c-832d-6be8b4d88e9b">

그림2. 독립 인스턴스 맵 세분화(IIM) 프레임워크  
출처 :  https://arxiv.org/abs/2012.04164

그림 2는 독립 인스턴스 맵 세분화(IIM)의 Crowd Localization 프레임워크를 보여준다. 이 프레임워크는 군중 장면에서 헤드 영역의 신뢰도 수준을 예측한 다음 이진화 모듈을 사용하여 신뢰도 맵을 독립 인스턴스 맵으로 변환한다. 추론하는 동안 임계값 모듈은 각 신뢰도 맵에 대한 임계값을 예측하고, 프레임워크는 4개의 연결된 구성 요소를 감지하여 각 독립 인스턴스 영역에 대한 상자 및 중심 좌표를 얻는다.

**예시** 

많은 사람들이 붐비는 거리의 사진이 있고  목표는 이미지에서 각 사람의 머리 위치를 정확하게 찾는 것이라고 가정해보자. 독립 인스턴스 맵 세그멘테이션(IIM)는  Crowd Localization  프레임워크는 다음과 같이 작동한다.

1. 먼저 이미지에서 사람의 머리가 포함될 가능성이 있는 여러 영역에 대한 신뢰 수준을 예측한다. 이러한 신뢰 수준은 모델이 머리가 포함된 각 영역에 대해 얼마나 확신하는지를 나타낸다.

2. 그런 다음 연속 값을 이진 값(0 또는 1)으로 변환하는 이진화 모듈을 사용하여 신뢰도 맵을 각 사람의 머리가 겹치지 않는 별도의 영역을 갖는 독립적인 인스턴스 맵으로 변환한다.

3. 추론 또는 예측 시간 동안 임계값 모듈은 특정 이미지의 특정 특성에 따라 각 신뢰도 맵에 대한 임계값을 동적으로 예측한다. 즉, 모든 이미지에 고정된 임계값을 사용하는 대신 모델이 각 개별 이미지에 맞게 임계값 전략을 조정한다. 이렇게 하면 다른 이미지에서 머리가 있는 영역과 머리가 없는 영역을 더 잘 구분할 수 있다.

4. 마지막으로, 이 독립적인 인스턴스 맵 내에서 연결된 구성 요소(픽셀이 서로 연결된 영역)를 감지하여 개별 인스턴스 또는 사람의 머리를 나타내는 영역 주변의 상자와 해당 영역 내의 중심 좌표를 식별하고 얻을 수 있다. 연결된 구성 요소 분석에서는 픽셀이 바로 이웃에 있는 다른 픽셀과 가장자리나 모서리를 공유하면 "연결된" 것으로 간주한다. 특히 IIM에서는 픽셀이 상/하/좌/우 네 방향으로 연결된 영역을 찾으며, 이를 "4연결 구성 요소"라고 한다. IIM에서 생성된 독립 인스턴스 맵 내에서 이러한 연결된 구성 요소를 감지하면 어떤 픽셀이 어떤 사람의 머리 영역에 속하는지 확인할 수 있다. 이를 통해 사람의 머리를 나타내는 각 개별 인스턴스/영역 주변의 바운딩 박스와 해당 영역 내의 중심 좌표를 얻을 수 있다.

따라서 IIM은 기본적으로 군중이 있는 입력 이미지를 가져와 개별 인스턴스를 나타내는 별도의 영역과 해당 영역 내의 위치를 출력한다.

<img width="349" alt="3" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/9e22dcf6-b6b3-465c-a4e9-1226d9ebd978">

그림3. 이진화 레이어 워크플로우  
출처 :  https://arxiv.org/abs/2012.04164

그림 3은 특정 비전 작업에서 정밀한 분할에 사용되는 구성 요소인 이진화 레이어의 워크플로우를 보여준다. 이 레이어는 학습 가능한 임계값을 사용하여 연속 히트 맵을 이진 맵으로 변환한다. 이를 통해 특정 임계값에 따라 영역을 분리할 수 있어 보다 정확한 분할이 가능하다.

**예시**

군중 장면의 이미지가 있고 이미지에서 각 사람의 위치를 정확하게 찾고 싶다고 가정해 보자 이진화 이진화 레이어는 **군중 속에서 사람들을 더 선명하게 볼 수 있도록 도와주는 특수 안경을 착용하는 것**과 같다. 사람들이 어디에 있는지 알 수 있는 모델에서 정보를 가져와 단순화된 버전으로 변환한다.

이진화 레이어는 회색 음영이나 각 픽셀마다 다른 값을 사용하는 대신 검정과 흰색의 두 가지 색상만 사용하여 작업을 단순화한다. 임계값(예: 0.5)을 설정하여 '사람'과 '사람이 아닌 것'을 구분하는 선 역할을 한다. 0.5보다 큰 픽셀은 사람의 일부(흰색)로 간주되고, 0.5보다 작거나 같은 픽셀은 빈 공간(검은색)으로 간주된다. 따라서 이 이진화 프로세스를 원본 이미지에 적용하면 사람이 있는 영역은 모두 흰색으로 채워지고 그 외의 모든 영역은 검은색으로 표시되는 윤곽선 그림과 같은 결과가 나타난다.

이렇게 하면 사람이 있는지 여부에 따라 영역을 구분할 수 있으므로 군중 장면에서 각 사람의 위치를 정확하게 파악하기가 더 쉬워진다.

**3.1 Binarization Layer**

3.1.1 Problem setting

학습 가능한 임계값을 사용하여 출력 이미지에서 어떤 픽셀이 0이고 어떤 픽셀이 1이어야 하는지 결정하는 이진화 계층을 도입한다. 목표는 각 픽셀이 0 또는 1의 값만 가질 수 있는 대상 이진 이미지와 가능한 한 유사한 출력 이미지를 만드는 것이다. 이진화 레이어는 각 개별 이미지의 임계값을 조정하여 다양한 입력 이미지에 대해 최적의 분할을 달성하는 것을 목표로 한다.

3.1.2 Theoretical Analysis

이진화 레이어는 목표 이진 이미지를 기반으로 입력 이미지를 전경과 배경으로 분리하는 임계값을 학습한다. 군중 장면의 입력 이미지가 있고, 군중 속의 각 사람을 전경으로 분류하고 나머지는 모두 배경으로 유지하는 것이 목표라고 가정해 보자

이진화 레이어는 이 입력 이미지를 가져와 각 사람을 장면의 나머지 부분과 구분하는 임계값을 찾는 방법을 학습한다. 입력 이미지의 픽셀 값과 대상 이진 이미지의 해당 픽셀 값(어떤 픽셀이 사람에 속하는지를 나타내는)을 비교하여 이를 수행한다.

조명이나 픽셀 값에 영향을 미치는 기타 요인에 변화가 있는 상황을 처리하기 위해 릴랙세이션과 선형 연산을 도입한다. 릴레이션을 사용하면 인접 픽셀의 정보를 고려하여 정확한 임계값을 찾는 데 어느 정도 유연성을 확보할 수 있다.

예를 들어, 대상 바이너리 맵에 따라 강도가 0.8인 사람(전경)에 속하는 픽셀과 강도가 0.2인 사람이 아닌 영역(배경)에 속하는 픽셀이 인접한 두 개의 픽셀이 있다면 "강도가 0.5를 초과하면 전경에 속한다"와 같은 엄격한 임계값을 사용하면 이 두 픽셀이 서로 가까이 있어도 다르게 분류된다.

그러나 완화 기법을 사용하면 인접한 두 픽셀의 강도가 ±0.x 차이 범위 이내처럼 충분히 가까운 경우 상황에 따라 두 픽셀을 모두 어느 클래스의 일부로 간주할 수 있도록 유연성을 확보할 수 있다.

선형 연산을 사용하면 역전파 과정에서 그라데이션이 문제 없이 이 비차별화 단계를 원활하게 통과할 수 있다.

**3.2 Binarization Module**

<img width="357" alt="4" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/e5428d3e-8476-40d7-98ec-a688e8d5a762">

그림4. 바이너리제이션 모듈을 네트워크에 삽입하는 순서도  
출처 : https://arxiv.org/abs/2012.04164

그림 4는  일반적인 바이너리화 모듈(BM)의 구조이다. 이 모듈은 일반적으로 컨볼루션 신경망에서 얻은 특징 맵 F를 기반으로 신뢰도 맵 I에 대한 임계값 예측을 생성한다. BM은 이러한 구성 요소를 결합하여 이진화 프로세스를 수행하여 연속 신뢰도 맵을 이진 인스턴스 맵으로 변환한다.

바이너리화 모듈(BM)은 연속 히트 맵(다른 영역에서 머리의 신뢰도 또는 강도를 나타내는)을 이진 맵(각 픽셀이 배경의 경우 0, 물체의 경우 1)으로 변환하여 이 작업을 수행하도록 도와준다.

사람들이 걸어 다니는 붐비는 거리의 이미지가 있다고 생각하자 해당 Task는 이미지에서 각 사람의 머리 위치를 찾는 것이다.

이를 위해 BM은 크게 두 부분으로 구성된다.

1. 임계값 인코더: 이 부분은 이미지에서 특징을 가져와 "임계값"이라는 값을 생성한다. 무언가를 머리로 간주하기 전에 어느 정도 확신을 가져야 하는지 결정한다고 생각하면 된다.
2. 이진화 레이어: 이 부분에서는 임계값을 사용하여 바이너리 맵에서 머리와 배경을 구분한다. 임계값 이상의 높은 신뢰도를 가진 픽셀을 1(헤드)로, 그 이하의 픽셀을 0(배경)으로 할당한다.

**신뢰도 예측자**

- 이 부분은 이미지의 여러 영역을 분석하여 머리일 가능성이 높은 영역에 더 높은 값/신뢰 점수를 할당한다.
- 예시: 혼잡한 거리 사진에는 걸어 다니는 사람들이 있을 수 있고, 일부는 모자나 다른 물건을 머리에 쓰고 있을 수 있다. 신뢰도 예측기는 실제 사람의 머리가 있을 것으로 예상되는 영역에 높은 값/신뢰도 점수를 할당한다.

**임계값 인코더**

- 이 부분은 물체/머리가 있는지 여부에 따라 임계값을 설정한다.
- 예시: 대부분의 사람의 머리가 작은 크기/작은 크기/큰 크기의 물체로 나타난다는 것을 알고 있다면, 임계값을 낮게 설정하여 크기가 작고 크기가 다른 머리도 정확하게 감지할 수 있도록 해야한다.

이제 이 두 가지 작업이 함께 작동하는 방식은 다음과 같다

- 신뢰도 예측기를 통해 필요한 경우(예: 잠재적인 헤드 위치가 있는 영역)에만 더 높은 신뢰도를 할당한다.
- 그리고 임계값 인코더를 사용하여 임계값을 적절히 조정(물체/머리 영역의 경우 임계값을 낮게 설정)한다.

나무나 건물과 같은 배경 노이즈가 '머리'로 간주되지 않도록 효과적으로 필터링하는 동시에 이미지에서 실제 사람의 머리 위치만 정확하게 찾아내는 데 집중할 수 있다.

궁극적으로 목표는 점수 매기기(신뢰도 예측)를 통해 **잠재적인 머리 위치를 확실하게 식별하는 것과 실제 머리 영역으로 인정되는 것에 대한 적절한 기준 설정(임계값 인코딩) 사이에서 효과적인 균형을 찾는 것**이다.

**3.3 Crowd Localization Framework**

**신뢰도 예측기 Confidence Predictor(CP)**

- 신뢰도 예측기(CP)는 군중 로컬라이제이션 프레임워크의 중요한 구성 요소이다. 이 기능은 VGG-16 + FPN 또는 HRNet-W48과 같은 고해상도 표현 백본을 사용하여 군중 장면에서 머리를 정확하게 감지하고 위치를 파악하는 데 도움을 준다.
- 이를 더 잘 이해하기 위해 붐비는 경기장에서 빨간 모자를 쓴 사람들을 모두 찾으려고 한다고 가정해 보자 . CP는 매우 선명하게 볼 수 있고 각 사람의 머리의 세세한 부분까지 포착할 수 있는 눈과 같은 역할을 한다. 이러한 고해상도 비전은 사람의 크기가 다르거나 다른 물체의 방해물로 인해 발생하는 문제를 제거하는 데 도움이 된다.

이제 CP에서 각 사람의 머리에 대한 세부적인 이미지를 얻었으면 어떤 픽셀이 머리에 속하고 어떤 픽셀이 머리에 속하지 않는지 결정해야 한다. 이때 임계값 인코더 Threshold Encoder(TE)가 중요한 역할을 한다.

**Threshold Encoder(TE)**

TE에는 이미지 레벨 바이너리제이션 모듈(IBM)과 픽셀 레벨 바이너리제이션 모듈(PBM)  두 가지 방식이 있다. 

1. **IBM**  : 모든 예측을 한 번에 이진화하기 위해 하나의 값을 임계값으로 사용한다.
- 예를 들어, 빨간 모자를 감지하기 위한 임계값으로 0.5를 설정하면 0.5를 초과하는 예측은 빨간 모자가 있는 것으로 간주하고 그 이하인 것은 그렇지 않은 것으로 간주한다.
1. **PBM :** 데이터 세트 편향이나 이미지 품질 문제로 인해 특정 샘플을 배경과 구별하기 어렵거나 머리 사이의 스케일 차이와 같은 요인에 따라 신뢰도 분포가 달라진다는 점을 고려한다. 따라서 PBM은 모든 사람에게 하나의 임계값만 사용하는 대신 이미지 내의 지역적 특성에 따라 서로 다른 임계값을 나타내는 픽셀 수준 맵을 생성한다. 
- 예를 들어, 이 예에서 TE는 작은 크기의 헤드가 중간 크기의 헤드에 비해 신뢰도 값이 낮은 경향이 있다는 것을 학습할 수 있다. 따라서 작은 크기의 헤드에는 특별히 낮은 임계값을 할당하고 큰 헤드에는 높은 임계값을 유지함으로써 개별 인스턴스 내에 존재하는 특정 특징에 따라 탐지 전략을 조정한다.

전반적으로 CP와 IBM 또는 PBM을 결합하면 먼저 각 헤드의 세부 이미지를 얻은 다음 적절한 임계값을 사용하여 헤드에 속하는 픽셀을 정확하게 결정할 수 있다. 이를 통해 군중 장면에서 머리를 효과적으로 찾아 세는 데 도움이 된다.

**사람이 많은 방에 있는데 안경을 쓴 사람들을 모두 찾고 싶다고 상상해 보자**

신뢰도 예측기(CP)는 작은 디테일까지 볼 수 있는 아주 예리한 시력을 가진 것과 같다. 안경을 착용했는지 여부를 포함하여 각 사람의 얼굴을 식별하는 데 도움이 된다.

이제 CP가 각 사람의 얼굴에 대한 자세한 이미지를 제공하면 어떤 픽셀이 안경에 속하고 어떤 픽셀이 안경에 속하지 않는지 결정해야 합니다. 이때 임계값 인코더(TE)가 필요하다.

TE에는 픽셀을 이진화하거나 분리하는 두 가지 방법이 있다: 이미지 레벨 바이너리제이션 모듈(IBM)과 픽셀 레벨 바이너리제이션 모듈(PBM)이다.

IBM에서는 모든 픽셀에 대해 한 번에 하나의 임계값을 사용한다. 예를 들어 안경 착용 여부를 감지하는 임계값을 0.5로 설정하면 0.5를 초과하는 예측은 안경을 착용한 사람으로 간주하고 그 미만은 그렇지 않은 것으로 간주한다.

반면, PBM은 안경 착용 여부를 예측할 때 사람마다 신뢰도가 다를 수 있다는 점을 고려하여 하나의 임계값만 사용하는 대신 이미지 내의 지역적 특성에 따라 서로 다른 임계값을 나타내는 픽셀 수준 맵을 생성한다.
예를 들어, 얼굴이 작은 사람은 큰 사람에 비해 신뢰도가 낮은 경향이 있다는 것을 학습하여 얼굴이 작은 사람에게는 낮은 임계값을 할당하고 얼굴이 큰 사람에게는 높은 임계값을 유지함으로써 개별 인스턴스 내에 존재하는 특정 특징에 따라 탐지 전략을 조정할 수 있다.

CP를 IBM 또는 PBM과 결합하여 먼저 각 사람의 얼굴을 보여주는 선명한 이미지를 얻은 다음 TE에서 결정한 적절한 임계값을 적용하여 안경을 착용한 사람을 정확하게 파악할 수 있으므로 혼잡 한 공간에서도 안경을 착용 한 모든 사람을 효과적으로 찾을 수 있다.

### 4. EXPERIMENTS

**EXPERIMENTS** 섹션에서는 제안된 프레임워크의 성능을 평가하는 데 사용된 실험 설정과 평가 메트릭에 대해 설명한다. 평가에 사용된 데이터 세트에는 주석이 달린 인스턴스가 있는 다양한 군중 이미지가 포함된 NWPU-Crowd, Shanghai Tech, UCF-QNRF 및 FDST가 포함된다. 구현 세부 사항에는 훈련 설정, 라벨 생성 방법, 그리고 모델의 성능을 평가하는 데 사용되는 인스턴스 수준 정밀도, 리콜, F1 측정값, 평균 절대 오차(MAE), 평균 제곱 오차(MSE), 정규화된 절대 오차(NAE) 등의 메트릭에 대한 설명이 나와 있다.

**4.1DataSet**

**NWPU-Crowd 데이터 세트**

- 5,000개 이상의 이미지와 2백만 개 이상의 주석이 달린 인스턴스를 포함하는 가장 크고 가장 까다로운 오픈 소스 군중 데이터 세트

**상하이 테크 데이터 세트**

- 각각 482개와 716개의 이미지로 구성된 파트 A와 파트 B의 두 개의 하위 집합으로 구성

 **UCF-QNRF 데이터 세트**

- 1,535개의 이미지와 1,251,642 인스턴스로 구성

**FDST 데이터 세트**

- 13개의 다른 장면에서 캡처한 100개의 비디오가 포함되어 있으며 총 394,081개의 헤드가 포함

<img width="732" alt="5" src="https://github.com/junyong1111/CrowdLocalization/assets/79856225/de77d949-d65a-47de-ae3b-9dbcc3370292">


그림5. 각각의 방법의 시각적 결과  
출처 : https://arxiv.org/abs/2012.04164

그림 5는 널리 사용되는 네 가지 방법과 제안된 IIM이라는 방법의 시각적 결과를 NWPU-Crowd 검증 세트에서 보여준다. 녹색 점은 정탐, 빨간색 점은 오탐, 마젠타색 점은 미탐을 나타낸다. 녹색 및 빨간색 원은 특정 반경의 기준 진실을 나타낸다. 이미지는 더 쉽게 해석할 수 있도록 gray sacale로 표시된다.