# 인구혼잡도 시각화 with Firebase


0. 기본 라이브러리 설치 
```
pip install -r requirements.txt
```

```bash
#-- requirements.txt
python==3.8 
pytorch==2.0.1
opencv-python==4.8.0.76
scipy==1.10.1
h5py==3.9.0
Pillow==9.4.0
imageio==2.31.1
nni==2.10.1
yacs==0.1.8
Pyrebase4==4.7.1
firebase-admin==5.3.0
```
1. git clone 명령어를 통해 해당 repository 다운로드
```
git clone https://github.com/junyong1111/CrowdLocalization.git
```
- 위 명령어를 통해 깃 클론을 받고 FIDTM/Projcet로 경로 이동

2. 아래 명령어를 통해 pre-trained model 다운로드 
- gdown 명령어가 없을 경우 설치 후 진행
```
gdown https://drive.google.com/uc?id=1TBZXWB00mqkZnKzRvWDzR35kgKuW7nP_
unzip /content/bestmodel.zip
```

3. 파이어베이스 연동을 위한 서비스키를 다운로드 같은 경로에 삽입

```
myfirebaseservicyKey
```
4. 다음 명령어를 통해 Crod Counting 시작
```
python people_counting.py --pre ./model_best_nwpu.pth  --video_path "IP카메라 URL"
```

![counting2](https://github.com/Winter-Toy-Project/Honjab-Obseoye/assets/79856225/37d96356-8ba5-4d8d-9e01-a7c89333e3c3)

![fidt2](https://github.com/Winter-Toy-Project/Honjab-Obseoye/assets/79856225/d7e48a26-c13b-4245-b954-2c0eed249354)

#-- 참고 링크 : https://github.com/dk-liang/FIDTM.git 
</div>

## 🔎 파일 구조

```
...
|-- images
|-- Networks
    └── HR_NEt/...
|-- config.py
|-- dataset.py
|-- image.py
|-- myfirebaseservicyKey
|-- LICENSE
|-- model_best_nwpu.pth 
|-- people_counting.py
|-- README.md
|-- requirements.txt
|-- utils.py

...
```

