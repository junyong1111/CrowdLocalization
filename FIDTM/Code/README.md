# Code [Focal Inverse Distance Transform Map]

### **FIDTM**

[GitHub - dk-liang/FIDTM: Focal Inverse Distance Transform Maps for Crowd Localization [IEEE TMM]](https://github.com/dk-liang/FIDTM)

**FIDTM을 이용하여 비디오에서 혼잡도를 계산하는 실습 진행**

- 해당 코드는 Colab에서 진행되었음

### STEP.0 코드 실행에 필요한 미리 학습된 모델과 동영상 다운로드

1.  직접 구글 드라이브에서 다운로드
- 베스트 모델 압축 파일
-  https://drive.google.com/file/d/1TBZXWB00mqkZnKzRvWDzR35kgKuW7nP_/view?usp=sharing
- 예제 동영상 파일
  -  https://drive.google.com/file/d/1ziRoC78jn5b3CWhoqB1zpA4x7gzCi7dK/view?usp=sharing
  - https://drive.google.com/file/d/1vFmYIKmdiuRHgiGNcHLZjbQguA7o86xC/view?usp=sharing

2. 아래 코드를 이용하여 다운로드(gdown 라이브러리 필요) => 코랩은 기본적으로 gdown 라이브러리 존재

```bash
#-- 베스트 모델 파일 구글 드라이브 파일 다운로드
!gdown https://drive.google.com/uc?id=1TBZXWB00mqkZnKzRvWDzR35kgKuW7nP_
!unzip /content/bestmodel.zip
#-- 예제 동영상 파일 구글 드라이브 파일 다운로드
!gdown https://drive.google.com/uc?id=1ziRoC78jn5b3CWhoqB1zpA4x7gzCi7dK
!gdown https://drive.google.com/uc?id=1vFmYIKmdiuRHgiGNcHLZjbQguA7o86xC
```

### Step.1 깃 클론 이후 필요 라이브러리 install

**Environment**

```bash
python >=3.6   
pytorch >=1.4  
opencv-python >=4.0  
scipy >=1.4.0  
h5py >=2.10   
pillow >=7.0.0   
imageio >=1.18  
nni >=2.0 (python3 -m pip install --upgrade nni)
```

- nni를 제외하고 나머지는 이미 있음
- 추가적으로 코드 실행을 위해 yacs 라이브러리 필요

```bash
!git clone https://github.com/dk-liang/FIDTM.git
```

```
!pip install nni
!pip install yacs
```

### Step2. 샘플 데이터를 통해 예제 확인

- --pre 다운받은 베스트 모델 중 하나 선택
- --video_path 다운받은 영상 또는 자신의 영상 경로

```bash
%cd /content/FIDTM
!python!python video_demo.py --pre /content/bestmodel/nwpu/model_best_nwpu.pth  --video_path /content/subway_test.mp4
```

### Step3. 결과 동영상 확인

- 코드에 맞게 실행 시 저장된 동영상 파일은 /content/FIDTM/demo.avi에 저장되어 있으며 해당 파일을 다운로드 하여 확인할 수 있음
- 아래 코드를 통해 코랩에서 바로 확인 가능

```python
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/FIDTM/demo.avi'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

https://github.com/junyong1111/CrowdLocalization/assets/79856225/84eb38ac-6275-46ce-86f7-c8b729805377