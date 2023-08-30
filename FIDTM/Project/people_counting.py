from __future__ import division
#-*- coding:utf-8 -*-

from distutils.command.config import config
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
from pytz import timezone
import firebase_admin
import collections
import pyrebase
import cv2
import time


import warnings

from Networks.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args

warnings.filterwarnings('ignore')
import time
import torch
logger = logging.getLogger('mnist_AutoML')
import os


config = {
    "apiKey": "AIzaSyDCuD25hjKtGylCHi1NtKqPHXjXAcuOEq4",
    "authDomain": "honjab-obseoye.firebaseapp.com",
    "databaseURL":"https://honjab-obseoye-default-rtdb.asia-southeast1.firebasedatabase.app/", ## 데이터베이스 추가
    "projectId": "honjab-obseoye",
    "storageBucket": "honjab-obseoye.appspot.com",
    "messagingSenderId": "1047002174264",
    "appId": "1:1047002174264:web:ca77afb6b73bbbed40a7cd",
    "measurementId": "G-P7KKPKD4G7",
    "serviceAccount": "honjab-obseoye-firebase-adminsdk-gxxrr-73e26176f1.json" ## 다운받은 비공개 키
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
credpath = r"honjab-obseoye-firebase-adminsdk-gxxrr-73e26176f1.json" ##다운받은 serviceAcc 경로
login = credentials.Certificate(credpath)
## firebase app 시작
firebase_admin.initialize_app(login)
## 데이터베이스에서 값 가져오기
db = firestore.client()
## 데이터베이스에 값 쓰기
img_ref = db.collection("users")
#-- 초기 설정
KST = timezone('Asia/Seoul')
Location = "OOO역 OOO입구"
MAX_TIME = 60 #-- 1분마다 갱신
s_factor = 0.9  #-- 채도를 줄이려면 값이 1보다 작은 값을 사용합니다.
#-- 초기 설정
print(args)
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent.parent)
FIDTM_DIR = BASE_DIR + "/FIDTM/"

print(type(FIDTM_DIR))

def main(args):
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cap = cv2.VideoCapture(args['video_path'])


    Number = -1
    Start_Time = time.time()
    ret, frame = cap.read()
    print(frame.shape)

    '''out video'''
    width = frame.shape[1] #output size
    height = frame.shape[0] #output size

    out_ori = cv2.VideoWriter(FIDTM_DIR + '/out_ori.avi', fourcc, 23, (width, height))
    out_show_fidt = cv2.VideoWriter(FIDTM_DIR + '/out_show_fidt.avi', fourcc, 23, (width, height))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda:
      print(torch.cuda.get_device_name(0))
    while True:
        try:
            ret, frame = cap.read()
            scale_factor = 0.5
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            ori_img = frame.copy()
        except:
            print("Video End")
            cap.release()
            break
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)

        with torch.no_grad():
            d6 = model(image)
            count, pred_kpoint = counting(d6)

            cvt_image = show_fidt_func(d6.data.cpu().numpy())
            hsv_image = cv2.cvtColor(cvt_image, cv2.COLOR_RGB2HSV)
            
            hsv_image[:,:,1] = hsv_image[:,:,1] * s_factor
            show_fidt = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            cv2.putText(ori_img, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            End_Time = time.time()
            Time  = int(End_Time - Start_Time)

            if Time == MAX_TIME :
              Time = 0
              Number +=1
              T = datetime.now(KST)
              T = T.strftime('%Y-%m-%d %H:%M:%S')

              ori_filename = 'counting' + str(Number+1)+".jpg"
              cv2.imwrite(FIDTM_DIR + '/images/' + ori_filename, ori_img)
              storage.child(ori_filename).put(FIDTM_DIR + "/images/"+ ori_filename)


              fidt_filename = 'fidt' + str(Number+1)+".jpg"
              cv2.imwrite(FIDTM_DIR + '/images/' + fidt_filename, show_fidt)
              storage.child(fidt_filename).put(FIDTM_DIR+ "/images/"+fidt_filename)


              ori_fileUrl = storage.child(ori_filename).get_url(0) #0은 저장소 위치 1은 다운로드 url 경로
              fidt_fileUrl =  storage.child(fidt_filename).get_url(0) #0은 저장소 위치 1은 다운로드 url 경로

              img_ref.document("crowd").set({
                  'date' : T,
                  "location": Location,
                  "counting": str(count),
                  'ori_fileUrl' : ori_fileUrl,
                  'fidt_fileUrl' : fidt_fileUrl
                  })
          # ESC를 누르면 종료
          #-- q 입력시 종료
              Start_Time = time.time()
              print("SAVE THE IMAGE ")
            # out_ori.write(ori_img)
            # out_show_fidt.write(show_fidt)
        print("pred:%.3f" % count)
    cap.release()
    cv2.destroyAllWindows()


def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1

    '''negative sample'''
    if input_max<0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint


def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)

    return point_map


def generate_bounding_boxes(kpoint, Img_data):
    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048

    if pts.shape[0] > 0: # Check if there is a human presents in the frame
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

        distances, locations = tree.query(pts, k=4)
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1:
                sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.04)

            if sigma < 6:
                t = 2
            else:
                t = 2
            Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                    (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return Img_data


def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)