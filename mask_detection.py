'''
python -m pip install --upgrade pip
pip install ultralytics
pip install pillow
pip install opencv-python
'''
from flask_restful import Resource
from flask import request,make_response


import base64
from PIL import Image
import cv2
import json
import numpy as np
import io
import os
import numpy as np
from ultralytics import YOLO

class MaskDetection(Resource):
    def __init__(self):
        self.model = YOLO('best.pt')
    def post(self):
        base64Encoded = request.form['base64Encoded']
        image_b64 = base64.b64decode(base64Encoded)
        image_memory = Image.open(io.BytesIO(image_b64))#이미지 파일로 디코딩
        image_memory.save('../images/new.jpg')#물리적으로 이미지 저장
        results = self.model.predict(['../images/new.jpg'],save=True)
        # train()후 prdict()가 아니기때문에 예측 이미지명은 그대로다
        #pred_image = Image.open(os.path.join(results[0].save_dir, 'new.jpg'))
        # 이미지 뷰어 윈도우 프로그램에 띄울때
        #pred_image.show()
        #예측 이미지를 base64로 인코딩
        with open(os.path.join(results[0].save_dir, 'new.jpg'),'rb') as f:
            base64Predicted= base64.b64encode(f.read()).decode('utf-8')
        print('base64Predicted\n',base64Predicted)

        return make_response(json.dumps({'base64':base64Predicted},ensure_ascii=False))
        #return make_response(json.dumps(y_pred_dict,ensure_ascii=False))