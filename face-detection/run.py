import cv2
import numpy as np
import insightface
from datetime import datetime
from insightface.app import FaceAnalysis
from get_image import get_image as ins_get_image

# # # Method-1, use FaceAnalysis
app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

# Method-2, load model directly
handler = insightface.model_zoo.get_model('/home/james/.insightface/models/ms1mv3_arcface_r50_fp16/ms1mv3_arcface_r50_fp16.onnx')
handler.prepare(ctx_id=0, input_size=(640, 640))

sample_feats = []
# Prepare 
for i in range(1, 26):
    img = ins_get_image(f'img{i}')
    faces = app.get(img)

    cnt = 0
    face_images = []
    for face in faces:
        bbox = face['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        
        face_img = img[y1:y2,x1:x2]
        face_images.append(face_img)

        start_time = datetime.now()
        feats = handler.get_feat(face_images)
        print(f'extract feature takes {datetime.now() - start_time}')
        sample_feats.append(feats[0])
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite(f"./output{i}.jpg", rimg)

# start_time = datetime.now()
# img = ins_get_image(f'img25')
# feats = handler.get_feat([img])
this_feat = sample_feats[0]
for t_feat in sample_feats:
    start_time = datetime.now()
    sim = handler.compute_sim(this_feat, t_feat)
    # print(sim)start_time = datetime.now()
    print(f'compare feature => {datetime.now() - start_time}')
    

