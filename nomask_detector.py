#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 라이브러리 임포트
import os
import cvlib as cv
import cv2
import numpy as np
import requests
import time
import datetime
from pygame import mixer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 이미지 읽어오기
path_dir1 = './nomask/'
path_dir2 = './mask/'
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
 
file_list1_num = len(file_list1)
file_list2_num = len(file_list2)
 
file_num = file_list1_num + file_list2_num


# In[3]:


# 이미지 전처리
num = 0;
all_img = np.float32(np.zeros((file_num, 224, 224, 3))) 
all_label = np.float64(np.zeros((file_num, 1)))
 
for img_name in file_list1:
    img_path = path_dir1+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 0 # nomask
    num = num + 1
 
for img_name in file_list2:
    img_path = path_dir2+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 1 # mask
    num = num + 1


# In[4]:


# 데이터셋 섞기(적절하게 훈련되게 하기 위함) 
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
all_label = all_label[indices]
all_img = all_img[indices]
 
    
# 훈련셋 테스트셋 분할
num_train = int(np.round(all_label.shape[0]*0.8))
num_test = int(np.round(all_label.shape[0]*0.2))
 
train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :] 
 
train_label = all_label[0:num_train]
test_label = all_label[num_train:]


# In[5]:


# 모델 구축
IMG_SHAPE = (224, 224, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
 
flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)
 
model = Sequential([
        base_model,
        flatten_layer,
        dense_layer1,
        bn_layer1,
        dense_layer2,
        ])
 
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 모델 훈련
start = datetime.datetime.now()
hist = model.fit(train_img, train_label, epochs=10, batch_size=16, validation_data = (test_img, test_label))
end = datetime.datetime.now()
time = end - start
print("학습시간: ", time) 


#모델 저장
model.save("model.h5")
print("Saved model to disk")  


# In[6]:


# 모델 요약 정보 출력
print("Number of layers in the base model: ", len(base_model.layers))
base_model.summary()
model.summary()


# In[7]:


# 모델 시각화
plot_model(model, to_file='model_shapes.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)


# In[11]:


# 학습 결과 그래프
plt.plot(hist.history['accuracy'], label='acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.xlabel("EPOCH")
plt.legend(loc='center left')
plt.show()
plt.savefig("train_history.png")


# In[12]:


# 미착용자 감지될 경우 알림음 재생
mixer.init()
sound = mixer.Sound('maskon.wav')
model = load_model('model.h5')


# In[13]:


# 미착용자 감지될 경우 슬랙 메시지 전송
def send_message(msg):
    url='https://hooks.slack.com/services/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'
    data = {'text':msg}
    resp = requests.post(url=url, json=data)
    return resp


# In[14]:


# 웹캠 실행
webcam = cv2.VideoCapture(0)
alarm = 0
count = 0 
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()


# In[15]:


# 웹캠 프레임당 루프
while webcam.isOpened():
 
    # 웹캠에서 프레임 읽어오기
    status, frame = webcam.read()
    
    if not status:
        print("Could not read frame")
        exit()
 
    # 얼굴 감지
    face, confidence = cv.detect_face(frame)
 
    # 감지된 얼굴에 루프
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            face_region = frame[startY:endY, startX:endX]
            
            face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
            
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            prediction = model.predict(x)
 
            if prediction < 0.5: # 마스크 미착용으로 판별되면,
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                alarm += 1
                count += 1
                if alarm == 15:  # 알림음 재생
                    sound.play()
                    alarm = 0
                if count == 50:  # 메시지 전송
                    send_message('마스크 미착용자 감지')
                    count = 0

                
            else: # 마스크 착용으로 판별되면
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0]*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                alarm = 0
                count = 0
    
    # 결과 출력
    cv2.imshow("mask nomask classify", frame)
 
    # Q키 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
mixer.stop()


# In[ ]:




