import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import cv2
import numpy as np
from playsound import playsound
import webbrowser

# load the trained model.
my_model = keras.models.load_model(r"C:\Users\user\AppData\Local\Programs\Python\Python38\ferNet2_acc_6328_try2.h5")
my_model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
my_model.summary()

labels_dict = {0:'ANGRY', 1:'HAPPY', 2:'NEUTRAL', 3:'SAD'}
face_classifier = cv2.CascadeClassifier(r"C:\Users\user\AppData\Local\Programs\Python\Python38\haarcascade_frontalface_default.xml")
#baymax_hi = cv2.imread(r"C:\Users\user\Pictures\baymax_pic.jpg")
baymax_voice = r"C:\Users\user\Downloads\baymax_intro.mp3"
happy_voice = r"C:\Users\user\Downloads\happy_baymax.mp3"
n = 0
while(n < 1):
    playsound(baymax_voice)
    source = cv2.VideoCapture(0)
    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.5, 5)
    num = len(faces)
    print(num)
    for (x,y,w,h) in faces:
        face_img = gray[y:y+w,x:x+w]
        resized = cv2.resize(face_img,(48,48))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,48,48,1))
        result = my_model.predict(reshaped)
        print(result)
        label = np.argmax(result,axis=1)[0]
        print(label)
        if(label == 1):
            playsound(happy_voice)
        elif(label == 2):
            webbrowser.open(r'https://www.google.com/logos/2010/pacman10-i.html', new=2)
        elif (label == 3):
            webbrowser.open(r'https://mentalhealth.com/home/', new=2)
        else:
            webbrowser.open(r'https://images.app.goo.gl/9PLfGab6vBwNtncEA', new = 2)
    key=cv2.waitKey(0)
    n += 1

cv2.destroyAllWindows()
source.release()
