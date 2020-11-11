# Detecção Facial

import cv2

import os

os.chdir('C:/OficinaTechDay/') 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame): 

    #Detecta as faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    #Para cada face detectada 
    for (x, y, w, h) in faces: 

        #Desenha um Retangulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 

        #Recorta a imagem delimitando uma nova área   
        roi_gray = gray[y:y+h, x:x+w] 

        roi_color = frame[y:y+h, x:x+w] 

        #Detecta os olhos
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) 
        
        #Para cada olho detectado desenha um retangulo
        for (ex, ey, ew, eh) in eyes: 
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) 

        #Detecta o sorriso
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 22) 
        
        #desenha um retangulo para cada sorriso 
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 


    return frame 

video_capture = cv2.VideoCapture(0) 

#Cria um loop infinito para ler continuamente os frames
while True:

    #Lê o frame apartir do dispositivo de entrada. No caso como não existe fonte especificada, 
    #lê direto da webcam.
    _, frame = video_capture.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    canvas = detect(gray, frame) 

    cv2.imshow('Video', canvas) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
       break 

video_capture.release()

cv2.destroyAllWindows() 
