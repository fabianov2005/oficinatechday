# -*- coding: cp1252 -*-
import numpy as np
import cv2
import os

os.chdir('C:/OficinaTechDay/') 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def criaArquivoDeRotulo():
    label = 0

    pasta = 'dadosTreino'
        
    if not os.path.exists(pasta):     
        os.mkdir(pasta)
    
    f = open("Treino", "w+")    
    
    for root, dirs, files in os.walk(pasta):
        
        for subDir in dirs:
            
            caminhoPasta = os.path.join(root, subDir)
            
            for filename in os.listdir(caminhoPasta):
                
                caminho = caminhoPasta + "\\" + filename
                
                f.write(caminho + ";" + str(label) + "\n")
                
            label = label + 1
            
    f.close()

def criaListaImagensChaves(arqDados):
    
    #Lê o arquivo de treino
    lines = arqDados.readlines()

    #Inicializa lista de imagens e chaves
    Imagens = []
    Chaves = []
    
    for line in lines:
        
        filename, label = line.rstrip().split(';')

        Imagens.append(cv2.imread(filename, 0))

        Chaves.append(int(label))    
                
    return Imagens, Chaves
    
def treinaModelo(imagens, chaves):
    
    #cria treina as autofaces
    model = cv2.face.FisherFaceRecognizer_create()
    
    model.train(imagens, np.array(chaves))
    
    return model


def reconheceVideo(modelo, arquivo):
    
    cap = cv2.VideoCapture(arquivo) #abre o arquivo para detecção
    
    counterFrames = 0;
    
    while(counterFrames < 1000): #quando chegar ao milésimo frame, para
        ret, img = cap.read()
        

        #frame não pode ser obtido? entao sair
        if(ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #se nenhuma face for achada, continue
        if not np.any(faces):
            continue

        rostos = []
        #achou uma face? recorte ela (crop)
        for (x, y, w, h) in faces:
            
            rosto = img[y:y+h, x:x+w]
            #esse rosto é grande o bastante pra ser levado
            #em conta
            if(((x + w) - x) > 100 and ((y + h) - y) > 100):

                #modifica o tamanho dele pra se ajustar ao
                #treinamento e pinte pra tons de cinza
                rosto = cv2.resize(rosto, (255, 255))
                
                rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)

                #aqui ele recebe a foto e diz qual rótulo
                #pertence (ou seja, quem é)
                label = modelo.predict(rosto)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                if(label[0] == 0): #Anallice
                
                    #então bota um texto em cima da caixinha
                    cv2.putText(img,'Anallice',(x - 20,y + h + 60), font, 3,(255,0,0),5,cv2.LINE_AA)
                    
                    #pinte um retângulo ao redor do rosto
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    
                if(label[0] == 1): #Fabiano
                
                    #então bota um texto em cima da caixinha
                    cv2.putText(img,'Fabiano',(x - 20,y + h + 60), font, 3,(0,0,255),5,cv2.LINE_AA)
                    
                    #pinte um retângulo ao redor do rosto
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    
                if(label[0] == 2): #Gabriel
                
                    #então bota um texto em cima da caixinha
                    cv2.putText(img,'Gabbriel',(x - 20,y + h + 60), font, 3,(0,0,255),5,cv2.LINE_AA)
                    
                    #pinte um retângulo ao redor do rosto

                if(label[0] == 3): #JJuliana
                
                    #então bota um texto em cima da caixinha
                    cv2.putText(img,'Juliana',(x - 20,y + h + 60), font, 3,(0,0,255),5,cv2.LINE_AA)
                    
                    #pinte um retângulo ao redor do rosto
                    
                if(label[0] == 4): #Mariana
                
                    #então bota um texto em cima da caixinha
                    cv2.putText(img,'Mariana',(x - 20,y + h + 60), font, 3,(0,0,255),5,cv2.LINE_AA)
                    
                    #pinte um retângulo ao redor do rosto

        #redimensione só pra ficar bonito na tela
        img = cv2.resize(img, (int(0.75 * img.shape[1]), int(0.75 * img.shape[0])))

        #exibir na tela!
        cv2.imshow("reconhecimento", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    
    cv2.destroyAllWindows()

#cria um arquivo que indica que aquela foto pertence
#a tal pessoa
criaArquivoDeRotulo()
    
#carrega o arquivo
try:
    arqTreino = open("Treino", "r")    
except OSError:
    print("Erro ao abrir o arquivo.") 

#Cria duas listas com as imagens e os labels com base no arquivo de treinamento
imagens, chaves = criaListaImagensChaves(arqTreino)

modelo = treinaModelo(imagens, chaves)

reconheceVideo(modelo, "todos.mp4")
    