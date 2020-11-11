#Biblioteca de manipulação de dados do Python
import numpy as np

#OpenCV
import cv2

#biblioteca de interface com sistema operacional do python
import os

os.chdir('C:/OficinaTechDay/') 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#verifica todos os nomes que deverão ser lidos para treinamento
def lerNomesCriaDiretorios(txt):
    
    listaNomesArquivo = []
    
    #Abre arquivo pppara leitura
    arqNomesFotosSeremLidas = open(txt, "r")
        
    #Cria uma lista com os labels a serem lidos
    for linhaArq in arqNomesFotosSeremLidas:
        
        listaNomesArquivo.append(linhaArq.rstrip())
        
    for nome in listaNomesArquivo:

        os.makedirs("dadosTreino" + "/"+ nome, exist_ok=True)
            
    return listaNomesArquivo        

def salvaFacesDetectadas(nome):

    cap = cv2.VideoCapture(nome + ".mp4") #inicia captura da câmera

    counterFrames = 0;
    
    # Preparado para ler 1000 frames
    while(counterFrames < 1000): #quando chegar ao milésimo frame, para

        print(counterFrames)

        # ret = indicador de que foi possível ler um frame
        # frame (imagem colorida)
        ret, frame = cap.read()

        #frame não pode ser obtido? entao sair
        if(ret == False):
            
            #Finaliza a leitura do arquivo.
            cap.release()        
            return

        #Transforma o frame em escala de Cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #faz a leitura da face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #se nenhuma face for achada, continue
        if not np.any(faces):
            continue

        #achou uma face? recorte ela (crop)
        for (x, y, w, h) in faces:
            rostoImg = frame[y:y+h, x:x+w]

        #imagens muito pequenas são desconsideradas
        larg, alt, _ = rostoImg.shape
        if(larg * alt <= 20 * 20):
            continue

        #salva imagem na pasta redimensionando
        rostoImg = cv2.resize(rostoImg, (255, 255))
        cv2.imwrite("dadosTreino" + "/"+ nome + "/" + str(counterFrames)+".png", rostoImg)
        counterFrames += 1
            
    #finaliza a leitura do arquivo.    
    cap.release()


#Arquivo contendo um labbel com o nome dos arquivos na qual faremos a leitura da imagem.
listaNome = lerNomesCriaDiretorios("entrada.txt")
    
 
for nome in listaNome:
    print("Analisando: " + nome)
    salvaFacesDetectadas(nome)


