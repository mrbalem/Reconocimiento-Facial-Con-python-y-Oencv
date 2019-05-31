#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

win = Tk()

#leer una imagen
def read_imagen(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Mostrar imagen
def mostrarImagen(image):
    cv2.imwrite('hola.jpg',image)
    #images = Image.open('hola.jpg')
    #images.show()
    #plt.imshow(image)
    #plt.xticks([]), plt.yticks([])
    #plt.show()

#funcion para detertar los rostros
def get_faces(image):
    
    image_copy = np.copy(image)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    # Extrae los diseños pre-establecido para detectar rostro de un archivo xml
    face_classifier = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    return faces 

#funcion para dibujar un rectangulo en los rostros detectados

def draw_faces(image, faces=None, plot=True):
    
    if faces is None:
        faces = get_faces(image)
    
    # para no sobreescrivir
    image_with_faces = np.copy(image)
    
    
    for (x,y,w,h) in faces:
        # agrega un cuadrado rojo en el rostro detectado
        cv2.rectangle(image_with_faces, (x,y), (x+w,y+h), (255,0,0), 3)
        
    if plot is True:
        mostrarImagen(image_with_faces)

    else:
        return image_with_faces
    
 
imagen = read_imagen('images/03.jpg')
rostros = get_faces(imagen)
#mostramos los rostros detectados
#print("Faces detected: {}".format(len(rostros)))
draw_faces(imagen,rostros)

lbl = Label(win, text="Faces destecte: {}".format(len(rostros))).pack()



imagess = Image.open('images/03.jpg')
resizeds = imagess.resize((300, 300),Image.ANTIALIAS)
imas = ImageTk.PhotoImage(resizeds)
Widgets = Label(win, image=imas).pack()


##############
images = Image.open('hola.jpg')
resized = images.resize((300, 300),Image.ANTIALIAS)
ima = ImageTk.PhotoImage(resized)
Widget = Label(win, image=ima).pack()
##########
win.mainloop()