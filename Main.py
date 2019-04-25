#!/usr/bin/python3

import tkinter

import numpy as np

import flower_classification
import cv2

from tkinter import ttk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from os import listdir
from os.path import isfile, join



def openDirectory():
    name = askdirectory()
    return (name)

def OpenFileName():
    name = askopenfilename(initialdir="C:/",
                           filetypes =(("Text File", ".txt"),("All Files",".*")),
                           title = "Choose a file."
                           )
    return (name)

def setModelLocation():
    name = OpenFileName()
    tf_model.delete(0,'end')
    tf_model.insert(0,name)

def setUnClassifiedLocation():
    path = openDirectory()
    tf_unclassfied.delete(0,'end')
    tf_unclassfied.insert(0,path)

def runPredict():
    model = flower_classification.model
    model.load_weights('C:\\Users\\user\\PycharmProjects\\GuiFlower\\flowers_model.h5')
    #mypath = tf_unclassfied.get()
    mypath = "C:\\Users\\user\\Downloads\\rose"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        img = cv2.imread(mypath + "\\" + file,mode='RGB')
        img = cv2.resize(img,(128,128))
        x = np.expand_dims(img, axis=0)
        img = np.vstack([x])
        predict = model.predict(img)
        print(predict)



root = tkinter.Tk()
root.title('File Opener')
root.geometry('600x600')


label1 = ttk.Label(root,text='Unclassefied Pictures')
label1.place(x=20, y=30)

tf_unclassfied = ttk.Entry(root,width=50)
tf_unclassfied.place(x=20, y=50)

btn_borwseUnClassified = ttk.Button(root,text='Browse',command=setUnClassifiedLocation)
btn_borwseUnClassified.place(x=350, y=47.5)

label2 = ttk.Label(root,text='Model')
label2.place(x=20, y=100)

tf_model = ttk.Entry(root,width=50)
tf_model.place(x=20, y=120)

btn_borwseModel = ttk.Button(root,text='Browse',command=setModelLocation)
btn_borwseModel.place(x=350, y=117.5)

btn_predict = ttk.Button(root,text='Predict',command=runPredict)
btn_predict.place(x=255, y=200)

label3 = ttk.Label(root,text='Results')
label3.place(x=20, y=230)

lb_results = tkinter.Listbox(root,width=92)
lb_results.place(x=20,y=250)


root.mainloop()