# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:25:57 2018

@author: Kaihua
"""

import os,os.path
import threading
import numpy as np
from PIL import Image, ImageTk 
from tkinter import messagebox
import tkinter as tki
import cv2
import csv
haar_file="haarcascade_frontalface_default.xml"
data_dir="data"
label_filename="labels.csv"
recognizer=cv2.face.EigenFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX



training_faces=[]
training_labels=[]
label_index={} #name:id
inverted_index={}#id:name
def load_image_and_label(path=data_dir,label_file=label_filename):
    
    full_path=os.path.join(path,label_file)
    if os.path.isfile(full_path) is False:
        print("File {} does not exist, initializing untrained recognizer".format(label_file))
    with open(full_path,'r') as csvfile:
        reader=csv.reader(csvfile,delimiter=",")
        for row in reader:
            if len(row)!=2:
                continue
            imgfile=row[0]
            
            label=row[1]
            if label not in label_index:
                count=len(label_index)
                label_index[label]=count
                inverted_index[count]=label
            else:
                count=label_index[label]
            #load imgfile
            face=cv2.imread(os.path.join(path,imgfile),cv2.IMREAD_GRAYSCALE)
            training_faces.append(face)
            training_labels.append(count)
            
            
    assert len(training_faces)==len(training_labels)
def neat_label(raw_label):
    parts=raw_label.split('-')
    ret=""
    for s in parts:
        ret+=s.title()+' '
    return ret

def train_recognizer():
    recognizer.train(training_faces,np.array(training_labels))

class FaceRec(object):
    def __init__(self,vs,output_dir):
        '''
        @para
        vs -> cv2.VideoCaputure() object as VideoStream(vs)
        outpu_dir -> str() directory where captured images and label files are stored
        '''
        self.vs = vs
        self.outputPath = output_dir
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.haar_cascade=cv2.CascadeClassifier(haar_file)
        self.captured_face=None
        #Create UI elements
        self.root = tki.Tk()
        self.panel = None
        self.rightGroup=tki.Frame(self.root)
        self.rightGroup.pack(side="right",padx=10)
        
        self.cap_display=tki.Label(self.rightGroup)
        self.cap_display.image=None
        self.cap_display.grid(row=0)
        tki.Label(self.rightGroup,text="Name").grid(row=1,column=0)
        self.name_input=tki.Entry(master=self.rightGroup)
        self.name_input.grid(row=1,column=1)
        self.train_button=tki.Button(self.rightGroup,text="Train",command=self.train)
        self.train_button.grid(row=2)
        
        self.capture_button=tki.Button(self.root,text="Capture",command=self.captureFrame)
        self.capture_button.pack(side="bottom",fill="both",expand="yes",padx=10,pady=10)
        
        #Start Video loop
        self.stopEvent=threading.Event()
        self.thread=threading.Thread(target=self.videoLoop,args=())
        self.thread.start()
        
        self.root.wm_title("Face")
        self.root.wm_protocol("WM_DELETE_WINDOW",self.onClose)
        
    def train(self):
        if self.cap_display.image  is None:
            messagebox.showinfo("Help","Capture a face before training")
            return
        
        #TODO: Actual training
        name=self.name_input.get()
        
        self.name_input.delete(0,'end')
        img=cv2.resize(self.captured_face,(50,50))
        filename=data_dir+"/{}.jpg".format(len(training_labels))
        cv2.imwrite(filename,img)
        #write to csv file
        with open(os.path.join(data_dir,label_filename),'a') as csvfile:
            line="{}.jpg".format(len(training_labels))+","+name+"\n"
            print(line)
            csvfile.write(line)
            
        if name not in label_index:
            ID=len(label_index)
            label_index[name]=ID
            inverted_index[ID]=name
        else:
            ID=label_index[name]
        training_faces.append(img)
        training_labels.append(ID)
        self.cap_display.configure(image="")
        self.cap_display.image=None
        print("Training {}".format(name))
        
        train_recognizer()
        
    def captureFrame(self):
        cur_frame=self.frame
        
        
        faces=self.haar_cascade.detectMultiScale(cur_frame,1.3,5)
        if(len(faces)!=1):
            messagebox.showerror("Error","Exactly one face need to be detected. Currently has {}".format(len(faces)))
            return
        x,y,w,h=faces[0]
        face=cur_frame[y: y + h, x: x + w]
        gray_face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        self.captured_face=gray_face
        image=Image.fromarray(gray_face)
        image=ImageTk.PhotoImage(image)
        self.cap_display.configure(image=image)
        self.cap_display.image=image
        
        
    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                ret,self.frame=self.vs.read()
                
                faces=self.haar_cascade.detectMultiScale(self.frame,1.3,5)
                display=self.frame.copy()
                gray=cv2.cvtColor(display,cv2.COLOR_BGR2GRAY)
                
                for x,y,w,h in faces:
                    cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
                    nbr_predicted, conf = recognizer.predict(cv2.resize(gray[y: y + h, x: x + w],(50,50)))
                    print(nbr_predicted, conf)
                    #TODO: Tweak threshold
                    if conf<2000:
                        pred=inverted_index[nbr_predicted]
                    else:
                        pred="unknown"
                    cv2.putText(display,pred,(x,y+h+10), font, 0.75,(25,25,255),1,cv2.LINE_AA)
                image=cv2.cvtColor(display,cv2.COLOR_BGR2RGB)
                image=Image.fromarray(image)
                image=ImageTk.PhotoImage(image)
                
                if self.panel is None:
                    #init the panel
                    self.panel=tki.Label(self.root,image=image)
                    self.panel.image=image
                    self.panel.pack(side="left",padx=10,pady=10)
                else:
                    self.panel.configure(image=image)
                    self.panel.image=image
        except RuntimeError as e:
            print(str(e))
    def onClose(self):
        self.stopEvent.set()
        print("stop event set")
        self.vs.release()
        print("cv2 released")
        self.root.destroy()
        
        print("tk destroyed")
if __name__=="__main__":
    if os.path.isdir(data_dir) is False:
        os.mkdir(data_dir)
    load_image_and_label()
    print(label_index)
    print (training_faces)
    print(inverted_index)
    print(training_labels)
    train_recognizer()
    vs=cv2.VideoCapture(0)
    output_dir=data_dir
    app=FaceRec(vs,output_dir)
    app.root.mainloop()
    