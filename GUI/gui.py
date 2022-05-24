# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
import csv
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model  # load the trained model to classify sign

model = load_model('my_traffic_model_batch_size_10_data2.h5')

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#F7F9F9')

label = Label(top, background='#F7F9F9', foreground='#ABEBC6', font=('arial', 15, 'bold'))
sign_image = Label(top)


# Defining function for getting texts for every class - labels
def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []

    # Opening 'csv' file and getting image's labels
    with open(file, 'r') as f:
        reader = csv.reader(f)
        # Going through all rows
        for row in reader:
            # Adding from every row second column with name of the label
            label_list.append(row[1])
        # Deleting the first element of list because it is the name of the column
        del label_list[0]
    # Returning resulted list
    return label_list


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    # print(image.shape)
    # pred = model.predict_classes([image])[0]
    scores = model.predict(image)
    # print(scores[0].shape)

    prediction = np.argmax(scores)

    # print(prediction)
    # sign = classes[pred+1]
    labels = label_text('label_names.csv')
    print('Label:', labels[prediction])
    sign = labels[prediction]
    label.configure(foreground='#2ECC71', text=sign)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        classify(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#E74C3C', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Traffic sign classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#F7F9F9', foreground='#E74C3C')
heading.pack()
top.mainloop()
