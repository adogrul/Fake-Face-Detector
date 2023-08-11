import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

model = load_model('deep.h5')

def load_image():
    global img
    file_path = filedialog.askopenfilename()
    img = load_img(file_path, target_size=(128, 128))

def predict():
    img_array = img_to_array(img).flatten() / 255.0
    img_array = np.array(img_array)
    img_array = img_array.reshape(-1, 128, 128, 3)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    if predicted_class[0] == 0:
        prediction_text.set('Prediction: Fake')
    else:
        prediction_text.set('Prediction: Real')

root = tk.Tk()

load_button = tk.Button(root, text='Load Image', command=load_image)
load_button.pack()

prediction_text = tk.StringVar()
predict_button = tk.Button(root, text='Predict', command=predict)
predict_button.pack()

prediction_label = tk.Label(root, textvariable=prediction_text)
prediction_label.pack()

root.mainloop()