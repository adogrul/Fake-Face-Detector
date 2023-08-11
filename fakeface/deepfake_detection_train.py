import os
import cv2
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


input_shape = (128, 128, 3)
data_dir = 'Veriler'

real_data = [f for f in os.listdir(os.path.join(data_dir, 'training_real')) if f.endswith('.jpg')]
fake_data = [f for f in os.listdir(os.path.join(data_dir, 'training_fake')) if f.endswith('.jpg')]

X = []
Y = []

for img in real_data:
    img = load_img(os.path.join(data_dir, 'training_real', img))
    img = img.resize((128, 128))  # Resizing the image
    X.append(img_to_array(img).flatten() / 255.0)
    Y.append(1)

for img in fake_data:
    img = load_img(os.path.join(data_dir, 'training_fake', img))
    img = img.resize((128, 128))  # Resizing the image
    X.append(img_to_array(img).flatten() / 255.0)
    Y.append(0)

Y_val_org = Y


X = np.array(X)
Y = to_categorical(Y, 2)

# Reshape
X = X.reshape(-1, 128, 128, 3)

# Train-Test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)


googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
googleNet_model.trainable = True
model = Sequential()
model.add(googleNet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False),
              metrics=['accuracy'])
model.summary()

# Currently not used
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=0, mode='auto')
EPOCHS = 20
BATCH_SIZE = 100
#modelin geçmişini history nesnesine kaydetme
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), verbose=1)
with open('history.pickle', 'wb') as file:
    pickle.dump(history.history, file)

# Kaydedilen history'yi yükleme
with open('history.pickle', 'rb') as file:
    loaded_history = pickle.load(file)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
t = f.suptitle('Pre-trained InceptionResNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ',
               fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, EPOCHS + 1))

#accuracy grafiği
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, EPOCHS + 1, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch #')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

# Loss grafiği
ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, EPOCHS + 1, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch #')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# Confusion Matrix
Y_pred = model.predict(X)
Y_pred_classes = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(Y_val_org, Y_pred_classes)
df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=ax3)  # font size
ax3.set_ylabel('Actual label', size=14)
ax3.set_xlabel('Predicted label', size=14)
ax3.set_xticklabels(['Fake', 'Real'], size=12)
ax3.set_yticklabels(['Fake', 'Real'], size=12)
ax3.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()
#  sınıflandırma başarımını elde etmek
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
model.save('deep.h5')