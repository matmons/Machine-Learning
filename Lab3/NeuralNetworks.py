
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:04:47 2019

@author: Group 5
"""
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
import keras
import sklearn.model_selection as skl
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix


data_test_input = np.load('mnist_test_data.npy')
data_train_input = np.load('mnist_train_data.npy')
label_test = np.load('mnist_test_labels.npy')
label_train = np.load('mnist_train_labels.npy')
print(data_test_input.shape)
print(data_train_input.shape)

#plt.imshow(data_train_input[1].reshape(28, 28), cmap = plt.cm.binary)
#plt.show()


data_test = np.divide(data_test_input,255)
data_train = np.divide(data_train_input,255)


train_labels_y = keras.utils.to_categorical(label_train)
test_labels_y = keras.utils.to_categorical(label_test)

training_data, validation_data, training_y, validation_y = skl.train_test_split(data_train,train_labels_y, test_size=0.3)
"""
model = Sequential([
        Flatten(input_shape=(28,28,1)),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
])

model.summary()

es = EarlyStopping(patience = 15, restore_best_weights=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy')
history = model.fit(training_data, training_y, batch_size=300, epochs=400, callbacks=[es], validation_data=(validation_data,validation_y))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train mlp','validation mlpx'],loc='upper right')
plt.show()



y_pred = model.predict_classes(data_test)
print(y_pred)
score = accuracy_score(label_test , y_pred)
print(score)
conf = confusion_matrix(label_test , y_pred)
print(conf)

"""
model_cnn = Sequential([
        Conv2D(16, kernel_size=3, activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
])

model_cnn.summary()

model_cnn.compile(optimizer='Adam', loss='categorical_crossentropy')
history_cnn = model_cnn.fit(training_data, training_y, batch_size=300, epochs=400, callbacks=[es], validation_data=(validation_data,validation_y))
plt.figure()
plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train cnn','validation cnn'],loc='upper right')
plt.show()

y_pred = model_cnn.predict_classes(data_test)
print(y_pred)
score = accuracy_score(label_test , y_pred)
print(score)
conf = confusion_matrix(label_test , y_pred)
print(conf)
