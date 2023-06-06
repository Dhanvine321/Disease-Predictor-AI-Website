import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#PREPROCESSING DATA
train_data = pd.read_csv('dataset/Training.csv')
test_data = pd.read_csv('dataset/Testing.csv')

#Splitting data into train and test
train_y = train_data['prognosis']
train_x = train_data.drop('prognosis', axis=1)
train_x = train_x.drop('Unnamed: 133', axis=1)

test_y = test_data['prognosis']
test_x = test_data.drop('prognosis', axis=1)

#CONVERTING DATA TO NUMPY ARRAY
train_x = np.array(train_x)
test_x = np.array(test_x)

#changing labels to one hot encoding
train_y = pd.get_dummies(train_y)
test_y = pd.get_dummies(test_y)
labels = train_y.columns
num_classes = len(labels)

#converting to numpy array
train_y = np.array(train_y)
test_y = np.array(test_y)


# #MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(132,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])