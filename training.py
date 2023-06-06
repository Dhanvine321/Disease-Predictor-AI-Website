import numpy as np
import pandas as pd
import tensorflow as tf
from mlmodel import train_data, labels, num_classes
from mlmodel import *

#these all done here to reduce the time taken to start the web app
#defining model is done in mlmodel.py(model architecture)

#COMPILING MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#TRAINING MODEL
history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y))

#SAVING MODEL
model.save('model.h5')

#PLOTTING ACCURACY AND LOSS
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

#load model and predict based on 
model = tf.keras.models.load_model('model.h5')
#input the symptoms as the label name and corresponding to that the value is 1
list_of_symptoms = (i for i in train_data.columns[:-2])
print("these are the symptoms: ", list(list_of_symptoms))
input_symptoms = str(input("Enter the symptoms you experience seperated by a space: "))
input_symptoms = input_symptoms.split()
print(input_symptoms)
symptoms = np.zeros(132)
symptoms = symptoms.reshape(1,132)
#for every symptom in the input, change the value of corresponding label index in symptoms to 1
for i in input_symptoms:
    symptoms[0][train_data.columns.get_loc(i)] = 1
print(symptoms)
#predict the disease
pred = model.predict(symptoms)
pred = np.array(pred)
#print probability of each disease
print(pred*100)


pred_disease = np.argmax(pred, axis=1)
#print the corresponding disease and its probability
print("You might have a case", labels[pred_disease[0]], " with a probability of ", pred[0][pred_disease[0]]*100, "%")



