import streamlit as st
from mlmodel import train_data, labels, num_classes
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')
st.set_page_config(page_title = 'Health Disease Pred App',layout="wide")

#display the title and description in centre
st.markdown("<h1 style='text-align: center; color: #0047AB;'>Health Disease Prediction App</h1>", unsafe_allow_html=True)
# st.subheader('Health Disease Prediction App', baseline_align='center')
st.markdown("<h3 style='text-align: center; color: #0047AB;'>This app predicts the disease based on the symptoms</h3>", unsafe_allow_html=True)
# st.write('This app predicts the disease based on the symptoms')
st.markdown("<h5 style='text-align: center; color: #0047AB;'>Created by Dhanvine, Ignatius and Ajax from Temasek Junior College</h5>", unsafe_allow_html=True)
st.write('The list of symptoms in our database: ')

#display the symptoms in 3 columns
#symptoms are divided into 3 columns to make it easier to read by not displaying as list and remove the brackets and quotes and commas
col1, col2, col3 = st.columns(3)
with col1:
    list_of_symptoms = (i for i in train_data.columns[:-2])
    st.write(list(list_of_symptoms)[:44])
with col2:
    list_of_symptoms = (i for i in train_data.columns[:-2])
    st.write(list(list_of_symptoms)[44:88])
with col3:
    list_of_symptoms = (i for i in train_data.columns[:-2])
    st.write(list(list_of_symptoms)[88:])

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
local_css("style.css")

#collect symptoms from user and predict the disease
list_of_symptoms = (i for i in train_data.columns[:-2])
symptoms = st.multiselect(
    'Enter the symptoms that you experience (the more symptoms you enter, the more accurate the results will be): ',
    list_of_symptoms)
st.write('You selected:', symptoms)

#based on symptom input, change the value of corresponding label index in symptoms to 1
symptoms_list = np.zeros(132)
symptoms_list = symptoms_list.reshape(1,132)
for i in symptoms:
    symptoms_list[0][train_data.columns.get_loc(i)] = 1
#predict the disease
#add button to predict disease only when pressed
if st.button('Predict'):
    #check if there is no input
    if len(symptoms) == 0:
        st.write("Please enter the symptoms and try again")
        st.stop()
    pred = model.predict(symptoms_list)
    pred = np.array(pred)
    pred_disease = np.argmax(pred, axis=1)

    #set a threshold for the probability of the disease to determiene if the disease is likely to be present
    if pred[0][pred_disease[0]] < 0.5:
        st.write('You are unlikely to have any of the serious diseases in our database based on the symptoms you entered. Please consult a doctor if you are still concerned.')
        st.write('If you have more of any symptoms, please enter them too and try again')
        st.write("There is a low probability of you having a case of ", labels[pred_disease[0]], " with a probability of ", pred[0][pred_disease[0]]*100, "%")
    else:
        #display the top 3 most likely diseases and their probability
        st.write("You might have a case of ", labels[pred_disease[0]], " with a probability of ", pred[0][pred_disease[0]]*100, "%")
        st.write("The top 3 most likely diseases are: ")
        #display the top 3 most likely diseases and their probability
        for i in range(3):
            st.write(i+1, ". ", labels[np.argsort(pred)[0][-i-1]], " with a probability of ", pred[0][np.argsort(pred)[0][-i-1]]*100, "%")

st.markdown("<hr style='border: 2px solid #0047AB;'>", unsafe_allow_html=True)
st.write("Do you want to know more about symptoms of a disease? Enter the disease name below and click on the button to find out more")
#collect disease name from user
disease = st.selectbox('Enter the disease name: ', options = labels)
#add button to display symptoms of the disease
if st.button('Find out more'):
    #check if there is no input
    if len(disease) == 0:
        st.write("Please enter the disease name and try again")
        st.stop()
    #display the symptoms of the disease
    st.write("The symptoms of ", disease, " are: ")
    #search the prognosis of the train data for the disease name
    #display the symptoms of the disease in axis = 0
    for i in range(len(train_data)):
        if train_data.iloc[i]['prognosis'] == disease:
            #display the symptoms of the disease but not in table format but as a bullet list
            st.write(train_data.iloc[i][:-2][train_data.iloc[i][:-2]==1].index)
            break
        


st.markdown("<hr style='border: 2px solid #0047AB;'>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #0047AB;'>Take Note!</h5>", unsafe_allow_html=True)
st.write("Please remember that our healthcare disease prediction app has limitations and should not be solely relied upon for medical advice. Its accuracy may be limited, and it is not a substitute for professional medical guidance. Privacy and security concerns exist, and the dynamic nature of healthcare means information may become outdated. Psychological impact should be considered, and seeking guidance from healthcare professionals is essential. Use our app as a tool, but consult a healthcare professional for comprehensive evaluations and decisions regarding your health.")
st.markdown("<p style='text-align: center; color: #0047AB;'>Created by Dhanvine, Ignatius and Ajax from Temasek Junior College for Intel AI Global Impact Creator's Competition 2023</p>", unsafe_allow_html=True)

#clear cache every time the app is run
#this is to prevent the app from crashing when the user enters a lot of symptoms
#the app will crash if the user enters a lot of symptoms as the cache memory is full
#clearing the cache memory will prevent the app from crashing
#the app will run slower when the cache memory is cleared
#the app will run faster when the cache memory is not cleared
#<a href="https://iconscout.com/lotties/self-protection" target="_blank">Free Self Protection Animated Illustration</a> by <a href="https://iconscout.com/contributors/nanoagency">nanoagency</a> on <a href="https://iconscout.com">IconScout</a>
st.cache_data.clear()