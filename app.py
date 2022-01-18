import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

# creating a prediction fn
def diabetes_pred(input_data) :
    # input_data=(1,89,66,23,94,28.1,0.167,21)
    input_data=np.asarray(input_data) #convert to numpy

    # reshape as we are predecting for one instance
    input_data=input_data.reshape(1,-1)
    pred=loaded_model.predict(input_data)
    print(pred)


    if(pred[0]==0):
        return 'diabetic'
    else :
        return 'not diabetic'


def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # code for prediction
    diagonsis = ''

    # button to predict
    if st.button('Diabetes Test Result'):
        diagonsis = diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagonsis)



if __name__ == '__main__':
    main()