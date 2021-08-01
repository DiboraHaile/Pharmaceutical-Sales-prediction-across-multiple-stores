from os import write
import sys
from sklearn import model_selection  
sys.path.insert(0, '../scripts')
import streamlit as st
import numpy as np
import pandas as pd
import modeling
import visualize
import pickle


# loading the trained model
pickle_in = open('models/2-8-2021-0-17-13.pkl', 'rb') 
model = pickle.load(pickle_in)

def file_uploader():
    uploaded_file = st.file_uploader("Upload Files",type=['csv'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        # st.write(file_details)
    df = pd.read_csv(uploaded_file)
    return df

@st.cache()  
# defining the function which will make 
# the prediction using data about the users 
def prediction(Store, df):   
    df["Store"] = int(Store)
    df = modeling.format_datetime(df)
    dates = df["Date"]
    df = df.drop(columns="Date")
    x = np.array(df)    
    # Making predictions 
    sale_prediction = modeling.inference_model(x,model)
    df_pred = pd.DataFrame({"Date":dates,"predictions":sale_prediction})
    # visualize.lineplot(df_pred,"Date","Predictions",size=7)
    return df_pred

    
def main_page( ):
        # create input fields to enter model inputs
    # Store, DayOfWeek, Date, Open, Promo, StateHoliday,SchoolHoliday
    Store = st.text_input("Input Store:")
    st.write('Input a csv file with columns "Date","IsHoliday","IsWeekend","IsPromo","Year","Part of the month"')
    df = file_uploader()

    # st.write("Store ",Store)
    st.write("Prediction for Store ",Store)
    result = pd.DataFrame()
    
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Store,df)
        st.write(result)
        # st.success('The user is {}'.format(result))     
  


     
if __name__=='__main__': 
    main_page()