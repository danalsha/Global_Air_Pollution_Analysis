import os 
import joblib 
import pandas as pd 
import streamlit as st

@st.cache_resource
def load_model_and_scaler():#cached so it only loads once per session, avoids reloading on every interaction
    model = joblib.load('rf_model.joblib')#load the trained random forest model
    scaler = joblib.load('scaler.joblib')#load the fitted scaler
    return model, scaler

def aqi_category(value):#map predicted AQI number to its EPA category and display color
    if value<= 50:
        return 'Good' '#2ecc71'
    if value<= 100:#
        return 'Moderate','#f1c40f'
    if value <= 150:
        return 'Unhealthy for Sensitive Groups' ,'#e67e22'
    if value <= 200:
        return 'Unhealthy' , '#e74c3c'
    if value <= 300:
        return 'Very Unhealthy','#9b59b6'
    return 'Hazardous','#7f0000'
st.set_page_config(page_title='AQI Predictor', page_icon='🌍', layout='centered')#page title and icon shown in browser tab
st.title('Global Air Quality Predictor')
st.write(
    'Enter the four pollutant AQI readings for a city and the model will '
    'predict the overall Air Quality Index (AQI) value and category. '
    'The underlying model is a Random Forest Regressor trained on the '
    'Global Air Pollution Dataset (23,463 cities, 175 countries).'
)
if not (os.path.exists( 'rf_model.joblib') and os.path.exists('scaler.joblib' ) ):#check if model files exist before running
    st.error( 
        "Could not find 'rf_model.joblib' and/or 'scaler.joblib'. "
        "Open Global_Air_Pollution_Analysis_Final.ipynb in Jupyter, "
        "run all cells from the top (Kernel → Restart & Run All), then refresh this page. "
        "Step 12 in the notebook writes both files to the project folder."
    )
    st.stop()#stop the app if files are missing

st.markdown('---')#horizontal divider

st.subheader('Pollutant AQI Readings')
col1,col2= st.columns(2)#two column layout for the inputs

with col1:
    co= st.number_input('CO AQI Value' ,min_value=0,max_value=500,value=1, step=1)#CO input
    ozone= st.number_input('Ozone AQI Value', min_value=0 ,max_value=500,value=35 , step=1)#Ozone input

with col2:
    no2 =st.number_input('NO2 AQI Value',min_value=0,max_value=500,value=3,step=1)#NO2 input
    pm25 =st.number_input('PM2.5 AQI Value',min_value=0 ,max_value=500,value=68,step=1)#PM2.5 input, 
st.markdown( '---')#horizontal divider

if st.button( 'Predict AQI', type='primary', use_container_width= True ):#trigger prediction on button click
    model,scaler=load_model_and_scaler()#load model and scaler from disk
    input_df=pd.DataFrame( [[co, ozone, no2, pm25]], columns=['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    )#single row dataframe matching training column names to avoid sklearn warnings
    input_scaled = scaler.transform(input_df)#scale inputs the same way training data was scaled
    predicted_aqi = float(model.predict(input_scaled)[0])#generate the AQI prediction
    category, colour = aqi_category(predicted_aqi)#get the category and color for the predicted value
    st.subheader('Prediction')
    st.metric(label='Predicted AQI Value', value=f'{predicted_aqi:.1f}')#display the numeric prediction
    st.markdown(
        f"<div style='padding:10px; border-radius:6px; background-color:{colour}; "
        f"color:white; text-align:center; font-size:18px; font-weight:bold;'>"
        f"Category: {category}</div>",#colored box showing the AQI category
        unsafe_allow_html=True,
    )
st.markdown('---')
st.caption('CS316: Introduction to AI and Data Science — Global Air Pollution Analysis')#footer