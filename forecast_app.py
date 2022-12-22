import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import plotly.express as px
import datetime
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from tensorflow import keras
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.preprocessing.sequence import TimeseriesGenerator
import math


# Page config
st.set_page_config(page_title="ZIZI bodywash forecast", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


scaler = MinMaxScaler()

header = st.container()
dashboard = st.container()
graph_dashboard = st.container()

#function for loading data
def load_data():
    total_sales = pd.read_csv('total_sales_ZIZI.csv', parse_dates = ['date'])
    #create year and month column from date column
    total_sales['month'] = [d.strftime('%B') for d in total_sales['date'] ]
    total_sales['year'] = [d.year for d in total_sales['date']]
    total_sales['month&year'] = total_sales.month + " " + total_sales.year.map(str)
    return total_sales

#load data
total_sales = load_data()



#function for input new data
def input(file, date):
    upload_file = pd.read_csv(file)
    upload_file['date'] = date
    upload_file.to_csv('total_sales_ZIZI.csv', mode = 'a', header = False, index=False)


#function for create new model
def create_model(selected_product):
    
    # define generator
    n_input = 7
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
    
    #define model 
    model = Sequential()
    model.add(SimpleRNN(100, activation='tanh', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(generator,epochs=100)
    
    #saving the model
    model_name = selected_product +" "+ total_sales['month'].tail(1) +" "+ total_sales['year'].tail(1).astype("string")
    model.save(f'models/{model_name}')
    model_name.to_csv('model_name.csv', mode = 'a', header = False, index=False)



#function for evaluating the model
def eval(model):

    test_predictions = []
    first_eval_batch = scaled_train[-7:]
    current_batch = first_eval_batch.reshape((1, 7, 1))

    for i in range (len(test)):
    
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]
    
        # append the prediction into the array
        test_predictions.append(current_pred)
    
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

    #mentranformasi kembali hasil prediksi
    eval_test = scaler.inverse_transform(test_predictions)
    return eval_test


#function for making prediction
def predict(model):
    predictions = []
    first_batch = scaled_test
    current_batch = first_batch.reshape((1, 7, 1))

    for i in range (3):
    
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]
    
        # append the prediction into the array
        predictions.append(current_pred)
    
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

    #mentranformasi kembali hasil prediksi
    result_predictions = scaler.inverse_transform(predictions)

    return result_predictions


#function for running forecast
def run_forecast(selected_product, selected_model):
    model_name = f'{selected_product} {selected_model}'
    model = keras.models.load_model(f'models/{model_name}')
    testing_result = eval(model)
    prediction_result = predict(model)
    
    return prediction_result, testing_result



with header:
    # Dashboard Title
    st.header('Demand Forecast')


with dashboard:
    left_col, right_col = st.columns(2)
    
    #input new data
    with left_col.expander("input new data", expanded =False):
        with st.form("my form"):
            file = st.file_uploader("Upload a csv file", type="csv")
            date = st.date_input("Choose sales date", datetime.date(2022, 6, 30))
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            st.write("input data success")
            input(file, date)
            total_sales = load_data()
            right_col.write(total_sales.tail(10))

    #selectbox product
    products = ("ZIZI Body Wash Mild Refill 450 ml", "ZIZI Body Wash Mild 250 ml", "ZIZI Body Wash Mild 100 ml")
    selected_product = left_col.selectbox(label= "Select product", options = products)

    #selectbox model
    models = pd.read_csv('model_name.csv')
    model_date = models['date'].unique()
    
    selected_model = left_col.selectbox(label= "Select forecast model version", options = model_date, help="Pilih model untuk membuat forecast")

    #filter body wash product from total sales
    body_wash = total_sales[total_sales["Nama Produk"].isin(["ZIZI Body Wash Mild Refill 450 ml","ZIZI Body Wash Mild 250 ml","ZIZI Body Wash Mild 100 ml"])]
    body_wash = body_wash[["Nama Produk", "Kuantitas Terjual","Total Nilai terjual"]]

    #groupby total sales per product
    sales_perProduct = body_wash.groupby(['Nama Produk']).sum().sort_values('Kuantitas Terjual', ascending = False)
    rounded_value = sales_perProduct.astype(np.int64)
    #rounded_value = sales_perProduct.round(decimals=2)
    right_col.text("Total Sales per product")
    right_col.write(sales_perProduct.round(decimals=2))


#menmfilter data produk yang dipilih dari total sales
selected_body_wash = total_sales.loc[total_sales['Nama Produk'] == selected_product ]
body_wash = selected_body_wash[['date','Kuantitas Terjual']]
body_wash = body_wash.set_index('date')

#membagi data yang akan menjadi data input
# test = body_wash.iloc[-math.ceil(len(body_wash) * .20):]
# train = body_wash.iloc[:-math.ceil(len(body_wash) * .20)]

test = body_wash.iloc[-7:]
train = body_wash.iloc[:-7]


#normalisasi data dengan scaling menjadi range antara 0 dan 1 
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

#call forecast function
prediction_result, testing_result = run_forecast(selected_product, selected_model)

#use make_future_dataframe() function from fbprophet to make future date column
df = selected_body_wash[['date','Kuantitas Terjual']]
df = df.rename(columns={"date":"ds","Kuantitas Terjual":"y"})
m = Prophet()
m.fit(df)
future_date = m.make_future_dataframe(periods=3, freq='M')
future_date = future_date.rename(columns={'ds':'date'})

tab1, tab2 = st.tabs(['Forecast (Next 3-Month)', 'Test Forecast'])

with tab1:
    raw_data = selected_body_wash[['date','Kuantitas Terjual']]
    prediction_value = pd.DataFrame(prediction_result, columns = ['Prediction'])
    #concat actual past data with predictions value in different column 
    forecast = pd.concat([raw_data,prediction_value], ignore_index = True)
    forecast = forecast.rename(columns={'Kuantitas Terjual':'actual'})
    forecast['date'] = future_date['date']
    #show forecast plot
    fig = px.line(data_frame = forecast, x = forecast["date"], y = forecast.columns[1:], title = selected_product)
    st.plotly_chart(fig)

with tab2:
    left, right = st.columns([2,1], gap='small')

    testing_forecast = raw_data.iloc[-7:]
    testing_forecast['Prediction'] = testing_result
    testing_forecast = testing_forecast.rename(columns={'Kuantitas Terjual':'actual'})
    fig = px.line(data_frame = testing_forecast, x = testing_forecast["date"], y = testing_forecast.columns[1:], title = selected_product)
    left.plotly_chart(fig)

    testing_forecast['month'] = [d.strftime('%B') for d in testing_forecast['date'] ]
    testing_forecast['year'] = [d.year for d in testing_forecast['date']]
    testing_forecast['month&year'] = testing_forecast.month + " " + testing_forecast.year.map(str)
    testing_forecast = testing_forecast.reset_index()
    right.table(testing_forecast[['month&year', 'actual', 'Prediction']])
