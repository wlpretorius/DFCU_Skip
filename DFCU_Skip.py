# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:36:55 2021

@author: Admin
"""
#import required packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import statsmodels
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
import base64
from streamlit import caching
from PIL import Image
from datetime import datetime, timedelta
from numpy.random import seed
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
import math
import seaborn as sns
import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.arima.stationarity import ADFTest

# Pages and Tabs
st.set_page_config(layout='wide', initial_sidebar_state="expanded")
sidebarimage = Image.open("Riskworx Wordmark Blue.png") 
st.sidebar.image(sidebarimage, width=250)
df = st.sidebar.file_uploader('Upload your CSV file here:', type='csv')
st.sidebar.header('Navigation')
tabs = ["About","Run all the models"]
page = st.sidebar.radio("Riskworx Pty (Ltd)",tabs)

if page == "About":
    icon = Image.open("RWx & Slogan.png")
    image = Image.open("RWx & Slogan.png")
    st.image(image, width=700)
    st.header("About")
    st.write("This interface is designed for the Development Finance Company of Uganda Bank Limited (DFCU) to forecast their interest rate data as provided to Riskworx Pty (Ltd).")
    st.header("Requirements")
    st.write("Currently, one input csv file is needed for the models to provide interest rate forecasts.")         
    st.header("How to use")  
    st.write("Please read the instructions below then insert your CSV file in the left tab. The models will update automatically. Note, all plots allow zooming.")
    
    # if st.checkbox("Show Instructions"):
    #     def show_pdf(file_path):
    #         with open(file_path,"rb") as f:
    #               base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    #         pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    #         st.markdown(pdf_display, unsafe_allow_html=True)
    #     show_pdf("C:\\Users\\Admin\\Desktop\\Riskworx\\ALP\\DFCU_Instructions_on_GUI.pdf")
    
    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html('DFCU_Instructions_on_GUI.pdf', 'Instructions'), unsafe_allow_html=True)
    
    st.write("")
    st.header("Author and Creator")
    st.markdown(""" **[Willem Pretorius](https://www.riskworx.com//)**""")
    st.markdown(""" **[Contact](mailto:willem.pretorius@riskworx.com)** """)
    st.write("Created on 30/03/2021")
    st.write("Last updated: **16/04/2021**")
    st.write("")
    st.header("Product Owner")
    st.markdown(""" **[Illse Nell](https://www.riskworx.com//)**""")
    st.markdown(""" **[Contact](mailto:illse.nell@riskworx.com)** """)
    st.write("")
    st.header("More about Streamlit")                        
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    
if page == "Run all the models":
    if df is not None:            
        appdata = pd.read_csv(df, index_col='Date', skip_blank_lines=True)
        appdata = appdata.dropna()
        appdata.index = pd.to_datetime(appdata.index, format='%Y-%m-%d', errors='ignore')
        appdata.index = appdata.index.date
        st.dataframe(appdata)
         
        # LIBOR - HW
        appdata_libor = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_libor = appdata_libor.dropna()
        appdata_libor.index = pd.to_datetime(appdata_libor.index)
        appdata_libor.index = appdata_libor.index.date
        periods_input = st.number_input('How many months would you like to forecast into the future?', min_value = 1, max_value = 3)    
        fitted_model_libor = ExponentialSmoothing(appdata_libor['6M_LIBOR'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
        predictions_libor = fitted_model_libor.forecast(periods_input)        
        predictions_libor.index = pd.to_datetime(predictions_libor.index)
        predictions_libor.index = predictions_libor.index.date
        # st.subheader("6M LIBOR Forecasted Values with Holt-Winters Triple Exponential Smoothing")
        # st.write(predictions_libor)
        
        # LIBOR - ARIMA
        # st.subheader("6M LIBOR Forecasted Values with ARIMA")
        adf_test_libor = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_libor.should_diff(appdata_libor) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1
        
        model_libor_arima = auto_arima(appdata_libor['6M_LIBOR'],d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_libor_arima.fit(appdata_libor['6M_LIBOR'])
        y_pred_libor_arima = model_libor_arima.predict(n_periods=periods_input)
        y_pred_libor_arima_df = pd.DataFrame(data = y_pred_libor_arima, columns=appdata_libor.columns).abs()
        y_pred_libor_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_libor_arima_df.index = pd.to_datetime(y_pred_libor_arima_df.index)
        y_pred_libor_arima_df.index = y_pred_libor_arima_df.index.date
        # y_pred_libor_arima_df.index = pd.to_datetime(y_pred_libor_arima_df.index, format='%Y-%d-%m', errors='ignore').strftime('%Y-%m')
        # st.write(y_pred_libor_arima_df)

        # FCY - HW        
        appdata_fcy = appdata.drop(columns=['Interbank_Rate', 'Prime Rate','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_fcy.index = pd.to_datetime(appdata_fcy.index)
        appdata_fcy.index = appdata_fcy.index.date
        final_model_fcy = ExponentialSmoothing(appdata_fcy,trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions_fcy = final_model_fcy.forecast(periods_input).round(5)
        predictions_fcy.index = pd.to_datetime(predictions_fcy.index)
        predictions_fcy.index = predictions_fcy.index.date
        # st.subheader("6M Fixed Deposit - FCY Forecasted Values with Holt-Winters Triple Exponential Smoothing")   
        # st.write(predictions_fcy)
            
        # FCY - ARIMA    
        # st.subheader("6M Fixed Deposit - FCY Forecasted Values with ARIMA")
        adf_test_fcy = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_fcy.should_diff(appdata_fcy) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1    
        
        model_fcy_arima = auto_arima(appdata_fcy,d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_fcy_arima.fit(appdata_fcy)
        y_pred_fcy_arima = model_fcy_arima.predict(n_periods=periods_input)
        y_pred_fcy_arima_df = pd.DataFrame(data = y_pred_fcy_arima, columns=appdata_fcy.columns).abs()
        y_pred_fcy_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_fcy_arima_df.index = pd.to_datetime(y_pred_fcy_arima_df.index)
        y_pred_fcy_arima_df.index = y_pred_fcy_arima_df.index.date
        # y_pred_fcy_arima_df.index = pd.to_datetime(y_pred_fcy_arima_df.index).strftime('%Y-%m')
        # st.write(y_pred_fcy_arima_df)

        # LCY - HW
        appdata_lcy = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_lcy.index = pd.to_datetime(appdata_lcy.index)
        appdata_lcy.index = appdata_lcy.index.date
        final_model_lcy = ExponentialSmoothing(appdata_lcy['6M Fixed Deposit - LCY'],trend='add',seasonal='mul',seasonal_periods=12).fit()
        predictions_lcy = final_model_lcy.forecast(periods_input)
        predictions_lcy.index = pd.to_datetime(predictions_lcy.index)
        predictions_lcy.index = predictions_lcy.index.date
        # st.subheader("6M Fixed Deposit - LCY Forecasted Values with Holt-Winters Triple Exponential Smoothing")
        # st.write(predictions_lcy)

        # LCY - ARIMA
        # st.subheader("6M Fixed Deposit - LCY Forecasted Values with ARIMA")
        adf_test_lcy = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_lcy.should_diff(appdata_lcy) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1 
        
        model_lcy_arima = auto_arima(appdata_lcy,d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_lcy_arima.fit(appdata_lcy)
        y_pred_lcy_arima = model_lcy_arima.predict(n_periods=periods_input)
        y_pred_lcy_arima_df = pd.DataFrame(data = y_pred_lcy_arima, columns=appdata_lcy.columns).abs()
        y_pred_lcy_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_lcy_arima_df.index = pd.to_datetime(y_pred_lcy_arima_df.index)
        y_pred_lcy_arima_df.index = y_pred_lcy_arima_df.index.date
        # y_pred_lcy_arima_df.index = pd.to_datetime(y_pred_lcy_arima_df.index).strftime('%Y-%m')
        # st.write(y_pred_lcy_arima_df) 

        # Demand Deposits - HW
        appdata_demanddeposits = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_demanddeposits.index = pd.to_datetime(appdata_demanddeposits.index)
        appdata_demanddeposits.index = appdata_demanddeposits.index.date
        final_model_demanddeposits = ExponentialSmoothing(appdata_demanddeposits['Demand_Deposits'],trend='add',seasonal='add',seasonal_periods=12).fit()
        predictions_demanddeposits = final_model_demanddeposits.forecast(periods_input)
        predictions_demanddeposits.index = pd.to_datetime(predictions_demanddeposits.index)
        predictions_demanddeposits.index = predictions_demanddeposits.index.date
        # st.subheader("Demand Deposits Forecasted Values with Holt-Winters Triple Exponential Smoothing")
        # st.write(predictions_demanddeposits)

        # Demand Deposits - ARIMA
        # st.subheader("Demand Deposits Forecasted Values with ARIMA")
        adf_test_demanddeposits = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_demanddeposits.should_diff(appdata_demanddeposits) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1         
        
        model_demanddeposits_arima = auto_arima(appdata_demanddeposits,d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_demanddeposits_arima.fit(appdata_demanddeposits)        
        y_pred_demanddeposits_arima = model_demanddeposits_arima.predict(n_periods=periods_input)
        y_pred_demanddeposits_arima_df = pd.DataFrame(data = y_pred_demanddeposits_arima, columns=appdata_demanddeposits.columns).abs()
        y_pred_demanddeposits_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_demanddeposits_arima_df.index = pd.to_datetime(y_pred_demanddeposits_arima_df.index)
        y_pred_demanddeposits_arima_df.index = y_pred_demanddeposits_arima_df.index.date
        # y_pred_demanddeposits_arima_df.index = pd.to_datetime(y_pred_demanddeposits_arima_df.index).strftime('%Y-%m')
        # st.write(y_pred_demanddeposits_arima_df)

        # Savings Deposits - HW
        appdata_savingsdeposits = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_savingsdeposits.index = pd.to_datetime(appdata_savingsdeposits.index)
        appdata_savingsdeposits.index = appdata_savingsdeposits.index.date
        final_model_savingsdeposits = ExponentialSmoothing(appdata_savingsdeposits['Savings_Deposits'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions_savingsdeposits = final_model_savingsdeposits.forecast(periods_input)
        predictions_savingsdeposits.index = pd.to_datetime(predictions_savingsdeposits.index)
        predictions_savingsdeposits.index = predictions_savingsdeposits.index.date
        # st.subheader("Savings Deposits Forecasted Values with Holt-Winters Triple Exponential Smoothing")
        # st.write(predictions_savingsdeposits) 

        # Savings Deposits - ARIMA
        # st.subheader("Savings Deposits Forecasted Values with ARIMA")
        adf_test_savingsdeposits = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_savingsdeposits.should_diff(appdata_savingsdeposits) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1    
        
        model_savingsdeposits_arima = auto_arima(appdata_savingsdeposits,d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_savingsdeposits_arima.fit(appdata_savingsdeposits)        
        y_pred_savingsdeposits_arima = model_savingsdeposits_arima.predict(n_periods=periods_input)
        y_pred_savingsdeposits_arima_df = pd.DataFrame(data = y_pred_savingsdeposits_arima, columns=appdata_savingsdeposits.columns).abs()
        y_pred_savingsdeposits_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_savingsdeposits_arima_df.index = pd.to_datetime(y_pred_savingsdeposits_arima_df.index)
        y_pred_savingsdeposits_arima_df.index = y_pred_savingsdeposits_arima_df.index.date
        # y_pred_savingsdeposits_arima_df.index = pd.to_datetime(y_pred_savingsdeposits_arima_df.index).strftime('%Y-%m')
        # st.write(y_pred_savingsdeposits_arima_df) 

        # Lending - Foreign - HW
        appdata_lendingforeign = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign'])
        appdata_lendingforeign.index = pd.to_datetime(appdata_lendingforeign.index)
        appdata_lendingforeign.index = appdata_lendingforeign.index.date
        final_model_lendingforeign = ExponentialSmoothing(appdata_lendingforeign['Lending_Rates-Foreign'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        predictions_lendingforeign = final_model_lendingforeign.forecast(periods_input)
        predictions_lendingforeign.index = pd.to_datetime(predictions_lendingforeign.index)
        predictions_lendingforeign.index = predictions_lendingforeign.index.date
        # st.subheader("Lending - Foreign Forecasted Values with Holt-Winters Triple Exponential Smoothing")
        # st.write(predictions_lendingforeign)

        # Lending - Foreign - ARIMA
        # st.subheader("Lending - Foreign Forecasted Values with ARIMA")
        adf_test_lendingforeign = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test_lendingforeign.should_diff(appdata_lendingforeign) 
        nr_diff = 0
        if p_val < 0.05:
            print('Time Series is stationary. p-value is',  p_val)
            nr_diff = 0
        else:
            print('Time Series is not stationary. p-value is',  p_val, '. Differencing is needed: ', should_diff)
            nr_diff = 1    
        
        model_lendingforeign_arima = auto_arima(appdata_lendingforeign,d=nr_diff,trace=True,start_p=0,start_q=0,max_p=10, max_q=10,seasonal=False,stepwise=False,suppress_warnings=True,error_action='ignore',approximation = False)
        model_lendingforeign_arima.fit(appdata_lendingforeign)        
        y_pred_lendingforeign_arima = model_lendingforeign_arima.predict(n_periods=periods_input)
        y_pred_lendingforeign_arima_df = pd.DataFrame(data = y_pred_lendingforeign_arima, columns=appdata_lendingforeign.columns).abs()
        y_pred_lendingforeign_arima_df.index = pd.date_range(appdata.index.max() + timedelta(1), periods = periods_input, freq='MS')
        y_pred_lendingforeign_arima_df.index = pd.to_datetime(y_pred_lendingforeign_arima_df.index)
        y_pred_lendingforeign_arima_df.index = y_pred_lendingforeign_arima_df.index.date
        # y_pred_lendingforeign_arima_df.index = pd.to_datetime(y_pred_lendingforeign_arima_df.index).strftime('%Y-%m')
        # st.write(y_pred_lendingforeign_arima_df)

        # Local Rates - VAR
        appdata_localrates = appdata.drop(columns=['6M Fixed Deposit - FCY','6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR','Demand_Deposits-Foreign', 'Savings_Deposits-Foreign','Lending_Rates-Foreign'])
        appdata_localrates.index = pd.to_datetime(appdata_localrates.index)
        appdata_localrates.index = appdata_localrates.index.date
        final_model_localrates = VAR(endog=appdata_localrates)
        model_fit_localrates = final_model_localrates.fit(1)
        yhat_localrates = model_fit_localrates.forecast(final_model_localrates.y, periods_input)
        true_predictions_localrates = pd.DataFrame(data=yhat_localrates, columns=appdata_localrates.columns)
        true_predictions_localrates['Central_Bank_Rate_(CBR)']=true_predictions_localrates['Central_Bank_Rate_(CBR)'].apply(np.floor)
        true_predictions_localrates.index = pd.to_datetime(true_predictions_localrates.index)
        index_localrates = pd.date_range(appdata_localrates.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions_localrates.index = index_localrates.date
        # true_predictions_localrates.index = pd.to_datetime(true_predictions_localrates.index).strftime('%Y-%m')
        # st.subheader("Local Rates Forecasted Values with Vector Autoregression")
        # st.dataframe(true_predictions_localrates)

        # Local Rates - VARMA
        model_localrates_varma = VARMAX(appdata_localrates, order=(1, 2))
        model_localrates_varma_fit = model_localrates_varma.fit(disp=False)
        yhat_localrates_varma = model_localrates_varma_fit.forecast(steps=periods_input)
        yhat_localrates_varma_df = pd.DataFrame(yhat_localrates_varma, columns=appdata_localrates.columns)
        yhat_localrates_varma_df.index = pd.date_range(appdata_localrates.index.max() + timedelta(1), periods = periods_input, freq='MS')
        yhat_localrates_varma_df['Central_Bank_Rate_(CBR)'] = yhat_localrates_varma_df['Central_Bank_Rate_(CBR)'].apply(np.floor)
        yhat_localrates_varma_df.index = yhat_localrates_varma_df.index.date
        # yhat_localrates_varma_df.index = pd.to_datetime(yhat_localrates_varma_df.index).strftime('%Y-%m')
        # st.subheader("Local Rates Forecasted Values with Vector Autoregression Moving Average")
        # st.dataframe(yhat_localrates_varma_df)

        # Foreign Deposits - VAR
        appdata_foreign = appdata.drop(columns=['Interbank_Rate', 'Prime Rate', '6M Fixed Deposit - FCY','Central_Bank_Rate_(CBR)', '6M Fixed Deposit - LCY', 'Demand_Deposits','Savings_Deposits', '6M_LIBOR', '6M T-Bill Rate','Lending_Rates-Foreign'])
        appdata_foreign.index = pd.to_datetime(appdata_foreign.index)
        appdata_foreign.index = appdata_foreign.index.date
        final_model_foreign = VAR(endog=appdata_foreign)
        model_fit_foreign = final_model_foreign.fit(1)
        yhat_foreign = model_fit_foreign.forecast(model_fit_foreign.y, periods_input)
        true_predictions_foreign = pd.DataFrame(data=yhat_foreign, columns=appdata_foreign.columns)
        index_foreign = pd.date_range(appdata_foreign.index.max() + timedelta(1), periods = periods_input, freq='MS')
        true_predictions_foreign.index = index_foreign.date
        # true_predictions_foreign.index = pd.to_datetime(true_predictions_foreign.index).strftime('%Y-%m')
        # st.subheader("Foreign Deposits Forecasted Values with Vector Autoregression")
        # st.dataframe(true_predictions_foreign)

        # Foreign Deposits - VARMA
        model_foreign_varma = VARMAX(appdata_foreign, order=(1, 2))
        model_foreign_varma_fit = model_foreign_varma.fit(disp=False)                                                         
        yhat_foreign_varma = model_foreign_varma_fit.forecast(steps=periods_input)
        yhat_foreign_varma_df = pd.DataFrame(yhat_foreign_varma, columns=appdata_foreign.columns)
        yhat_foreign_varma_df.index = pd.date_range(appdata_foreign.index.max() + timedelta(1), periods = periods_input, freq='MS')
        yhat_foreign_varma_df.index = yhat_foreign_varma_df.index.date
        # yhat_foreign_varma_df.index = pd.to_datetime(yhat_foreign_varma_df.index).strftime('%Y-%m')
        # st.subheader("Foreign Deposits Forecasted Values with Vector Autoregression Moving Average")
        # st.dataframe(yhat_foreign_varma_df)


        # Combining everything into one dataframe
        st.subheader("Holt-Winters Combined Output of Forecasts")
        column_name = ['6M_LIBOR']
        full_csv_hw = pd.DataFrame(data = predictions_libor, columns = column_name)
        full_csv_hw.index = pd.to_datetime(full_csv_hw.index)
        full_csv_hw.index = full_csv_hw.index.date
        full_csv_hw['6M Fixed Deposit - FCY'] = predictions_fcy
        full_csv_hw['6M Fixed Deposit - LCY'] = predictions_lcy
        full_csv_hw['Demand_Deposits'] = predictions_demanddeposits
        full_csv_hw['Savings_Deposits'] = predictions_savingsdeposits
        full_csv_hw['Lending_Rates-Foreign'] = predictions_lendingforeign
        st.write(full_csv_hw)
        
        st.subheader("ARIMA Combined Output of Forecasts")
        full_csv_arima = pd.DataFrame(data = y_pred_libor_arima_df, columns = column_name)
        full_csv_arima['6M Fixed Deposit - FCY'] = y_pred_fcy_arima_df
        full_csv_arima['6M Fixed Deposit - LCY'] = y_pred_lcy_arima_df
        full_csv_arima['Demand_Deposits'] = y_pred_demanddeposits_arima_df
        full_csv_arima['Savings_Deposits'] = y_pred_savingsdeposits_arima_df
        full_csv_arima['Lending_Rates-Foreign'] = y_pred_lendingforeign_arima_df
        st.write(full_csv_arima)
        
        st.subheader("Vector Autoregression Combined Output of Forecasts")
        column_name_var = ['Interbank_Rate', 'Prime Rate', 'Central_Bank_Rate_(CBR)', '6M T-Bill Rate']
        full_csv_var = pd.DataFrame(data = true_predictions_localrates, columns = column_name_var)
        full_csv_var_merged = pd.concat([full_csv_var, true_predictions_foreign], axis=1)
        st.write(full_csv_var_merged)
        
        st.subheader("Vector Autoregression Moving Average Combined Output of Forecasts")
        column_name_var = ['Interbank_Rate', 'Prime Rate', 'Central_Bank_Rate_(CBR)', '6M T-Bill Rate']
        full_csv_varma = pd.DataFrame(data = yhat_localrates_varma_df, columns = column_name_var)
        full_csv_varma_merged = pd.concat([full_csv_varma, yhat_foreign_varma_df], axis=1)
        st.write(full_csv_varma_merged)
        
        # Downloads
        st.subheader("The links below allows you to download the newly created forecasts to your computer for further analysis and use.")
        
        csv_exp_hw = full_csv_hw.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64_hw = base64.b64encode(csv_exp_hw.encode()).decode()  # some strings <-> bytes conversions necessary here
        href_hw = f'<a href="data:file/csv;base64,{b64_hw}">Download CSV File</a> (right-click and save as ** &lt;HW__forecasts_&gt;.csv**)'
        st.markdown(href_hw, unsafe_allow_html=True)
        
        csv_exp_arima = full_csv_arima.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64_arima = base64.b64encode(csv_exp_arima.encode()).decode()  # some strings <-> bytes conversions necessary here
        href_arima = f'<a href="data:file/csv;base64,{b64_arima}">Download CSV File</a> (right-click and save as ** &lt;ARIMA__forecasts_&gt;.csv**)'
        st.markdown(href_arima, unsafe_allow_html=True)
        
        csv_exp_var = full_csv_var_merged.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64_var = base64.b64encode(csv_exp_var.encode()).decode()  # some strings <-> bytes conversions necessary here
        href_var = f'<a href="data:file/csv;base64,{b64_var}">Download CSV File</a> (right-click and save as ** &lt;VAR__forecasts_&gt;.csv**)'
        st.markdown(href_var, unsafe_allow_html=True)
        
        csv_exp_varma = full_csv_varma_merged.to_csv(index=True)
        # When no file name is given, pandas returns the CSV as a string
        b64_varma = base64.b64encode(csv_exp_varma.encode()).decode()  # some strings <-> bytes conversions necessary here
        href_varma = f'<a href="data:file/csv;base64,{b64_varma}">Download CSV File</a> (right-click and save as ** &lt;VARMA__forecasts_&gt;.csv**)'
        st.markdown(href_varma, unsafe_allow_html=True)

       
        
        

































    
