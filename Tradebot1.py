# libraries
import re

from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
import csv
import os
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
import customtkinter as ctk  # Importing ctk from customtkinter module
import time
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from telethon import TelegramClient, events
# import tensorflow as tf
# import yfinance as yf
import pandas as pd
import random
import yfinance as yf
# from fbprophet import Prophet
# lib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
from datetime import datetime, timedelta
import requests

# from fbprophet import Prophet

# variable
api_id = 17793112
api_hash = '3ea8b89c01d269be1d5a505bc59efe69'
username = 'mmms911'
user_input_channel = 'https://t.me/Python' and 'https://t.me/Forex_Signals_PIPs_Signal_FX' and 'https://t.me/Forex' \
                     and 'https://t.me/Volatilitymastertrades_supports' and 'https://t.me/Qalcommander0' and \
                     'https://t.me/metatrader5forexsigna'
refined_data_folder = r'C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder'
data_eur = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\eur_data_trede.txt"
data_jpy = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\jpy_data_trede.txt"
eur_li = []
jpy_li = []

# layout

PATH = r'C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\trade_assistant\icon_logo.png'
image = Image.open(PATH)
root = ctk.CTk()  # Creating root window using ctk.CTk()
root.resizable(0, 0)
root.geometry('1100x600')
root.title('Trading Assistant')
icon = ImageTk.PhotoImage(image)
root.iconphoto(True, icon)
er_list = []
bp_list = []
ap_list = []
it_list = []
global jpy_price, euro_price

# var
X = []
Y = []
data_eur = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\eur_data_trede.txt"
data_jpy = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\jpy_data_trede.txt"
data_li = []
gross_li = []
ordinal_time = []
hourly_time = []
eur_usd = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\EURUSD5.txt"
usd_chf = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\USDCHF5.txt"
usd_cad = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\USDCAD5.txt"
gbp_usd = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\GBPUSD5.txt"
usd_yen = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\USDJPY5 (1).txt"
xau_usd = r"C:\Users\mmms\Desktop\project\random_file_lib\tredeBOT\data_folder\XAUUSD5.txt"


def get_range(np_data):
    return pd.DataFrame([np.average(price_li) for price_li in np_data])


def predict_next_5_minutes(historical_data, live_data):
    historical_data_x = historical_data[:, 0].reshape(-1, 1)
    historical_data_y = historical_data[:, 1]
    lin_model = LinearRegression()
    lin_model.fit(historical_data_x, historical_data_y)
    live_data_x = live_data.reshape(-1, 1)
    linear_predicted_close_prices = lin_model.predict(live_data_x)

    df = pd.DataFrame({'ds': historical_data[:, 0], 'y': historical_data[:, 1]})
    pr_model = Prophet()
    pr_model.fit(df)
    future = pd.DataFrame({'ds': live_data})
    pr_forecast = pr_model.predict(future)
    pr_predictions = pr_forecast['yhat'].tail().values()

    return linear_predicted_close_prices, pr_predictions


def group_time(curr_time):
    ordinal_time = []
    hourly_time = []
    for t in curr_time:
        time_val = str(t[0])
        time_val = datetime.strptime(time_val, '%Y-%m-%d %H:%M')
        ordinal_time.append(int(time_val.toordinal()))
        hourly_time.append(int(time_val.hour))
    return ordinal_time, hourly_time, curr_time


def get_float(str_li):
    for row in str_li:
        float_data = [float(item) for item in row[1:5]]
        gross_li.append(float_data)
    return np.array(gross_li)


def manipulate_values(val):
    for items in val.values:
        for item in items:
            data_li.append(str(item).split('\t'))
    return np.array(data_li)


def extract_values(f_data):
    with open(f_data, 'r') as rf:
        curr = pd.DataFrame(pd.read_csv(rf))
    return manipulate_values(curr)


DataF = pd.DataFrame()


def clean_data():
    unaveraged_li = []
    #   range_arr = get_range(get_float(extract_values(eur_usd)))
    ord_arr, hour_arr, range_arr = group_time([i for i in extract_values(eur_usd)])
    # range_arr = get_range(get_float(extract_values(eur_usd)))
    DataF['Close'] = range_arr
    # DataF['Hour'] = hour_arr
    # DataF['Ordinal Time'] = ord_arr

    # print(len(group_time([i[0] for i in extract_values(eur_usd)])))
    for i in DataF.values:
        unaveraged_li.append(i[0])

    averaged_li = get_range(get_float(unaveraged_li))
    DataF['Close'] = averaged_li
    DataF['Hour'] = hour_arr
    DataF['Ordinal Time'] = ord_arr
    # [print(i[1:]) for i in DataF.values[1]]


def get_api_data():
    reference_time = datetime(2023, 6, 6, 0, 0, 0)
    time_increment = timedelta(seconds=1)
    current_time = datetime.now()
    elapsed_time = current_time - reference_time
    incremental_time = elapsed_time // time_increment
    it_list.append(incremental_time)
    api_key = 'ZASMYPGR0SQTETJN'
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "EUR",
        "to_currency": "USD",
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()
    try:
        exchange_rate = data['Realtime Currency Exchange Rate']['5. Exchange Rate']
        bid_price = data['Realtime Currency Exchange Rate']['8. Bid Price']
        ask_price = data['Realtime Currency Exchange Rate']['9. Ask Price']
        DataF1 = pd.DataFrame()
        DataF1['Exchange_rate'] = exchange_rate
        DataF1['Bid_price'] = bid_price
        DataF1['Ask_price'] = ask_price
        clean_data()
        return predict_next_5_minutes(np.array(DataF), np.array(DataF1))


    except:
        pass


# scrollbar

scrollbar = ctk.CTkScrollbar(root)
scrollbar.pack(side=ctk.RIGHT, fill=ctk.Y)

# widget

tg = tk.Text(root, bg='#5b96f5', yscrollcommand=scrollbar.set, height=39, width=160, padx=5, pady=5, font=('calibri',
                                                                                                           8,
                                                                                                           'italic'))
tg.place(x=110, y=20)
scrollbar.configure(command=tg.yview)


def refresh():
    def clear_label():
        green_light.configure(text='')

    tg.delete(1.0, tk.END)
    time.sleep(0.5)
    green_light = ctk.CTkLabel(root, text='Refreshed!')
    green_light.place(x=1000, y=20)
    tg.after(5000, clear_label)


# button

myli = [0.01, 0.05, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08]
re_pattern = r'\b(?:sl|stop|loss|take|profit)\b'


def print_mode_activated():
    start_button.configure(bg_color='green', text='STARTED')
    try:
        client = TelegramClient(username, api_id, api_hash)

        @client.on(events.NewMessage(chats=user_input_channel))
        async def newMessageListener(event):
            l_pred, p_pred = get_api_data()
            tg.insert(ctk.END, f'linear model prediction : {l_pred}  prophet model prediction {p_pred}')
            tg.insert(ctk.END, f'\t   [{datetime.now()}]')
            tg.insert(ctk.END, '\n')
            if re.search(re_pattern, event.message.message, re.IGNORECASE):
                for char in event.message.message:
                    tg.insert(ctk.END, char)
                    root.update_idletasks()
                    time.sleep(myli[random.randint(0, 7)])
                tg.insert(ctk.END, f'\t   [{datetime.now()}]')
                tg.insert(ctk.END, '\n')

        with client:
            client.run_until_disconnected()

    except ConnectionError or OSError:
        print('Internet connection fail  ')


start_button = ctk.CTkButton(root, text='   START  ', cursor='plus', font=('calibri', 10, 'bold'), width=20,
                             command=print_mode_activated)
start_button.place(x=18, y=15)

refresh_button = ctk.CTkButton(root, text='  REFRESH ', cursor='plus', font=('calibri', 10, 'bold'), command=refresh,
                               width=20)
refresh_button.place(x=16, y=195)

indicators_button = ctk.CTkButton(root, text='INDICATORS', cursor='plus', font=('calibri', 10, 'bold'), width=20)
indicators_button.place(x=12, y=150)


def buy_mode_activated():
    def clear_label():
        green_light.configure(text='')

    green_light = ctk.CTkLabel(root, text='buy mode activated')
    green_light.place(x=900, y=20)
    tg.after(5000, clear_label)


buy_button = ctk.CTkButton(root, text='    BUY MODE  ', cursor='plus', width=20, font=('calibri',
                                                                                       10, 'bold'),
                           command=buy_mode_activated)
buy_button.place(x=10, y=60)


def sell_mode_activated():
    def clear_label():
        green_light.configure(text='')

    green_light = ctk.CTkLabel(root, text='sell mode activated')
    green_light.place(x=900, y=20)
    tg.after(5000, clear_label)


sell_button = ctk.CTkButton(root, text='   SELL MODE  ', cursor='plus', width=20, font=('calibri',
                                                                                        10, 'bold'),
                            command=sell_mode_activated)
sell_button.place(x=10, y=105)

root.mainloop()
