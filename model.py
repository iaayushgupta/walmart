#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from datetime import date
import holidays
import datetime
import joblib



model = Flask(__name__,static_folder='static', static_url_path='/static',template_folder='template')


def create_Holiday_Type(df):
    def create_holiday_type_column(df, dates, holiday_type, name):
        df.loc[
            df['Date'].isin(dates),
            'HolidayType'
        ] = holiday_type

    df['HolidayType'] = -1
    
    holiday_list = [
        (['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'], 'Super_Bowl'),
        (['2010-09-10','2011-09-09', '2012-09-07', '2013-09-06'], 'Labor_Day'),
        (['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'], 'Thanksgiving'),
        (['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'], 'Christmas')
    ]
    
    for index in range(0, len(holiday_list)):
        holiday = holiday_list[index]
        create_holiday_type_column(df, holiday[0], index, holiday[1])
    
    for x in df:
        if df[x].dtypes == "int64":
            df[x] = df[x].astype(float)
            
def holiday_count(train):
    dates =[]
    count = 0
    for ptr in holidays.US(years = int(train['Year'][0])).items():
        dates.append(ptr[0])
    dt=[]
    for i in range(0,5):
        dt.append(train['Date'][0] - datetime.timedelta(days = i))
    for i in range(1,3):
        dt.append(train['Date'][0] + datetime.timedelta(days = i))
    for date in dates:
        if date in dt:
            count +=1
    train['Holidays'] = np.array([count])
    
@model.route('/')
def index():
    return render_template('walmart.html')

@model.route('/predict',methods=['POST'])
def predict(): 
    stores=pd.read_csv("stores.csv")
    train = pd.DataFrame()
    train['Store'] = np.array([int(request.form['Store'])])
    train['Dept'] = np.array([int(request.form['dept'])])
    train['Date'] = np.array([request.form['Date']])
    train['IsHoliday'] = np.array([bool(request.form['IsHoliday'])])
    create_Holiday_Type(train)
    train.Date = pd.to_datetime(train.Date)
    train['Year']=train['Date'].dt.year
    train['Month']=train['Date'].dt.month
    train['Week']=train['Date'].dt.week
    train['Day']=train['Date'].dt.day
    train['n_days']=(train['Date'].dt.date-train['Date'].dt.date.min()).apply(lambda x:x.days)
    train = train.merge(stores, on = ['Store'], how = 'inner')
    holiday_count(train)
    data = train[['Store', 'Size', 'Dept', 'Month','Type', 'Year','Week', 'Day','n_days' ,'IsHoliday','Holidays','HolidayType']]
    data['Type'] = data['Type'].astype('category')
    data['Type'] = data['Type'].cat.codes
    rf = joblib.load(r'C:\Users\Dell\Downloads\walmart\Model_File.pkl')
    pred = rf.predict(data)
    
    return render_template('sale.html', variable=pred)


    
    
if __name__ == '__main__':
    model.run(debug=True)

