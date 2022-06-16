from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
import requests
import datetime

app = Flask(__name__)

train_df = pd.read_csv('mysore_data.csv')

train_df['day']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.day
train_df['month']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.month
train_df['year']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.year

train_df.drop(['Date'],axis=1,inplace=True)

## filling mean values
def fill_mean(data):#replacing null fields with the mean value of the respective filed
    null_fields = data.isna().sum()
    col = data.columns #storing the column names
    x = 0
    for i in null_fields:
        if i != 0:
            data = data.fillna({col[x]:data[col[x]].mean()})# replaces null field with mean of column values
        x += 1     
    return data

train_df = fill_mean(train_df)

## handling outliers

def detect_outliers_iqr(data):
    outliers = []
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers

def collect_outliers_iqr(data):
    outliers_detected_iqr = {}
    for i in data.columns:
        outliers = detect_outliers_iqr(data[i])
        outliers_detected_iqr[i] = outliers
    return outliers_detected_iqr


def floor_clapp_outliers(data, outliers):
    for i, j in outliers.items():
        if len(outliers[i]) != 0:
            IQR = data[i].quantile(0.75) - data[i].quantile(0.25)
            lower_bridge = data[i].quantile(0.25) - (IQR*1.5)
            upper_bridge = data[i].quantile(0.75) + (IQR*1.5)
            data.loc[data[i] > upper_bridge, i] = upper_bridge
            data.loc[data[i] < lower_bridge, i] = lower_bridge
    return data

outliers = collect_outliers_iqr(train_df)
train_df = floor_clapp_outliers(train_df, outliers)

train_df=train_df.round(decimals=3)

X = train_df[['Temperature', 'Humidity', 'Gas', 'CO', 'NH3',
       'day', 'month', 'year']]
y = train_df['PM 2.5 (ug/m3)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators = 150,random_state = 0)
rfc.fit(X_train, y_train)

import math
y_pred = rfc.predict(X_test)
y_pred=y_pred.round(decimals=3)
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test,y_pred))
print(math.sqrt(mean_squared_error(np.array(y_test),y_pred)))

import pickle
file = open('air_pollution.pkl', 'wb')
pickle.dump(rfc, file)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/developer")
def developers():
    return render_template("developer.html")

@app.route("/predict_pm", methods = ["GET", "POST"])
def predict_pm():
    if request.method == "POST":

        # Date
        Date = request.form["Date"]
        day=int(pd.to_datetime(Date,format="%Y-%m-%d").day)
        month=int(pd.to_datetime(Date,format="%Y-%m-%d").month)
        year = int(pd.to_datetime(Date,format="%Y-%m-%d").year)

        PM10=request.form['PM10']

        NO=request.form['NO']

        NO2=request.form['NO2']

        NOx=request.form['NOx']

        NH3=request.form['NH3']

        CO=request.form['CO']

        SO2=request.form['SO2']

        O3=request.form['O3']

        Benzene=request.form['Benzene']

        Toluene=request.form['Toluene']

        Xylene=request.form['Xylene']
        
        prediction=rfc.predict([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
        Benzene, Toluene, Xylene,day,month,year
        ]])
        #print(prediction)

        output=round(prediction[0],3)
        #print(output)
        def air_condition(pm_conc):
            if pm_conc<=12.0:
                aqi = "AQI level is Between 0 to 50 i.e Good Weather Conditions"
            elif pm_conc>12.1 and pm_conc<=35.4:
                aqi = "AQI level is Between 51 to 100 i.e Moderate Weather Conditions"
            elif pm_conc>35.5 and pm_conc<=55.4:
                aqi = "AQI level is Between 101 to 150 i.e Unhealthy Weather Conditions for Older Age Groups"
            elif pm_conc>55.5 and pm_conc<=150.4:
                aqi = "AQI level is Between 151 to 200 i.e Unhealthy Weather Conditions"
            elif pm_conc>150.5 and pm_conc<=250.4:
                aqi = "AQI level is Between 201 to 300 i.e Very Unhealthy Weather Conditions"
            else:
                aqi = "AQI level is Above 300 i.e Hazardous Weather Conditions"
            return aqi

        aqi = air_condition(output)
        print(aqi)
        return render_template('prediction.html',prediction_text="PM2.5 level is {} and {}".format(output,aqi))

    return render_template("home.html")


@app.route("/auto_prediction")
def predict_auto_pm():
    now = datetime.datetime.now()
    msg = requests.get("https://thingspeak.com/channels/935349/feed.json")

    PM10=str(msg.json()['feeds'][-1]['field1'])

    NO=str(msg.json()['feeds'][-1]['field1'])

    NO2=str(msg.json()['feeds'][-1]['field1'])

    NOx=str(msg.json()['feeds'][-1]['field1'])

    NH3=str(msg.json()['feeds'][-1]['field1'])

    CO=str(msg.json()['feeds'][-1]['field1'])

    SO2=str(msg.json()['feeds'][-1]['field1'])

    O3=str(msg.json()['feeds'][-1]['field1'])

    Benzene=str(msg.json()['feeds'][-1]['field1'])

    Toluene=str(msg.json()['feeds'][-1]['field1'])

    Xylene=str(msg.json()['feeds'][-1]['field1'])
        
    prediction=rfc.predict([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
        Benzene, Toluene, Xylene,now.day,now.month,now.year
        ]])

    output=round(prediction[0],3)
    def air_condition(pm_conc):
            if pm_conc<=12.0:
                aqi = "AQI level is Between 0 to 50 i.e Good Weather Conditions"
            elif pm_conc>12.1 and pm_conc<=35.4:
                aqi = "AQI level is Between 51 to 100 i.e Moderate Weather Conditions"
            elif pm_conc>35.5 and pm_conc<=55.4:
                aqi = "AQI level is Between 101 to 150 i.e Unhealthy Weather Conditions for Older Age Groups"
            elif pm_conc>55.5 and pm_conc<=150.4:
                aqi = "AQI level is Between 151 to 200 i.e Unhealthy Weather Conditions"
            elif pm_conc>150.5 and pm_conc<=250.4:
                aqi = "AQI level is Between 201 to 300 i.e Very Unhealthy Weather Conditions"
            else:
                aqi = "AQI level is Above 300 i.e Hazardous Weather Conditions"
            return aqi

    aqi = air_condition(output)
    return render_template('prediction_auto.html',prediction_text="PM2.5 level is {} and {}".format(output,aqi))


if __name__ == '__main__':
    app.run(debug=True)