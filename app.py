from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

train_df = pd.read_csv('city_day.csv/city_day.csv')

train_df['PM2.5'].fillna(np.mean(train_df['PM2.5']),inplace=True)
train_df['PM10'].fillna(np.mean(train_df['PM10']),inplace=True)
train_df['NO'].fillna(np.mean(train_df['NO']),inplace=True)
train_df['NO2'].fillna(np.mean(train_df['NO2']),inplace=True)
train_df['NOx'].fillna(np.mean(train_df['NOx']),inplace=True)
train_df['NH3'].fillna(np.mean(train_df['NH3']),inplace=True)
train_df['CO'].fillna(np.mean(train_df['CO']),inplace=True)
train_df['SO2'].fillna(np.mean(train_df['SO2']),inplace=True)
train_df['O3'].fillna(np.mean(train_df['O3']),inplace=True)
train_df['Benzene'].fillna(np.mean(train_df['Benzene']),inplace=True)
train_df['Toluene'].fillna(np.mean(train_df['Toluene']),inplace=True)
train_df['Xylene'].fillna(np.mean(train_df['Xylene']),inplace=True)
train_df['AQI'].fillna(np.mean(train_df['AQI']),inplace=True)

train_df['day']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.day
train_df['month']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.month
train_df['year']=pd.to_datetime(train_df.Date,format="%Y-%m-%d").dt.year

train_df.drop(['Date','City','AQI_Bucket'],axis=1,inplace=True)
train_df=train_df.round(decimals=3)

X = train_df[['PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
       'Benzene', 'Toluene', 'Xylene','day','month','year']]
y = train_df['PM2.5']

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
#model = pickle.load(open("employee_burnout_rf.pkl", "rb"))
#employee_b = pickle.load(model)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/employee_info")
def employee_info():
    return render_template("employee_info.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/sign")
def signs():
    return render_template("signs.html")

@app.route("/prevention")
def prevent():
    return render_template("prevent.html")

@app.route("/developer")
def developers():
    return render_template("developer.html")

@app.route("/burnout", methods = ["GET", "POST"])
def burnout():
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
        return render_template('employee_info.html',prediction_text="PM2.5 level is {}".format(output))


    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)