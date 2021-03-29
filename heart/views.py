from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def home(request):
    return render(request,'index.html',{"predicted":""})

def predict(request):
    exp1 = int(request.GET['exp1'])
    exp2 = int(request.GET['exp2'])
    exp3 = int(request.GET['exp3'])
    rawdata = staticfiles_storage.path('housing_prices_MLR(1).csv')
    dataset = pd.read_csv(rawdata)
    X = dataset[["AREA","FLOOR","ROOM"]]
    y = dataset["PRICE"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42)
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    yet_to_predict = np.array([[exp1, exp2, exp3]])
    y_pred = regressor.predict(yet_to_predict)
    accuracy = regressor.score(X_test, y_test)
    accuracy = accuracy*100
    accuracy = int(accuracy)
    return render(request,'index.html',{"predicted":y_pred[0],"exp1":exp1,"exp2":exp2,"exp3":exp3})

  
