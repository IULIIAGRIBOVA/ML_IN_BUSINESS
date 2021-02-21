from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import accuracy_score
import requests
import os
import dill
import pandas as pd

import urllib.request
import json


path = 'app/models/'
path2='app/'


# Загружаем обученные модели
print(os.curdir)
with open(path+"logreg_pipeline.dill", 'rb') as in_strm:
    model = dill.load(in_strm)
X_test = pd.read_csv(path2+"X_test.csv")
y_test = pd.read_csv(path2+"y_test.csv")

y_test["Attrition"] =  y_test["Attrition"].apply(lambda p: 1 if p=='Yes' else 0)

# формируем запрос
def send_json(x):
    body = {'Age': x['Age'],
            'DailyRate': x['DailyRate'],
            'DistanceFromHome': x['DistanceFromHome'],
            'Education': x['Education'],
            'EmployeeCount': x['EmployeeCount'],
            'EmployeeNumber':x['EmployeeNumber'],
            'EnvironmentSatisfaction': x['EnvironmentSatisfaction'],
            'HourlyRate': x['HourlyRate'],
            'JobInvolvement': x['JobInvolvement'],
            'JobLevel': x['JobLevel'],
            'JobSatisfaction': x['JobSatisfaction'],
            'MonthlyIncome': x['MonthlyIncome'],
            'MonthlyRate': x['MonthlyRate'],
            'NumCompaniesWorked': x['NumCompaniesWorked'],
            'PercentSalaryHike': x['PercentSalaryHike'],
            'PerformanceRating': x['PerformanceRating'],
            'RelationshipSatisfaction': x['RelationshipSatisfaction'],
            'StandardHours': x['StandardHours'],
            'StockOptionLevel': x['StockOptionLevel'],
            'TotalWorkingYears': x['TotalWorkingYears'],
            'TrainingTimesLastYear': x['TrainingTimesLastYear'],
            'WorkLifeBalance': x['WorkLifeBalance'],
            'YearsAtCompany': x['YearsAtCompany'],
            'YearsInCurrentRole': x['YearsInCurrentRole'],
            'YearsSinceLastPromotion': x['YearsSinceLastPromotion'],
            'YearsWithCurrManager': x['YearsWithCurrManager'],
            'Attrition': x['Attrition'],
            'BusinessTravel':x['BusinessTravel'],
            'Department': x['Department'],
            'EducationField': x['EducationField'],
            'Gender': x['Gender'],
            'JobRole': x['JobRole'],
            'MaritalStatus':x['MaritalStatus'],
            'Over18': x['Over18'],
            'OverTime': x['OverTime']
            }

    myurl = 'http://127.0.0.1:5000/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(myurl, json=body, headers=headers)
    return response.json()['predictions']



if __name__ == '__main__':
    N=50
    predictions = X_test.iloc[:N, :].apply(lambda x: send_json(x), 1)
    print('предсказание',predictions)