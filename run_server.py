from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import accuracy_score
import requests
import os
import dill
import pandas as pd

import urllib.request
import json


path = 'app/models/'
path2='/'

#!pip install -r requirements. txt
# Загружаем обученные модели

with open("path"+"logreg_pipeline.dill", 'rb') as in_strm:
    model = dill.load(in_strm)
#X_test = pd.read_csv(path2+"X_test.csv")
#y_test = pd.read_csv(path2+"y_test.csv")

#y_test["Attrition"] =  y_test["Attrition"].apply(lambda p: 1 if p=='Yes' else 0)

app = Flask(__name__)

@app.route("/")
def hello():
    return "hello"

@app.route('/predict',  methods=['GET', 'POST'])
def predict():
    data = {"success": False}

    if request.method == "POST":
                request_json = request.get_json()

                Age = request_json['Age']
                DailyRate = request_json['DailyRate']
                DistanceFromHome = request_json['DistanceFromHome']
                Education = request_json['Education']
                EmployeeCount = request_json['EmployeeCount']
                EmployeeNumber = request_json['EmployeeNumber']
                EnvironmentSatisfaction = request_json['EnvironmentSatisfaction']
                HourlyRate = request_json['HourlyRate']
                JobInvolvement = request_json['JobInvolvement']
                JobLevel = request_json['JobLevel']
                JobSatisfaction = request_json['JobSatisfaction']
                MonthlyIncome = request_json['MonthlyIncome']
                MonthlyRate = request_json['MonthlyRate']
                NumCompaniesWorked = request_json['NumCompaniesWorked']
                PercentSalaryHike = request_json['PercentSalaryHike']
                PerformanceRating = request_json['PerformanceRating']
                RelationshipSatisfaction = request_json['RelationshipSatisfaction']
                StandardHours = request_json['StandardHours']
                StockOptionLevel = request_json['StockOptionLevel']
                TotalWorkingYears = request_json['TotalWorkingYears']
                TrainingTimesLastYear = request_json['TrainingTimesLastYear']
                WorkLifeBalance = request_json['WorkLifeBalance']
                YearsAtCompany = request_json['YearsAtCompany']
                YearsInCurrentRole = request_json['YearsInCurrentRole']
                YearsSinceLastPromotion = request_json['YearsSinceLastPromotion']
                YearsWithCurrManager = request_json['YearsWithCurrManager']
                Attrition = request_json['Attrition']
                BusinessTravel = request_json['BusinessTravel']
                Department = request_json['Department']
                EducationField = request_json['EducationField']
                Gender = request_json['Gender']
                JobRole = request_json['JobRole']
                MaritalStatus = request_json['MaritalStatus']
                Over18 = request_json['Over18']
                OverTime = request_json['OverTime']

                X = pd.DataFrame({
                    "Age": [Age],
                    "DailyRate": [DailyRate],
                    "DistanceFromHome": [DistanceFromHome],
                    "Education": [Education],
                    "EmployeeCount": [EmployeeCount],
                    "EmployeeNumber": [EmployeeNumber],
                    "EnvironmentSatisfaction": [EnvironmentSatisfaction],
                    "HourlyRate": [HourlyRate],
                    "JobInvolvement": [JobInvolvement],
                    "JobLevel": [JobLevel],
                    "JobSatisfaction": [JobSatisfaction],
                    "MonthlyIncome": [MonthlyIncome],
                    "MonthlyRate": [MonthlyRate],
                    "NumCompaniesWorked": [NumCompaniesWorked],
                    "PercentSalaryHike": [PercentSalaryHike],
                    "PerformanceRating": [PerformanceRating],
                    "RelationshipSatisfaction": [RelationshipSatisfaction],
                    "StandardHours": [StandardHours],
                    "StockOptionLevel": [StockOptionLevel],
                    "TotalWorkingYears": [TotalWorkingYears],
                    "TrainingTimesLastYear": [TrainingTimesLastYear],
                    "WorkLifeBalance": [WorkLifeBalance],
                    "YearsAtCompany": [YearsAtCompany],
                    "YearsInCurrentRole": [YearsInCurrentRole],
                    "YearsSinceLastPromotion": [YearsSinceLastPromotion],
                    "YearsWithCurrManager": [YearsWithCurrManager],
                    "Attrition": [Attrition],
                    "BusinessTravel": [BusinessTravel],
                    "Department": [Department],
                    "EducationField": [EducationField],
                    "Gender": [Gender],
                    "JobRole": [JobRole],
                    "MaritalStatus": [MaritalStatus],
                    "Over18": [Over18],
                    "OverTime": [OverTime]})
                preds = model.predict(X)
                #data["accuracy"] = accuracy_score(y_test, preds)
                predictions_mas = int(preds)
                data["predictions"] = predictions_mas
                data["success"] = True
    return jsonify(data)


if __name__ == '__main__':
    app.run()
