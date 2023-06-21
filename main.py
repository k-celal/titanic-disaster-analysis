# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:04:21 2023

@author: Celal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#veri yükleme ve ilk 5 satır
data=pd.read_csv("titanic.csv") 
print(data.head())
#veri setinin bilgileri
print(data.info())

def show_table(data):
   print("Veri Kümesinin Şekli:")
   print(data.shape)

   print("Veri Kümesinin Türü:")
   print(data.dtypes)

   print("Veri Kümesinin Son 5 Değeri:")
   print(data.tail())

   print("Veri Kümesinin İlk 5 Değeri:")
   print(data.head())

   print("Veri Kümesindeki Kayıp Değerler:")
   print(data.isnull().sum())
   
   print("Veri Kümesinin Tanımlayıcı İstatistikleri:")
   print(data.describe())
   
show_table(data)
print("Veri Kümesindeki eksik Değerler:")
print(data.isnull().sum())

age_mean=data['age'].mean()
print("Yaş Sütununun Ortalaması:", age_mean)

sb.boxplot(data['age'])
plt.show()
print("Yaş Değişkeninin Tanımlayıcı İstatistikleri:")
print(data['age'].describe())


print("Ücret Değişkeninin Tanımlayıcı İstatistikleri:")
print(data['fare'].describe())

data[['age','fare']].hist()
plt.show()
print("Pclass Sütununda Toplam Sınıf Sayısı:")
print(len(data['pclass'].unique()))
print("Pclass Sütununun Türü:")
print(data['pclass'].dtype)

categorical_variable=[]
numeric_variable=[]
for i in data.columns:
    if data[i].dtype in ['int64','float64']:
        numeric_variable.append(i)
    else:
        categorical_variable.append(i)

print(categorical_variable)
print(numeric_variable)

sb.barplot(data=data, x=data["survived"], y=data["survived"].value_counts());
plt.show()
print("Kadın ve erkeklerin hayatta kalma oranı:")
print(data.groupby("sex")["survived"].mean())
data_new = data.loc[(data["age"] > 50) & (data["sex"] == "male")
 & ((data["embark_town"] == "Cherbourg") | (data["embark_town"] == "Southampton")),
 ["age", "class", "embark_town"]]
print(data_new.head())
sb.barplot(data_new,x=data_new['age'],y=data_new['embark_town'])
plt.show()
from collections import Counter
def detect_outlier(data,features):
    outlier_indices = []
    
    for feature in features:
        # first quartile
        q1 = np.percentile(data[feature],25)
        # third quartile
        q3 = np.percentile(data[feature],75)
        # IQR
        IQR = q3 - q1
        outlier_step = IQR * 1.5
        outliers_ = data[(data[feature] < q1 - outlier_step) | (data[feature] > q3 - outlier_step)].index
        outlier_indices.extend(outliers_)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, k in outlier_indices.items() if k > 2)
    return multiple_outliers
dropped_outlier = data.loc[detect_outlier(data, ["age", "fare", "sibsp", "parch"])]
print(dropped_outlier)

data = data.drop(detect_outlier(data, ["age", "fare", "sibsp", "parch"]), axis=0).reset_index(drop=True)
print(data)

print(data.isnull().sum())

data["embarked"] = data["embarked"].fillna("C")
print(data.isnull().sum())

correlation_list = ["age", "fare", "sibsp", "parch", "survived"]
sb.heatmap(data[correlation_list].corr(), annot=True);
plt.show()
sb.regplot(["age", "fare"], data=data, robust=True, ci=None, color="seagreen");