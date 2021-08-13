import pandas as pd
import numpy as np
from eda import *
from data_prep import *
df=pd.read_csv("hitters.csv")
check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_summary(df, "Salary")
for col in num_cols:
    num_summary(df, col, plot=False)

for col in num_cols:
    target_summary_with_num(df, "Salary", col)

                        ## özellik-çıkarım-çabam ##
# Tüm vuruşlara göre topa isabetli vurma oranı
df["succes_hits_rate"] = df["Hits"]/df["AtBat"]
# Tüm vuruşlara göre en değerli vuruş oranı
df["HmRun_rate"] = df["HmRun"]/df["AtBat"]
# Bir oyuncunun koşturduğu oyuncu sayısına göre yaptırdığı hata sayısı
df["Walks_according_to_RBI"] = df["Walks"]/df["RBI"]
# Oyuncunun kariyeri boyunca yaptığı isabetli vuruşun, tüm vuruşa oranı
df["succes_hits_rate_all_life"] = df["CHits"]/df["CAtBat"]
# Oyuncunun en değerli sayılarının tüm değerli sayılara oranı
 df["CHmRun_according_to_HmRun"] = df["CHmRun"]/df["HmRun"]
# Oyuncunun yıllara göre ortalama attığı sayı
df["CRuns_according_to_Years"] = df["CRuns"]/df["Years"]
# 1986-1987 yılları arasında oyuncunun verimi = 3* sayı + 2* asist / 1* hata_sayısı
df["Efficiency"] = (3*df["Runs"] + 2*df["Assits"] ) / df["Errors"]



# eksik değer
missing_values_table(df)
# aykırı değer
for col in num_cols:
    print(col, check_outlier(df, col))

grab_outliers(df, "Salary")

# aykırı değerlerin silinmesi
replace_with_thresholds(df, "Salary")

#encode

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
for col in binary_cols:
    dataframe = label_encoder(df, col)
# one_hot
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)


#model
x = df.drop(["Salary"], axis=1)
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.33, random_state=12)
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# ACCURACY
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# PRECISION
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)

# RECALL
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)

# F1
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
