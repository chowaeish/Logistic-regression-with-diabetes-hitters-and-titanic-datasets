import pandas as pd
import numpy as np
from eda import *
from data_prep import *
df=pd.read_csv("diabetes.csv")
check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_summary(df, "Outcome")
for col in num_cols:
    num_summary(df, col, plot=False)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# yeni_özellik_ekleme_çabam
df.columns = [col.upper() for col in df.columns]
df.loc[(df["AGE"] >= 21) & (df["AGE"] < 40), "AGE_CAT"] = 0
df.loc[(df["AGE"] >= 40) & (df["AGE"] < 60), "AGE_CAT"] = 1
df.loc[(df["AGE"] >= 60) & (df["AGE"] <= 81), "AGE_CAT"] = 2
df["AGE_CAT"].head(10)
df["GLUCOSE_INSULIN"]=df["GLUCOSE"]*df["INSULIN"]
df["GLUCOSE_INSULIN"].head()

# eksik değer
missing_values_table(df)
# aykırı değer
for col in num_cols:
    print(col, check_outlier(df, col))

grab_outliers(df, "INSULIN")

# aykırı değerlerin silinmesi
replace_with_thresholds(df, "INSULIN")
#model
x = df.drop(["OUTCOME"], axis=1)
y = df["OUTCOME"]

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



