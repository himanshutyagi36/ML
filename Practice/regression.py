# Courtesy: pythonprogramming.net

import pandas as pd
import quandl
import math, datetime, time
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
# print(df.head())

## Feature selection
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# print(df.head())
forecast_col = 'Adj. Close'
## to fill empty data points; filling this way, it will be treated as outlier
df.fillna(-99999, inplace=True)

## here we are trying to predict out 10% of data.
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

## creating labels
df['label'] = df[forecast_col].shift(-forecast_out)

# print (df.head())


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

## scale the feature values in the range (-1,1) using scikit learn preprocessing module

## scale X ; normalize

## split the data into training and test sets, 80/20 ratio.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# clf = LinearRegression()
# clf = svm.SVR(kernel= 'poly')
# clf.fit(X_train,y_train)

# # use pickle to save the trained classifier
# with open('pickles/lr.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# load the saved model
pickle_in = open('pickles/lr.pickle','rb')
clf = pickle.load(pickle_in)

## for linear regression, accuracy is squared error
accuracy = clf.score(X_test,y_test)

#print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)


df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]

print (df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
