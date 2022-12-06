import warnings 
warnings.filterwarnings('ignore')

import sys
basePath = "C:/Users/Raider/Downloads/IIT ISM Dhanbad/project/lstm from scratch 2"
if basePath not in sys.path: sys.path.append(basePath)

from helper import getColNames, getFeatureCount
colNames = getColNames()
from getXy import getXy
from mergeBatches import mergeBatches
from featureScaling import featureScaling

#Load and transform data
training, test = {}, {}
basePath = "C:/Users/Raider/Downloads/IIT ISM Dhanbad/project/"
training["files"] = [basePath + "Data/NASA Damage Propogation Modeling/train_FD00" + str(i) + ".txt" for i in range(1,5)]
test["files"] = [basePath + "Data/NASA Damage Propogation Modeling/test_FD00" + str(i) + ".txt" for i in range(1,5)]

#Merge multiple batches
training = mergeBatches(training)
test = mergeBatches(test)

#Perform feature scaling
training, test = featureScaling(training, test)

import numpy as np
df = training["data"]
arr = []
for i in range(1,1+training["sampleCount"]): arr.append(df[df["Machine_Id"]==i].count()[0])
data = np.array(arr)

# calculate quartiles
quartiles = np.percentile(data, [25, 50, 75])

IQR = quartiles[2]-quartiles[0]
outlierLowerLimit = np.count_nonzero(data < quartiles[0]-1.5*IQR)
outlierUpperLimit = np.count_nonzero(data > quartiles[1]+1.5*IQR)

# calculate min/max
data_min, data_max = data.min(), data.max()

"""
# print 5-number summary
print("\n\t5-number summary")
print('\tMin: %.3f' % data_min)
print('\tQ1: %.3f' % quartiles[0])
print('\tMedian: %.3f' % quartiles[1])
print('\tQ3: %.3f' % quartiles[2])
print('\tMax: %.3f' % data_max)
print("\n")
print("\tLower fence: " + str(quartiles[0]-1.5*IQR))
print("\tUpper fence: " + str(quartiles[1]+1.5*IQR))
print("\tMachines outside outlier Lower Limit = " + str(outlierLowerLimit))
print("\tMachines outside outlier Upper Limit = " + str(outlierUpperLimit))
print("\tOutlier perentage in data: " + str((100*outlierLowerLimit+outlierUpperLimit)/training["sampleCount"]))
"""

import matplotlib.pyplot as plt

maxMachinLife = training["data"][colNames["Time_Stamp"]].max()
x, y, y2 = [], [], []
x1, y1, y12 = [], [], []
for i in range(maxMachinLife):
    val = training["data"][colNames["Time_Stamp"]].value_counts()[i+1]
    if val < outlierUpperLimit and val > outlierLowerLimit:
        x.append(i+1)
        y.append(val/training["sampleCount"])
        y2.append(val)
    else:
        x1.append(i+1)
        y1.append(val/training["sampleCount"])
        y12.append(val)
#print(y[0], y1[0])

"""
fig, ax1 = plt.subplots()
#ax1.yaxis.tick_right()
plt.plot(x, y2, color = 'red')
plt.plot(x1, y12, color = 'blue')
for i in range(1,10): plt.plot([i*maxMachinLife/10 for j in range(training["sampleCount"])], [j for j in range(training["sampleCount"])], color = 'black')
#plt.xlabel('Time')
#plt.yticks(np.arange(0, 1.1, 0.10))
#plt.ylabel('No. of working machine')
#plt.title('Creating time slots')

ax2 = ax1.twinx()
plt.plot(x, y, color = 'red')
plt.plot(x1, y1, color = 'blue')
color = 'tab:green'
ax2.set_ylabel('Probability of survival')
#ax2.tick_params(axis ='y')
plt.show()

"""
#Add y
training["data"]["y"] = training["data"][colNames["RUL"]]/maxMachinLife
test["data"]["y"] = test["data"][colNames["RUL"]]/maxMachinLife

"""
dic = {}
for i in range(5): 
    dic[i] = (y1[(i)*maxMachinLife//10], y1[(i+1)*maxMachinLife//10-1], 
                             (y1[(i)*maxMachinLife//10]+y1[(i+1)*maxMachinLife//10-1])/2)
for i in range(4): 
    dic[5+i] = (y[(i)*maxMachinLife//10], y[(i+1)*maxMachinLife//10-1], 
                (y[(i)*maxMachinLife//10]+y[(i+1)*maxMachinLife//10-1])/2)
dic[9] = (y[(i)*maxMachinLife//10], 0, y[(i)*maxMachinLife//10]//2)
"""

X_train = training["data"][colNames["sensorData"]].values
y_train = training["data"]["y"].values
"""
for val in training["data"]["y"].values:
    for i in range(10):
        if val < dic[i][0] and val >= dic[i][1]:
            y_train.append(dic[i][2])
            break
    else: y_train.append(0)
"""
X_test = test["data"][colNames["sensorData"]].values
y_test = test["data"]["y"].values
"""
for val in test["data"]["y"].values:
    for i in range(10):
        if val < dic[i][0] and val >= dic[i][1]:
            y_test.append(dic[i][2])
            break
    else: y_test.append(0)
"""
from sklearn.linear_model import LinearRegression
import numpy as np
print("\nLinearRegression")
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test.reshape(104897,-1))

from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred = regressor.predict(X_test)
print("\tmean absolute error: "+str(mean_absolute_error(y_test, y_pred)))
#print("\tmean squared error: "+str(mean_squared_error(y_test,y_pred)))

#from sklearn.metrics import r2_score
#print("r2_score: "+str(r2_score(y_test, y_pred)))

print("\nDecision Tree Regression")
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test.reshape(104897,-1))

print("\tmean absolute error: "+str(mean_absolute_error(y_test, y_pred)))
#print("\tmean squared error: "+str(mean_squared_error(y_test,y_pred)))
#print("r2_score: "+str(r2_score(y_test, y_pred)))

print("\nRandom Forest Regression")
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test.reshape(104897,-1))

print("\tmean_absolute_error: "+str(mean_absolute_error(y_test, y_pred)))
#print("\tmean squared error: "+str(mean_squared_error(y_test,y_pred)))
#print("r2_score: "+str(r2_score(y_test, y_pred)))

"""
print("SVR")
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test.reshape(104897,-1))

print("mean_absolute_error: "+str(mean_absolute_error(y_test, y_pred)))
print("r2_score: "+str(r2_score(y_test, y_pred)))

y_pred = regressor.predict(X_test)
#for i in range(5): print(y_test[i], y_pred[i])
print(y_test[0], y_pred[0])

import matplotlib.pyplot as plt

maxMachinLife = training["data"][colNames["Time_Stamp"]].max()
x, y, y2 = [], [], []
for i in range(maxMachinLife):
    x.append(i+1)
    y2.append(training["data"][colNames["Time_Stamp"]].value_counts()[i+1])
    y.append(training["data"][colNames["Time_Stamp"]].value_counts()[i+1]/(training["sampleCount"]))

#print("Total Training samples: " + str(training["sampleCount"]))
#print("Total Testing samples: " + str(test["sampleCount"]))

featureCount = getFeatureCount()

fig, ax1 = plt.subplots()
ax1.yaxis.tick_right()
plt.plot(x, y2, color = 'blue')
plt.xlabel('Time')
#plt.yticks(np.arange(0, 1.1, 0.10))
plt.ylabel('No. of working machine')
plt.title('Machine decay Graph')

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Probability of survival')
ax2.plot(x, y)
ax2.tick_params(axis ='y')

plt.show()

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=x, event_observed=y)
kmf.plot_survival_function()

#print("Information regarding each sensor data in training data")
#print(training["data"][colNames["sensorData"]].info())

#print("Information regarding each sensor data in test data")
#print(test["data"][colNames["sensorData"]].info())

#XTrain, yTrain, maxLenTrain = getXy(training)
#XTest, yTest, maxLenTest = getXy(test)

#print(XTrain, yTrain, maxLenTrain)
#print(XTest, yTest, maxLenTest)

import tensorflow as tf

def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
    return mse, mae

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
"""