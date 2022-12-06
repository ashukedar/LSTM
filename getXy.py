import numpy as np

from helper import getColNames
colNames = getColNames()

def getXy(df):
    X, y, maxLen = [], [], 0
    for i in range(1, 1+df["sampleCount"]):
        temp = df["data"].copy(deep=True)
        temp.query(colNames["Machine_Id"] + ' == ' + str(i), inplace = True)
        X.append(np.array(temp[colNames["sensorData"]]))
        y.append(np.array(temp[colNames["RUL"]]))
        maxLen = max(maxLen, len(y))
    return np.array(X), np.array(y), maxLen