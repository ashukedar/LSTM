import sys

basePath = "C:/Users/Raider/Downloads/IIT ISM Dhanbad/project/lstm from scratch 2"
if basePath not in sys.path: sys.path.append(basePath)

from helper import getColNames
colNames = getColNames()

def featureScaling(training, test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    training["data"][colNames["sensorData"]] = sc.fit_transform(training["data"][colNames["sensorData"]])
    test["data"][colNames["sensorData"]] = sc.transform(test["data"][colNames["sensorData"]])
    return training, test