import pandas as pd;
import sys

basePath = "C:/Users/Raider/Downloads/IIT ISM Dhanbad/project/lstm from scratch 2"
if basePath not in sys.path: sys.path.append(basePath)

from addRemainingUsefulLifeInDF import add_remaining_useful_life
from helper import getColNames

def getDataFromFile(file_name, machineCount):
    data = pd.read_csv(file_name,sep=" ",header=None)

    #Drop useless data columns
    data.drop(columns=[26,27],inplace=True)
    data.drop(columns=[2,3,4,5],inplace=True) #settings

    #Assign columns user friendly names
    colNames = getColNames()
    data.columns = colNames["index_names"] + colNames["sensorData"]
    #data.columns = colNames["index_names"] + colNames["settingNames"] + colNames["sensorData"]

    #Fixed machine id for different batches
    if machineCount != 0: data[colNames["Machine_Id"]] = data["Machine_Id"] + machineCount
    
    #Caculate and add RUL to data
    return add_remaining_useful_life(data)