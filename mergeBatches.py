import pandas as pd

from getDataFromFile import getDataFromFile
from helper import getColNames
colNames = getColNames()

def mergeBatches(data):
    data["data"] = pd.DataFrame()
    data["sampleCount"] = 0
    #print()
    for fileName in data["files"]:
        #print("Getting batch data from " + fileName)
        intial = [data["sampleCount"], len(data["data"])]
        if not data["data"].shape[0]: data["data"] = getDataFromFile(fileName, data["sampleCount"])
        else: data["data"] = pd.concat([data["data"], getDataFromFile(fileName, data["sampleCount"])], axis=0)
        data["sampleCount"] = data["data"][colNames["Machine_Id"]].unique().size
        #print("\tNo. of new samples = " + str(data["sampleCount"]-intial[0]))
        #print("\tNo. of new sensory data items = " + str(len(data["data"])-intial[1]))
        #print("\tTotal no. of samples = " + str(data["sampleCount"]))
        #print("\tTotal no. of sensory data items = " + str(len(data["data"])))
        #print()
    return data