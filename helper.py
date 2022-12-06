def getColNames():
    dic = {}
    dic["Machine_Id"] = "Machine_Id"
    dic["Time_Stamp"] = "Time_Stamp"
    #dic["Setting"] = "Setting_"
    dic["SensorData"] = "SensorData_"
    dic["RUL"] = "RUL"

    dic["index_names"] = [dic["Machine_Id"], dic["Time_Stamp"]]
    #dic["settingNames"] = [dic["Setting"] + str(i) for i in range(4)]
    dic["sensorData"] = [dic["SensorData"] + str(i) for i in range(20)]

    return dic

def getInterval(): return 25
def getFeatureCount(): return 20
"""
def fixArraySizes(arr, maxSize):
    for _ in range(len(arr), maxSize):
        arr.append(0)
"""