import sys
basePath = "C:/Users/Raider/Downloads/IIT ISM Dhanbad/project/lstm from scratch 2"
if basePath not in sys.path: sys.path.append(basePath)

from helper import getColNames, getInterval

colNames = getColNames()
interval = getInterval()

def add_remaining_useful_life(df):
    # Get number of cycles for each machine
    max_cycle_for_each_machine = df.groupby(by=colNames["Machine_Id"])[colNames["Time_Stamp"]].max()

    # Merge max cycles into each row of data
    result_frame = df.merge(max_cycle_for_each_machine.to_frame(name="max_cycle"), 
                            left_on=colNames["Machine_Id"], 
                            right_index=True)

    # Calculate and add remaining useful life for each row
    result_frame[colNames["RUL"]] = result_frame["max_cycle"] - result_frame[colNames["Time_Stamp"]]
    #print("Added RUL(Remaining uselful time)")

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    #print(result_frame.head())
    return result_frame