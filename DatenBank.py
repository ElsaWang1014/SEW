import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd

load_path = "/media/campus/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"

second = 2
rx_index = 1
scenario_index = 1
round_number = 77  

############    linspetrum    #############################################################################
filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
full_filename = os.path.join(load_path, filename)
if os.path.exists(full_filename):
  mat = scipy.io.loadmat(full_filename)
  #print(mat)
cir = mat["cirs"]
cir = 10 * np.log10(np.abs(cir))
cir_data = np.array(cir)
#print(mat["cirs"])
lin_spec = mat["linSpectrum"]
lin_spec = 10 * np.log10(np.abs(lin_spec)) 
lin_spec_data = np.array(lin_spec)
print("lin_spec_data:", lin_spec_data)

############      RF           ############################################################################
list_all_files = os.listdir(load_path)
list_relevant_files = []
for file in list_all_files:
            string_entries = file.split('_')
            rf_index = int(string_entries[5])
list_relevant_files.append(file)


############      Time         ############################################################################
excel_file = '/home/campus/Desktop/Datenbearbeitung_1/Messplan.xlsx'

if scenario_index == 1 :
    sheets = ['Scenario 1']
    columns = [3,5]

    row_ranges = [(8, 23), (25, 70), (73, 78)]

    Uhrzeit = {}

    for sheet in sheets:
        df_list = []
        for start_row, end_row in row_ranges:
           
            df_part = pd.read_excel(
                excel_file, 
                sheet_name=sheet, 
                usecols=columns, 
                skiprows=start_row, 
                nrows=end_row - start_row + 1,
                header=None
            )
            df_list.append(df_part)
        
    df_combined = pd.concat(df_list, ignore_index=True)
    Uhrzeit[sheet] = df_combined

elif scenario_index == 2 :
    sheets = ['Scenario 2']
    columns = [3,5]

    row_ranges = [(8, 53), (56, 60), (66, 78)]

    Uhrzeit = {}

    for sheet in sheets:
        df_list = []
        for start_row, end_row in row_ranges:
           
            df_part = pd.read_excel(
                excel_file, 
                sheet_name=sheet, 
                usecols=columns, 
                skiprows=start_row, 
                nrows=end_row - start_row + 1,
                header=None
            )
            df_list.append(df_part)
        
    df_combined = pd.concat(df_list, ignore_index=True)
    Uhrzeit[sheet] = df_combined

elif scenario_index == 3 :
    sheets = ['Scenario3']
    columns = [3,5]

    row_ranges = [(8, 31), (34, 78), (81, 90), (94, 113)]

    Uhrzeit = {}

    for sheet in sheets:
        df_list = []
        for start_row, end_row in row_ranges:
           
            df_part = pd.read_excel(
                excel_file, 
                sheet_name=sheet, 
                usecols=columns, 
                skiprows=start_row, 
                nrows=end_row - start_row + 1,
                header=None
            )
            df_list.append(df_part)
        
    df_combined = pd.concat(df_list, ignore_index=True)
    Uhrzeit[sheet] = df_combined

else  :
    sheets = ['Scenario4']
    columns = [3,5]

    row_ranges = [(17, 42), (45, 83), (87, 93), (100, 104)]

    Uhrzeit = {}

    for sheet in sheets:
        df_list = []
        for start_row, end_row in row_ranges:
           
            df_part = pd.read_excel(
                excel_file, 
                sheet_name=sheet, 
                usecols=columns, 
                skiprows=start_row, 
                nrows=end_row - start_row + 1,
                header=None                                    #definite the first line the part of the data
            )
            df_list.append(df_part)
        
    df_combined = pd.concat(df_list, ignore_index=True)
    Uhrzeit[sheet] = df_combined


for sheet, df in Uhrzeit.items():
        print(f"Data from {sheet}:")
        print(df)
        print()


column_3_values = df_combined.iloc[:, 0].astype(int).tolist()  #transfer to int , list
column_5_values = df_combined.iloc[:, 1].astype(str).tolist()  #transfer to str , list

    
round_time_map = dict(zip(column_3_values, column_5_values))   #zip : combines rounds and time into pairs  
                                                               #dict : pairs into a dictionary

time_for_round = round_time_map.get(round_number)              #get time using round number
time_parts = time_for_round.split(":")                         #seperate the time using :
time_for_round_formatted = f"{time_parts[0]}_{time_parts[1]}"  #hour combines min with _


############      Polarization         ############################################################################
if 1 <= round_number <= 4 or 160 <= round_number <= 167 or 309 <= round_number <= 317 :
     polarization = 'vertikal'
elif 5 <= round_number <= 10 or 166 <= round_number <= 176 or 318 <= round_number <= 325 :
     polarization = 'horizontal'
elif 77 <= round_number <= 82 or 131 <= round_number <= 136 or 232 <= round_number <= 242 or 375 <= round_number <= 382 :
     polarization = 'senkrecht_mit_Antenne_AGV_Horizontal'
else :
     polarization = 'senkrecht'
print(f"Polarization:{polarization}")


############      Position                    ############################################################################
def sew_track_function(t, scenario_index):

    if scenario_index == 1:
        if t>=0 and t<0.6:
            alpha = 34.4 + (180 / (np.pi * 0.68)) * 0.5 * t ** 2
            x = 15.25 + 0.68 * np.cos(np.deg2rad(alpha))
            y = 5.29 - 0.68 + 0.68 * np.sin(np.deg2rad(alpha))
        elif t>=0.6 and t<1.4:
            alpha = 90 - 40.44 + (180 / (np.pi * 0.68)) * 0.6 * (t - 0.6)
            x = 15.25 + 0.68 * np.cos(np.deg2rad(alpha))
            y = 5.29 - 0.68 + 0.68 * np.sin(np.deg2rad(alpha))
        elif t>=1.4 and t<=25:
            alpha = 0
            x = 15.25 - 0.6 * (t - 1.4)
            y = 5.29

    elif scenario_index == 2:
        if t>=0 and t<0.6:
            alpha = 90 - (180 / (np.pi * 0.78)) * 0.82 - (180 / (np.pi * 0.78)) * 0.5 * 0.6 ** 2 + (180 / (np.pi * 0.78)) * 0.5 * t ** 2
            x = 15.25 + 0.78 * np.cos(np.deg2rad(alpha))
            y = 3.4 - 0.78 * np.sin(np.deg2rad(alpha))
        elif t>=0.6 and t< (0.6+0.82/0.6):
            alpha = 90 - (180 / (np.pi * 0.78)) * 0.82 + (180 / (np.pi * 0.78)) * 0.6 * (t - 0.6)
            x = 15.25 + 0.78 * np.cos(np.deg2rad(alpha))
            y = 3.4 - 0.78 * np.sin(np.deg2rad(alpha))
        elif t>=0.6+0.82/0.6 and t<=25:
            alpha = 0
            x = 15.25 - 0.6 * (t - (0.6+0.82/0.6))
            y = 2.62

    elif scenario_index == 3:
        if t>=0 and t<0.6:
            alpha = 270 + 90 - (160 / (2 * np.pi * 72)) * 90 + (180 / (np.pi * 0.72)) * 0.5 * t ** 2
            x = -3.35 - 0.72 + 0.72 * np.cos(np.deg2rad(alpha))
            y = 11.26 + 0.72 * np.sin(np.deg2rad(alpha))
        elif t>=0.6 and t< (0.6+0.22/0.6):
            alpha = 270 + 90 - (160 / (2 * np.pi * 72)) * 90 + (180 / (np.pi * 0.72)) * 0.5 * 0.6 ** 2 + (180 / (np.pi * 0.72)) * 0.6 * (t - 0.6)
            x = -3.35 - 0.72 + 0.72 * np.cos(np.deg2rad(alpha))
            y = 11.26 + 0.72 * np.sin(np.deg2rad(alpha))
        elif t>=0.6+0.22/0.6 and t<=20:
            alpha = 0
            x = -3.35
            y = 11.26 + 0.6 * (t - (0.6+0.22/0.6))

    elif scenario_index == 4:
        x_start = 1.1 / np.sqrt(1 + np.square(0.32 / 1.05)) - 11.55
        y_start = (0.32/1.05) * (1.1 / np.sqrt(1 + np.square(0.32 / 1.05))) + 7.43
        if t>=0 and t<0.6:
            alpha = 180 + np.rad2deg(np.arctan(0.32/1.05))
            x = x_start + np.cos(np.deg2rad(alpha)) * 0.5 * t ** 2
            y = y_start + np.sin(np.deg2rad(alpha)) * 0.5 * t ** 2
        elif t>=0.6 and t< (0.6+0.92/0.6):
            alpha = 180 + np.rad2deg(np.arctan(0.32 / 1.05))
            x = x_start + np.cos(np.deg2rad(alpha)) * 0.5 * 0.6 ** 2 + np.cos(np.deg2rad(alpha)) * 0.6 * (t - 0.6)
            y = y_start + np.sin(np.deg2rad(alpha)) * 0.5 * 0.6 ** 2 + np.sin(np.deg2rad(alpha)) * 0.6 * (t - 0.6)
        elif t>=0.6+0.92/0.6 and t<=32:
            alpha = 0
            x = -11.55 - 0.6 * (t - (0.6 + 0.92/0.6))
            y = 7.43

    return x, y, alpha

   
if scenario_index == 1:
        time_indices = np.arange(25000)
elif scenario_index == 2:
        time_indices = np.arange(25000)
elif scenario_index == 3:
        time_indices = np.arange(20000)
elif scenario_index == 4:
        time_indices = np.arange(30000)
else:
        raise ValueError("Invalid scenario index")

    
position = []

# every time and position
for time_index in time_indices:
        x, y, alpha = sew_track_function(time_index / 1000, scenario_index)
        position.append({
            'Time': time_index / 1000,
            'X': x,
            'Y': y,
            'Alpha': alpha
        })

#print(f" Position Data: {position}")




############      save the file       ############################################################################
save_path = "/media/campus/SEW/Bearbeitet_Data/Final"
#json_filename = f"Round_{round_number}_RX_{rx_index}_RF_{rf_index}_Scenario_{scenario_index}_Second_{second}.json"
json_filename = os.path.join(save_path, f"_Scenario_{scenario_index}_Round_{round_number}_RX_{rx_index}_RF_{rf_index}_Polarization_{polarization}_Uhrzeit_{time_for_round_formatted}_Second_{second}.json")

data = {
    "cir": cir_data.tolist(),
    "lin_spec": lin_spec_data.tolist(),
    "position": position
}
with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)

print(f"Data saved to {json_filename}")


############      plot       ############################################################################

t = np.linspace(3.71,3.79,400)


plt.plot(t,lin_spec[:,999])
plt.show()