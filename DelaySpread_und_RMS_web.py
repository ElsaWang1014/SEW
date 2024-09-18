import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask
import math
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


app = Flask(__name__)
@app.route('/')
def index():
    return "Hello, Heroku!"

# load data
def load_data(load_path, round_numbers, num_seconds=25):
    data = []
    for second in range(1, num_seconds + 1):
        second_data = []
        for round_number in round_numbers:
            filename = f"Round_{round_number}_AP_1_RF_0_Sec_{second}.mat"
            full_filename = os.path.join(load_path, filename)
            if os.path.exists(full_filename):
                mat = scipy.io.loadmat(full_filename)
                cirs_data = mat["cirs"]
                second_data.append(np.abs(cirs_data[28:440, :]) ** 2)  
            else:
                print(f" File {filename} not found.")
        data.append(second_data)
    
    data = np.array(data)
    data = np.concatenate(data, axis=2)
    return data

# APDP  &  RMS DS
def compute_apdp_and_peaks(data, num_milliseconds):
    delays = np.arange(data.shape[1]) * sampling_interval - 10e-9
    apdp_db_all = []
    all_peaks_all = []
    rms_delay_spread_array = np.zeros(num_milliseconds)
    co_bandwidth_array = np.zeros(num_milliseconds)

    for ms in range(num_milliseconds):
        apdp = np.mean(data[:, :, ms], axis=0)
        apdp_db = 10 * np.log10(apdp)

        '''max_index = np.argmax(apdp_db)
        apdp_db_after_max = apdp_db[max_index:]'''

        min_height = np.max(apdp_db[200:]) + 3
        peaks, _ = find_peaks(apdp_db, height=min_height, prominence=(0.1, None))
        #peaks = peaks + max_index

        total_power = np.sum(apdp[:200])
        time_weighted_power = np.sum(delays[:200] * apdp[:200])
        tau_bar = time_weighted_power / total_power
        squared_delays = (delays[:200] - tau_bar) ** 2
        rms_delay_spread = np.sqrt(np.sum(squared_delays * apdp[:200]) / total_power)
        rms_delay_spread_array[ms] = rms_delay_spread
        co_bandwidth_array[ms] = 1 / (2 * math.pi * rms_delay_spread)

        apdp_db_all.append(apdp_db)
        all_peaks_all.append(peaks)

    return apdp_db_all, all_peaks_all, rms_delay_spread_array, co_bandwidth_array

# DATA PATH
load_path = "/media/student/SEW/Bearbeitet_Data/Rx1/Tag1_Scenario1_AGVHorizontal/"
round_numbers = [77, 78, 79, 80, 81, 82]
sampling_interval = 10e-9
data = load_data(load_path, round_numbers)
num_milliseconds = data.shape[2]
apdp_db_all, all_peaks_all, rms_delay_spread_array, co_bandwidth_array = compute_apdp_and_peaks(data, num_milliseconds)

#  Dash APP
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Average Power Delay Profil and RMS Delay Spread"),
    dcc.Graph(id='apdp-graph'),
    dcc.Slider(
        id='ms-slider',
        min=1,
        max=num_milliseconds ,
        step=1,
        value=0,
        marks={i: f'{i} ms' for i in range(num_milliseconds)}
    ),
    dcc.Graph(id='rms-graph')
])

@app.callback(
    [Output('apdp-graph', 'figure'),
     Output('rms-graph', 'figure')],
    [Input('ms-slider', 'value')]
)
def update_graph(ms):
    apdp_db = apdp_db_all[ms]
    peaks = all_peaks_all[ms]
    delays = np.arange(data.shape[1]) * sampling_interval - 10e-9

    max_index = np.argmax(apdp_db)
    max_value = apdp_db[max_index]

    apdp_trace = go.Scatter(
        x=delays * 1e9,
        y=apdp_db,
        mode='lines',
        name='APDP (dB)'
    )

    peaks_trace = go.Scatter(
        x=delays[peaks] * 1e9,
        y=apdp_db[peaks],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Peaks'
    )

    

    apdp_figure = {
        'data': [apdp_trace, peaks_trace],
        'layout': go.Layout(
            title=f"APDP for Rounds {round_numbers} at Millisecond {ms}",
            xaxis={'title': 'Delay Time (ns)'},
            yaxis={'title': 'APDP (dB)'}
        )
    }

    rms_trace = go.Scatter(
        x=np.arange(num_milliseconds),
        y=rms_delay_spread_array * 1e9,
        mode='lines',
        name='RMS delay Spread (ns)'
    )

    rms_figure = {
        'data': [rms_trace],
        'layout': go.Layout(
            title='RMS Delay Spread over Time',
            xaxis={'title': 'Time (ms)'},
            yaxis={'title': 'RMS delay spread (ns)'}
        )
    }

    return apdp_figure, rms_figure

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



