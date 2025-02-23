# Plot intensity data

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import pandas as pd
rc('text', usetex=False)

#Enter directory and name of measurement
using_mac = True
if using_mac == True:
    data_directory = r'/Users/longbnguyen/Library/Mobile Documents/com~apple~CloudDocs/gridium_data//'
    fig_directory = r'/Users/longbnguyen/Library/Mobile Documents/com~apple~CloudDocs/gridium_paper/Illustration//'
    cd_sym = "//"
fig, ax = plt.subplots(figsize = [5,4])

################
# fname = data_directory + 'Formatted_data' + cd_sym + 'Soft_Gridium_Two_Tone_phi_sweep_theta=0.csv'
# df = pd.read_csv(fname)

# freq = np.linspace(0.01, 8.01, 1601)
# flux = np.linspace(0,1,301)
# data=df['amplitude']
# data = np.array(data)
# data=data.reshape((301,1601))

# X,Y = np.meshgrid(flux+0.004,freq)
# for idx in range(301):
#     data[idx,:] = data[idx,:] - np.average(data[idx,:])
#     # if np.max(data[idx,:]) > 4 or np.min(data[idx,:]) < -4:
#     #     data[idx,:] = [0]*1601
# Z = (data).transpose()
# axes=ax.pcolormesh(X,Y,Z, cmap = 'RdBu_r', vmin = -10.01, vmax = 10.01)
############################

fname = data_directory + 'Formatted_data' + cd_sym+'EJ=7_Gridium_Two_Tone_phi_sweep_theta=pi.csv'
df = pd.read_csv(fname)
freq = np.unique(df.frequency)*1e-9
current = np.unique(df.Current)*1e6
amplitude = np.reshape(df.amplitude, [len(current), len(freq)])
phase = np.reshape(df.phase, [len(current), len(freq)])
phi_ext = np.linspace(0,1,len(current))

X,Y = np.meshgrid(phi_ext, freq)
Z = amplitude.transpose()
Z = phase.transpose()
for idx in range(len(current)):
    Z[:,idx] = Z[:,idx]-np.average(Z[:,idx])
axes=ax.pcolormesh(X,Y,Z, cmap = 'RdBu_r', vmin = -2, vmax = 2)
###############################################  
#   
#Click on the points on screen to define an approximation line for interpolation
def onclick(event):
    print ('[%f, %f],'%(event.xdata, event.ydata))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()