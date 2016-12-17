
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
warnings.simplefilter('error', OptimizeWarning)
plt.figure(figsize=(5, 5))
plt.rc('font', family='serif')

def T1_func(x,a,b,c,d):
    return a*np.exp(-(x-c)/b) + d

directory = "G:\Projects\Fluxonium\Data\Fluxonium #10_python code by Jon\T1_T2_(0to1)_YOKO_38p5to38p76mA\T1_T2_(0to1)_YOKO_38p76mA"
measurement = 't1_pulse_2.951e9_708'

current_uncorr = 38.76
current = 38.74
fname = 'T1_T1err_'+str(current_uncorr)+'mA.csv'

path = directory + '\\' + measurement
time = np.genfromtxt(path+'_time0.csv')
phase_avg = np.zeros(len(time))
T1_guess = 1500e-6
count = 0
T1_array = []
T1_err_array = []
for idx in range(0,10):
    phase = np.genfromtxt(path + '_phase' + str(idx)+'.csv')
    phase = -phase*180/np.pi
    phase = phase - np.min(phase)
    guessA = np.max(phase)-np.min(phase)
    guess =[guessA, T1_guess, 0, 0]
    if guessA < 0.3 :
        continue
    try:
        popt,pcov = curve_fit(T1_func, time*1e-9, phase, guess)
    except RuntimeError:
        continue
    except OptimizeWarning:
        continue
    a,b,c,d = popt
    perr = np.sqrt(abs(np.diag(pcov)))
    T1 = b*1e6
    T1_err = perr[1]*1e6
    T1_array = np.append(T1_array, T1)
    T1_err_array = np.append(T1_err_array, T1_err)
    phase_avg = phase_avg + phase
    count = count + 1
    print T1, ',', T1_err
    plt.plot(time,phase)

###################################################################
#Save file
directory = 'G:\Projects\Fluxonium\Data\Summary of T1_T2_vs flux_Fluxonium#10\Corrected flux\Individual fits'
data = np.zeros((count, 4))
data[0:,0] = np.ones(count)*current
data[0:,1] = np.ones(count)*current_uncorr
data[0:,2] = T1_array
data[0:,3] = T1_err_array
path = directory + '\\' + fname
np.savetxt(path, data, delimiter =',', header='YOKO, YOKO_uncorr, T1, T1_err')
#####################################################################
############################AVG PHASE################################
#####################################################################
phase_avg = phase_avg / count
guessA = np.max(phase_avg)-np.min(phase_avg)
guess =[guessA, T1_guess, 0, 0]
popt,pcov = curve_fit(T1_func, time*1e-9, phase_avg, guess)
a,b,c,d = popt
perr = np.sqrt(abs(np.diag(pcov)))
T1 = b*1e6
T1_err = perr[1]*1e6
print "Average: ", T1,',', T1_err
plt.show()