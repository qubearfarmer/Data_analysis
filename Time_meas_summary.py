import numpy as np
from matplotlib import pyplot as plt
from qutip import*

###################################################################################
directory = 'G:\\Projects\\Fluxonium\\Data\\Fluxonium #10_python code by Jon\\T1_T2_YOKO_41p5to mA_0to2'
measurement = 'T1 avg_T2_qubit f(0to2) vs flux_41p5 to 41p77mA.txt'
path = directory + '\\' + measurement
data = np.genfromtxt(path)
print data
t1 = data[1::,2]
flux = data[1::,0]
t2 = data[1::,4]

fig, ax1 = plt.subplots()
#####################################################################################################################################################################################

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

"""
First section of the script attempts to plot the energies vs external flux
"""
#Hamiltonian definition
def Ho(N,E_l, E_c, E_j_sum, d, phi_squid, phi_ext):
    E_j1 = 0.5*E_j_sum*(1+d)
    E_j2 = 0.5*E_j_sum*(1-d)
    a = tensor(destroy(N))
    mass = 1.0/(8.0*E_c)
    w = sqrt(8.0*E_c*E_l)
    phi = (a+a.dag())*(8*E_c/E_l)**(0.25)/np.sqrt(2)
    na = 1j*(a.dag()-a)*(E_l/(8*E_c))**(0.25)/np.sqrt(2)
    ope1 = 1j*(-phi + phi_ext)
    ope2 = 1j*(phi - phi_ext + phi_squid) # phi_squid and phi_ext here are the external phases, or normalized flux, = flux*2pi/phi_o
    H = 4.0*E_c*na**2 + 0.5*E_l*(phi)**2 - 0.5*E_j1*(ope1.expm() + (-ope1).expm()) - 0.5*E_j2*(ope2.expm() + (-ope2).expm())
    return H.eigenenergies()

def coupled_H(Na, E_l, E_c, E_j_sum, d, phi_squid, phi_ext, Nr, wr, g):
    E_j1 = 0.5*E_j_sum*(1 + d)
    E_j2 = 0.5*E_j_sum*(1 - d)
    a = tensor(destroy(Na), qeye(Nr))
    b = tensor(qeye(Na), destroy(Nr))
    phi = (a + a.dag())*(8.0*E_c/E_l)**(0.25)/np.sqrt(2.0)
    na = 1.0j*(a.dag() - a)*(E_l/(8*E_c))**(0.25)/np.sqrt(2.0)
    ope1 = 1.0j*(phi_ext - phi)
    ope2 = 1.0j*(phi + phi_squid - phi_ext)
    H_f = 4.0*E_c*na**2 + 0.5 * E_l*(phi)** 2 - 0.5*E_j1*(ope1.expm()+(-ope1).expm()) - 0.5*E_j2*(ope2.expm()+(-ope2).expm())
    H_r = wr*(b.dag()*b + 1.0/2)
    H_c = -2*g * na * (b.dag + b)
    H = H_f + H_r + H_c
    return H.eigenenergies()


def trans_energies(N, E_l, E_c, E_j_sum, d, A_j, A_c, B_coeff, beta_squid, beta_ext, level_num, current, iState):
    B_field = current*B_coeff*1e-4  # in T, this depends on a seperate measurement
    phi_squid = B_field*A_j # these are flux, not normalized
    phi_ext = B_field*A_c
    trans_energy = np.zeros((level_num-iState,len(phi_ext)))
    for idx in range(len(phi_ext)):
        energies = Ho(N,E_l, E_c, E_j_sum, d, 2*np.pi*(phi_squid[idx]/phi_o - beta_squid), 2*np.pi*(phi_ext[idx]/phi_o - beta_ext)) #normalize the flux -> phase here
        for level in range(iState+1,level_num):
            trans_energy[level-iState,idx]=energies[level]-energies[iState]
    return trans_energy
#############################################
N = 50
E_l = 0.746959655208
E_c = 0.547943694372
E_j_sum = 21.9627179709
level_num = 3
B_coeff = 60
A_j = 3.80888914574e-12
A_c = 1.49982268962e-10
beta_squid = 0.00378012644185
beta_ext = 0.341308382441
d=0.0996032153487
current = np.linspace(0.041,0.042,100)


iState = 0
spectrum = trans_energies(N, E_l, E_c, E_j_sum, d, A_j, A_c, B_coeff, beta_squid, beta_ext, level_num, current, iState)
for idx in range(iState,level_num):
    line = ax1.plot(current*1e3+ (41.6813-41.6413), spectrum[idx,:])  # transition from state (iState)
    plt.setp(line,linewidth=1.0, linestyle ='-', color = "black", alpha=0.5)

ax2.set_ylabel('Trans energy (GHz)')
for tl in ax1.get_yticklabels():
    tl.set_color('k')

ax2 = ax1.twinx()
ax2.plot(flux, t1, 'bo-')
ax2.set_ylabel('T1_02 (us)')
for tl in ax2.get_yticklabels():
    tl.set_color('b')
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
plt.show()