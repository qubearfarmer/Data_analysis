import numpy as np

e = 1.602e-19
h = 6.626e-34

#HFSS calculation
freq = 6.1935e9
L = 30e-9
C_shunt = ((freq*2*np.pi)**2*L)**-1.0*1e15
print ('Shunting capacitance: '+str(round(C_shunt,4))+' fF')
junc_area = 0.1*0.1     #um x um
CJ = 45*junc_area    #45fF per um^2
print ('Junction capacitance: '+str(round(CJ,4))+' fF')
C_sum = (CJ + C_shunt)*1e-15
EC = e**2/(2*C_sum)/h
print ('Charging energy: ' +str(round(EC*1e-9,4)) +' GHz')

L = 2500e-9 #nH
phi_o = h/(2*e*2*np.pi)
EL = phi_o**2/L/h
print ('Inductive energy: ' +str(round(EL*1e-9,4))+' GHz')


