import numpy as np

e = 1.602e-19
h = 6.626e-34

junc_area = 0.4*2     #um x um
CJ = 45*junc_area    #45fF per um^2
print ('Junction capacitance: '+str(round(CJ,4))+' fF')
C_shunt = 0
C_sum = (CJ + C_shunt)*1e-15
EC = e**2/(2*C_sum)/h
print ('Charging energy: ' +str(round(EC*1e-9,4)) +' GHz')

L = 200e-9 #nH
phi_o = h/(2*e*2*np.pi)
EL = phi_o**2/L/h
print ('Inductive energy: ' +str(round(EL*1e-9,4))+' GHz')