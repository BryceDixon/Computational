# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# quick code to integrate a pair of masses coupled with springs
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45 as RK

# spring constants and masses

k = 1
m = 1
omega = np.sqrt(k/m)

# this is the coupled ODEs. y is now an array of 4 values, and dydt returns
# an array as well

def dydt(t,y):
    
    dydt_array = np.zeros(4)
    
    x1 = y[2] # these are the positions
    x2 = y[3]
    dydt_array[0] = k*(-x1+k*(x2-x1))/m #acceleration of mass 1
    dydt_array[1] = -k*(x2-x1) -k * x2# acceleration of mass 2
    dydt_array[2] = y[0] # velocity 1
    dydt_array[3] = y[1] # velocity 2
    
    return dydt_array

# performs one 4th order Runge-Kutta step

def rungaKutta(y,h):
    k1 = h * dydt(y)
    k2 = h * dydt(y + 0.5*k1)
    k3 = h * dydt(y + 0.5*k2)
    k4 = h * dydt(y + k3)    
    return (y + k1/6. + k2/3. + k3/3. + k4/6.)

# start and end times
time = 0.
tEnd = 30.

# small timestep for accuracy

h = 0.01


x1 = np.zeros(0)
x2 = np.zeros(0)

timeArray = np.zeros(0)
y = np.zeros(4)
y[0] = 0. # masses at rest
y[1] = 0.
y[2] = -0.1 # masses displaced by 10cm
y[3] = 0.1

# let's track the total energy to make sure everything's ok. Need to track the
# kinetic energy of the two masses plus the elastic energy of three springs

initialEnergy = 0.5*k*y[2]**2 + 0.5*k*y[3]**2 + 0.5*m*y[0]**2 + 0.5*m*y[1]**2 + 0.5*k*(y[2]-y[3])**2

# now let's set up the period depending on whether the masses are in phase or anti-phase
if (y[2]*y[3]>0.):
    period = 2.*np.pi/(omega)
else:
    period = 2.*np.pi/(np.sqrt(3)*omega)

# this sets up some points for plotting the period on the graph at the end
n = int(tEnd/period)
xdots = np.linspace(0.,n*period,n+1)
ydots = np.zeros(n+1)

runga = RK(dydt, time, y, tEnd, max_step = 1, first_step = 0.01, atol = 1e-8, rtol = 1e-6)
# the time loop
num = 0
print(num)
while (time < tEnd):
    
    runga.step()
    # call the integrator and step forward in time
    y_vals = runga.y
    time_vals = time + runga.t
    
    # append the mass positions and time to arrays for plotting 
    x1= np.append(x1,y_vals[2])
    x2= np.append(x2,y_vals[3])
    timeArray = np.append(timeArray, time_vals)
    num+=1
    print(num)
    # calculate the total energy
    energy = 0.5*k*y_vals[2]**2 + 0.5*k*y_vals[3]**2 + 0.5*m*y_vals[0]**2 + 0.5*m*y_vals[1]**2 + 0.5*k*(y_vals[2]-y_vals[3])**2
    
    # report the current energy as a relative deviations from the initial energy
    print("Relative Energy error:", energy/initialEnergy-1)
    if runga.status == 'finished':
        break

# plot the results
fig,ax =  plt.subplots()
ax.plot(timeArray,x1,'-')
ax.plot(timeArray,x2,'-')
ax.plot(xdots,ydots,'o')
ax.set_xlabel('Time (s)')
plt.show()