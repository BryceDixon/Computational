# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# quick code to integrate a couple pendulum using Runge-Kutta integration
# either fixed step size or an adaptive step size
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from numpy import array

# the lengths of the two pendulums in m

L1 = 1.
L2 = 0.5
L = L1 + L2 # total length

# the masses of the pendulums
M1 = 1.
M2 = 1.

# gravitational acceleration in m/s^2
G = 9.8

# performs one 4th order Runge-Kutta step

def rungeKutta(dydt,x,y,h):
    k1 = h * dydt(x,y)
    k2 = h * dydt(x+0.5*h,y + 0.5*k1)
    k3 = h * dydt(x+0.5*h,y + 0.5*k2)
    k4 = h * dydt(x+h,y + k3)    
    return array(y + k1/6. + k2/3. + k3/3. + k4/6.)


def integrateFixed(tStart, tEnd, H, y, dumpstep):
    
    # this is a fixed timestep 4th order RK solver
    
    Y = []
    timeArray = []
    
    time = tStart
    Y.append(y)
    timeArray.append(time)
    tdump = dumpstep
 

    while (time < tEnd):
    
        thisH = H
        dump = False
        if (time+thisH > tdump): # make sure we append arrays exactly on times that are multiples of dumpstep
            thisH = tdump - time
            dump = True
        y= rungeKutta(coupledPendulum,time,y,thisH)
        time = time + thisH
        if (dump):
            Y.append(y)
            timeArray.append(time)
            tdump = tdump + dumpstep
    return array(timeArray), array(Y)

def integrateAdaptive(tStart, tEnd, H, y, dumpstep):
    
    # this is a 4th order RK solver with an adaptive step size
    
    Y = []
    timeArray = []
    
    time = tStart
    Y.append(y)
    timeArray.append(time)
    tdump = dumpstep
 
    thisH = H
    
    maxChange = 5.e-4
    minChange = 1.e-4
    hmin = 1.e-5
    while (time < tEnd):
      
        ycurrent = y
        yhalf = rungeKutta(coupledPendulum,time,y,thisH/2.)
        ydouble = rungeKutta(coupledPendulum,time,y,thisH*2.)
        
        if (any(abs((yhalf-ycurrent)/ycurrent) > maxChange)):
            thisH = max(thisH / 2.,hmin)
        elif  (any(abs((ydouble-ycurrent)/ycurrent) < minChange)):
            thisH = thisH * 2.
        
        
        dump = False
        if (time+thisH > tdump):
            tdash = tdump - time
            dump = True
            y = rungeKutta(coupledPendulum,time,y,tdash)
            time = time + tdash
        else:
            y = rungeKutta(coupledPendulum,time,y,thisH)
            time = time + thisH
        
        if (dump):
            print(time)
            Y.append(y)
            timeArray.append(time)
            tdump = tdump + dumpstep
    return array(timeArray), array(Y)

def coupledPendulum(x, y):
    
    # there are four coupled equations
    F = np.zeros(4)
    
    # here 
    # y[0] = \theta_1
    # y[1] = d\theta_1 by dt
    # y[2] = \theta_2
    # y[3] = d\theta_2 by dt
    
    F[0] = y[1] 
    delta = y[2]-y[0]
    den1 = (M1+M2)*L1 - M2*L1*np.cos(delta)*np.cos(delta)
    F[1] = (M2*L1*y[1]*y[1]*np.sin(delta)*np.cos(delta)\
    + M2*G*np.sin(y[2])*np.cos(delta) + M2*L2*y[3]*y[3]*np.sin(delta)\
    - (M1+M2)*G*np.sin(y[0]))/den1
    
    F[2] = y[3] 
    
    den2 = (L2/L1)*den1
    
    F[3] = (-M2*L2*y[3]*y[3]*np.sin(delta)*np.cos(delta) \
    + (M1+M2)*G*np.sin(y[0])*np.cos(delta) \
    - (M1+M2)*L1*y[1]*y[1]*np.sin(delta) \
    - (M1+M2)*G*np.sin(y[2]))/den2
    
    return array(F)

tStart = 0.0
tEnd = 10.0
H = 0.01
dumpstep = 0.01


y = array([0.0, 5.0,0.,2.]) # initial conditions

# call the integrator

timeArray, Y = integrateFixed(tStart, tEnd, H, y, dumpstep)
print(timeArray)
 
# potential energy

V = -(M1+M2)*G*L1*np.cos(Y[:,0]) - M2*G*L2*np.cos(Y[:,2]) 

# kinetic energy energy

T = 0.5*M1*L1**2*Y[:,1]**2+0.5*M2*(L1**2*Y[:,1]**2+L2**2*Y[:,3]**2 \
                                   + 2*L1*L2*Y[:,1]*Y[:,3]*np.cos(Y[:,0]-Y[:,2]))

# relative error in total energy compared to initial value

energy = (T+V) /(T[0]+V[0]) - 1.


# position of first mass

x1 = L1*np.sin(Y[:,0])
y1 =-L1*np.cos(Y[:,0])

# position of second mass

x2 = L2*np.sin(Y[:,2]) + x1
y2 =-L2*np.cos(Y[:,2]) + y1


# the animation code is based on that given for the matplotlib demo here
# https://matplotlib.org/stable/gallery/animation/double_pendulum.html

history_len = 50  # how many trajectory points to display in animation

# set up figure

fig = plt.figure(figsize=(5, 4))

# choose a box big enough for the entire length of combined pendulum

ax = fig.add_subplot(autoscale_on=False, xlim=(-L*1.1, L*1.1), ylim=(-L*1.1, 1.))
ax.set_aspect('equal')
ax.grid()

#this is the magenta line with circles at the points
line, = ax.plot([], [], 'mo-', lw=2)

# this is the blue line that traces part of the path of M2
trace, = ax.plot([], [], ',-', lw=1)

# two text labels for time and energy
time_template = 'time = %.1fs'
energy_template = 'energy error = %.5e'

# the positions of the two labels on the plot (near the top)

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
energy_text = ax.text(0.45, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


# do frame i

def animate(i):
    thisx = [0, x1[i], x2[i]] # positions of the masses of the masses
    thisy = [0, y1[i], y2[i]]

    if i == 0:   # no history for first frame
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[2])  # add the position of the 2nd mass to the history
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy) # the lines
    trace.set_data(history_x, history_y) # the trace of the second mass
   
    time_text.set_text(time_template % (i*dumpstep)) # time label
    
    energy_text.set_text(energy_template % (energy[i])) # energy label
    
    return line, trace, time_text, energy_text #return all the stuff

# this bit does the animation

ani = animation.FuncAnimation(
    fig, animate, len(Y), interval=dumpstep*1000, blit=True)
#writergif = animation.PillowWriter(fps=30) 
#ani.save('anim.gif', writer=writergif)
plt.show()