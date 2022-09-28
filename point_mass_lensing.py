#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:31:21 2022

@author: chema
"""

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import imageio

plt.style.use('seaborn-white')

rng = default_rng()
np.random.seed(0)

plt.close('all')

create_gif = False # This takes time ~ 1'

Einstein_ring = False # Set all parameters to create an E. ring.

#######################################################################
#                   Physical constants / parameters                   #
#######################################################################
# Do not touch this (Maybe just D, but change z_s accordingly).
c = 2.99792458e8 # Speed of light. [m/s]
G = 6.67408e-11 # Gravitational constant. [m**3/(kg s)]
D = 30 * 3.2408e20 # Effective distance. [m]
M_s = 1.98847e30 # Solar mass. [kg]
theta_e = np.sqrt(4*G*M_s / (D*c**2)) # Einstein radius (1M_s).
z_d = 0.17 # Lens plane redshift.

#######################################################################
#                        Simulation parameters                        #
#######################################################################
'''
    The IRS technique used for computing the caustics needs a larger
    area for N ~> 8, that will prob. exceed the available values.
    
    For greater values of N with this size the caustics will not be
    OK, but the rest probably is.
'''
N = 5 # Lens number.
M = np.random.uniform(low=0.4, high=1.3, size=N) # Lens masses. M_s
Nx = 1000 # Number of pixels in the x-direction.
Ny = 1000 # Number of pixels in the y-direction.
size_theta_e = 50 * theta_e # Size (Theta_e) along each axis.
pix_size = 8 * 5e-10 / Nx # Physical size of the pixel. rad
dx = np.random.uniform(low=Nx/4, high=3*Nx/4, size=N) \
    - Nx/2 # Lens x-positions.
dy = np.random.uniform(low=Ny/4, high=3*Ny/4, size=N) \
    - Nx/2 # Lens y-positions.
Beta_x = 20 * pix_size
Beta_y = -100 * pix_size # Source position fixed at origin.
# Wrt the center of the simulation.

if Einstein_ring:
    N = 1
    M = np.array([1])
    dx = np.array([0])
    dy = np.array([0]) # Lens 1 fixed at origin.
    Beta_x = 0 * pix_size
    Beta_y = 0 * pix_size # Source position fixed at origin.


def set_axes(ax):
    label = r' $\theta_E$'
    size = size_theta_e / theta_e
    tick_labels = [str(i)+label for i in np.linspace(-size,
                   size, 6, endpoint=True)]
    position = np.linspace(0, Nx, 6, endpoint=True)

    ax.set_xticks(position[1:-1], tick_labels[1:-1])
    ax.set_yticks(position[1:-1], tick_labels[1:-1])
    
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    fig.set_size_inches(7, 7)
    
def makeGaussian(size,sigma=3,
                 center=None,
                 toplot=False):

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    y = np.arange(0, size, 1, float) # Asummes a square grid
    x = y[:,np.newaxis] # This couple of lines generates a very efficient
                        # structure of axes (horizontal and vertical)
                        # that can be filled very quickly with a function
                        # such as the Gaussian defined in the next line.
                        # This is far much faster than the old-fashioned
                        # nested FOR loops.

    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    
#######################################################################
x_shift = np.linspace(-Nx/2*pix_size, Nx/2*pix_size,
                      num=Nx, endpoint=True) # x-step.
y_shift = np.linspace(-Ny/2*pix_size, Ny/2*pix_size,
                      num=Ny, endpoint=True) # y-step.
theta_x = np.tile(x_shift, (Ny, 1)).T # x-position map.
theta_y = np.tile(y_shift, (Nx, 1)) # y-position map.
theta = np.zeros((Nx, Ny, 2),dtype=float) # position map.
distance = np.zeros((Nx, Ny, N),dtype=float) # Distance map to a lens.

psi = np.zeros((Nx, Ny),dtype=float) # Potential.
alpha_x = np.zeros((Nx, Ny),dtype=float) # Deflection angle (x).
alpha_y = np.zeros((Nx, Ny),dtype=float) # Deflection angle (y).
mu = np.zeros((Nx, Ny),dtype=float) # Magnification (lens plane).

# Compute psi and alpha at each pixel given all the lenses.
for k in tqdm(range(N)):
    distance = np.sqrt((-theta_x + dx[k]*pix_size)**2 \
                       + (-theta_y + dy[k] * pix_size)** 2) # [rad]
        
    psi += (theta_e*M[k])**2 * np.log(np.abs(distance)) # [J/kg]
    
    alpha_x += (theta_e*M[k])**2 \
                * (theta_x - dx[k]*pix_size) / distance**2 # [rad]
    alpha_y += (theta_e*M[k])**2 \
                * (theta_y - dy[k]*pix_size) / distance**2 # [rad]
      
# Derivatives
axx = np.zeros((Nx, Ny),dtype=float) 
ayy = np.zeros((Nx, Ny),dtype=float) 
axy = np.zeros((Nx, Ny),dtype=float) 
ayx = np.zeros((Nx, Ny),dtype=float) 

ayy = np.diff(alpha_y,axis=0) / pix_size # d(alpha_y)/d(y)
ayx = np.diff(alpha_y,axis=1) / pix_size # d(alpha_y)/d(x)
axy = np.diff(alpha_x,axis=0) / pix_size # d(alpha_x)/d(y)
axx = np.diff(alpha_x,axis=1) / pix_size # d(alpha_x)/d(x))

# Magnification
mu = 1/abs((1-np.delete(axx, -1, axis=0) - np.delete(ayy, -1, axis=1)
            + (np.delete(axx, -1, axis=0)) * np.delete(ayy, -1, axis=1)
            - (np.delete(axy, -1, axis=1)) * np.delete(ayx, -1, axis=0)
          ))

del axx, axy, ayx, ayy

'''fig, ax0 = plt.subplots()           
im = ax0.imshow(mu, norm=colors.LogNorm(), origin='lower')
ax0.set_title(r'Magnification lens plane', fontsize=20)
set_axes(ax0)'''

del mu
# Caustics
beta_x = (theta_x - alpha_x)
beta_y = (theta_y - alpha_y)

beta = np.zeros((int(Nx),
                 int(Ny)),dtype=float)

temp_x = (np.round(beta_x/pix_size) + Nx/2).astype(int)
temp_y = (np.round(beta_y/pix_size) + Nx/2).astype(int)

del  beta_y, beta_x

for i in tqdm(range(Nx)):
    for j in range(Ny):
        k = temp_x[i][j]
        l = temp_y[i][j]
        
        if (k >= min(Nx, Ny) or k < 0 \
         or l >= min(Nx, Ny) or l < 0):
            continue
        else:
            beta[l][k] += 1

del temp_x, temp_y

fig, ax1 = plt.subplots()           
im = ax1.imshow(beta, norm=colors.LogNorm(), origin='lower')
ax1.set_title(r'Caustics', fontsize=20)
set_axes(ax1)

del beta

t = np.ones((Nx,Ny),dtype=float) * (1+z_d) / 3e8*3e4*3e16 
# Time delay function.

t *= (1/2 * ((theta_x - Beta_x*pix_size)** 2
           + (theta_y - Beta_y*pix_size)** 2) - psi) 

fig, ax2 = plt.subplots()

t_x = np.zeros((Nx, Ny),dtype=float) 
t_y = np.zeros((Nx, Ny),dtype=float) 
t_prime = np.zeros((Nx, Ny),dtype=float)  

t_x = np.diff(t,axis=1) / pix_size # dT/dx.
t_y = np.diff(t,axis=0) / pix_size # dT/dy.
t_prime = (abs(np.delete(t_x, -1, axis=0)) \
         + abs(np.delete(t_y, -1, axis=1))) * pix_size # dT/dt.
    
del t, t_x, t_y, psi

im = ax2.imshow(abs(t_prime),norm=colors.LogNorm(), origin='lower')
ax2.set_title(r'abs(Time delay derivative)', fontsize=20)
set_axes(ax2)

del t_prime

# Compute images.
s = 3 # Source size in pixels.

d = np.zeros((Nx, Ny), dtype=float)
d_x = np.zeros((Nx, Ny), dtype=float)
d_y = np.zeros((Nx, Ny), dtype=float)

arcs = np.zeros((Nx,Ny),dtype=float)

d_x = (Beta_x-theta_x+alpha_x) / pix_size 
d_y = (Beta_y-theta_y+alpha_y) / pix_size 

d = np.sqrt(d_x**2 + d_y**2)

arcs = np.exp(-d**2/(2*s**2))

fig, ax3 = plt.subplots()
im = ax3.imshow(arcs,norm=colors.SymLogNorm(linthresh=1e-5),
                origin='lower')
ax3.set_title(r'Image position', fontsize=20)
plt.show()
set_axes(ax3)

if create_gif:
    plt.close('all')
    
    filenames = []
    for i in tqdm(np.linspace(-200,200,num=60,endpoint=True)):
        fig, ax = plt.subplots(ncols=2, nrows=1)
        # plot the line chart
        s = 3 
        Beta_x = i * pix_size
        Beta_y = i * pix_size
    
        d = np.zeros((Nx, Ny), dtype=float)
        d_x = np.zeros((Nx, Ny), dtype=float)
        d_y = np.zeros((Nx, Ny), dtype=float)
        
        d_b = np.zeros((Nx, Ny), dtype=float)
        d_bx = np.zeros((Nx, Ny), dtype=float)
        d_by = np.zeros((Nx, Ny), dtype=float)
    
        arcs = np.zeros((Nx,Ny),dtype=float)
        source = np.zeros((Nx,Ny),dtype=float)
    
        d_x = (Beta_x-theta_x+alpha_x) / pix_size 
        d_y = (Beta_y-theta_y+alpha_y) / pix_size 
    
        d = np.sqrt(d_x**2 + d_y**2)
        
        d_bx = (Beta_x) / pix_size 
        d_by = (Beta_y) / pix_size 
    
        d_b = np.sqrt(d_bx**2 + d_by**2)
    
        arcs = np.exp(-d**2/(2*s**2))
        source = makeGaussian(Nx, sigma=s,
                              center=(i+Nx//2, i+Ny//2))
    
        plt.imshow(arcs,norm=colors.SymLogNorm(linthresh=1e-5),
                   origin='lower')
        ax[1].set_title(r'Image plane (Arcs)', fontsize=20)
        
        ax[0].imshow(source,
                     origin='lower')
        ax[0].set_title(r'Source plane', fontsize=20)
        
        for j in range(len(dx)):
            if j == 0:
                ax[0].scatter(dx[j]+Nx//2, dy[j]+Ny//2, marker='o',
                              color='blue', label='Lens', linewidths=1)
            else:
                ax[0].scatter(dx[j]+Nx//2, dy[j]+Ny//2, marker='o',
                              color='blue', linewidths=1)
        
        ax[0].legend(loc='upper left', fontsize=15, frameon=True)
        
        set_axes(ax[0])
        set_axes(ax[1])
        
        fig.set_size_inches(14, 7)
        
        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()# build gif
    with imageio.get_writer('arcos.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
        



