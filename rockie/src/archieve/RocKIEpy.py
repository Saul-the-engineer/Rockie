# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:49:56 2020

@author: saulg
"""

import RocKIE_Func as RF


#Math
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist

#Import Data
RockRaw = loadmat('rock303.mat')
Raw = RF.Mat_To_Pandas(RockRaw)

#Calc Centroid of Data
Cent = RF.Centroid_2d(Raw['X'],Raw['Y'])

#Create Alphashape
edge_index = RF.alpha_shape(Raw.drop(['Z'], axis = 1).to_numpy(), 1.0, only_outer=True)

#Create Boundary Dataframe
Raw2d_Elev = pd.DataFrame(columns = ['X','Y','Z'])
for i, edge in enumerate(edge_index):
        Raw2d_Elev.loc[i] = Raw.iloc[edge]
        
#Establish Primary Axis
Prim_Axis = RF.Primary_Axis(Raw2d_Elev['X'],Raw2d_Elev['Y'])

#Establish Secondary Axis
Sec_Axis, Metrics = RF.Secondary_Axis(Raw2d_Elev['X'],Raw2d_Elev['Y'], Prim_Axis, Cent)

#Finding Intercept
Intercept_Bottom = RF.Bottom_Intercept(Prim_Axis, Sec_Axis)

#Plot 2d Results
plt.figure(figsize=(6,2))
plt.scatter(Raw2d_Elev['X'], Raw2d_Elev['Y'])
plt.plot(Prim_Axis['X'], Prim_Axis['Y'])
plt.plot(Sec_Axis['X'], Sec_Axis['Y'])
plt.scatter(Intercept_Bottom['X'], Intercept_Bottom['Y'], color = 'r')
plt.axis('equal')
plt.show()
plt.close()

#Grid Volume Surface Creation
step = 10
xGridSize = (Raw['X'].max() -Raw['X'].min())/step
yGridSize = (Raw['Y'].max() -Raw['Y'].min())/step
Surface_Top = RF.Surface(Raw, step, Raw['X'].min(), Raw['X'].max(), Raw['Y'].min(), Raw['Y'].max())
Bottom_Surface = RF.Surface(Raw2d_Elev, step, Raw['X'].min(), Raw['X'].max(), Raw['Y'].min(), Raw['Y'].max())
Surface_Volume_Grid = (Surface_Top - Bottom_Surface)
Volume_Surface = (Surface_Volume_Grid*xGridSize*yGridSize).sum().sum()
print('The Surface Volume is: ' + str(Volume_Surface))

#Elipsoid Volume
x_index_tert = math.floor((Intercept_Bottom.iloc[0][0] - Raw['X'].min())/xGridSize)
y_index_tert = math.floor((Intercept_Bottom.iloc[0][1] - Raw['Y'].min())/yGridSize)

Ter_Axis = Surface_Volume_Grid[y_index_tert][x_index_tert]
Prim_Len = pdist(Prim_Axis,'euclidean')
Sec_Len = pdist(Sec_Axis,'euclidean')

Volume_Elipsoid = (4/3)*np.pi*(Prim_Len/2)*(Sec_Len/2)*(Ter_Axis)
Volume_Elipsoid = Volume_Elipsoid[0]
print('The Elipsoid Volume is: ' + str(Volume_Elipsoid))
#Final Result
#Will determine what the best way to present final results later

print('debug')










