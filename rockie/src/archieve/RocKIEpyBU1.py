# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:33:41 2020

@author: Saul
"""

#Math
import math
import numpy as np
from scipy.io import loadmat
from scipy.spatial import Delaunay


# Geometry
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation as ROT
from scipy.linalg import norm

# Interpolation
from scipy.interpolate import griddata

RockRaw = loadmat(r'C:\Users\Saul\Desktop\RocKIE\Original\rock303.mat')

#Cleanup the imported Dataset
XimpTemp=RockRaw['Dataset'][0][1]
YimpTemp=RockRaw['Dataset'][0][2]
ZimpTemp=RockRaw['Dataset'][0][3]
Rock3dRaw=np.concatenate((XimpTemp,YimpTemp,ZimpTemp),axis=1)
Rock2dRaw=np.concatenate((XimpTemp,YimpTemp),axis=1)
#Section Clean-up
del XimpTemp
del YimpTemp
del ZimpTemp
del RockRaw

#if len(Rock3dRaw)<3:
    #break



#Centroid of Dataset
CentY=np.dot(Rock2dRaw[:,0],Rock2dRaw[:,1])/np.sum(Rock2dRaw[:,0])
CentX=np.dot(Rock2dRaw[:,0],Rock2dRaw[:,1])/np.sum(Rock2dRaw[:,1])
Centroid=np.asarray([CentX,CentY])
#Section Clean-Up
del CentY
del CentX


#Finding Boundary Points
#Source: https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second




def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst

edges = alpha_shape(Rock2dRaw, alpha=1000, only_outer=True)
bIndex=stitch_boundaries(edges)
bIndex=bIndex[0]
bIndex=np.asarray(bIndex)
bIndex=bIndex[:,0]
Boundary2d=Rock2dRaw[bIndex]
Boundary2d_3d=Rock3dRaw[bIndex]
##Section Cleanup
del edges


# Establishing Primary Axis

# Distance matrix
DistanceMatrix = squareform(pdist(Boundary2d,'euclidean'))
# Longest Line
LongestLine = np.unravel_index(indices=np.argmax(DistanceMatrix),shape=DistanceMatrix.shape)

# Define Primary axis
PrimAxPt0 = Boundary2d[LongestLine[0],:]
PrimAxPt1 = Boundary2d[LongestLine[1],:]
if PrimAxPt1[0]<PrimAxPt0[0]:
    temp=PrimAxPt1
    PrimAxPt1=PrimAxPt0
    PrimAxPt0=temp
    del temp
primary_ax = PrimAxPt1-PrimAxPt0
Primary_Axis_Leng=math.sqrt(primary_ax[0]**2+primary_ax[1]**2)



#####################
##Secondary Axis
#####################
if len(Rock2dRaw)>50:               #Dataset larger than 50 (best)
    wd=0.50                             #Weight to Dot Product
    wl=0.25                             #Weight to Length
    wc=1-wd-wl                          #Weight to Centroid Proximity
elif len(Rock2dRaw)<20:              #Small Dataset(x<20)
    wd=0.98
    wl=0.01
    wc=1-wd-wl
else:                                #Medium Dataset (20<x<50)
    wd=0.25
    wl=0.15
    wc=0.60

LSA=0
fit0=0
#Define Unit Vector 1
Vector1=PrimAxPt1-PrimAxPt0
VNorm1=math.sqrt((Vector1[0])**2+(Vector1[1])**2)
VUnit1=Vector1/VNorm1

for i in range(len(Boundary2d)):
    tempX3=Boundary2d[i][0]
    tempY3=Boundary2d[i][1]
    TempPt3=np.asarray([tempX3,tempY3])
    
    for j in range(len(Boundary2d)):
        tempX4=Boundary2d[j][0]
        tempY4=Boundary2d[j][1]
        TempPt4=np.asarray([tempX4,tempY4])
        #Length Calculation
        Len2=math.sqrt((tempX4-tempX3)**2+(tempY4-tempY3)**2)
        if Len2==0:
            pass

        else:
            #Normal Calculations
            Vector2=TempPt4-TempPt3
            VNorm2=math.sqrt((Vector2[0])**2+(Vector2[1])**2)
            VUnit2=Vector2/VNorm2
            PerpFactor=abs(np.dot(VUnit1,VUnit2))
            
            #Distance to Centroid Calculations
            CentFactor=norm(np.cross(Vector2, [TempPt3-Centroid]))/norm(Vector2)
    
            #Length Factoring
            LenFactor=1/(1+math.exp(-(Len2/Primary_Axis_Leng)))
            if LenFactor>0.68:
                LenFactor=0.68
            fit=wd*(1-PerpFactor)+wl*LenFactor+wc*CentFactor
            if fit>fit0:
                fit0=fit
                Secondary_P0=TempPt3
                Secondary_P1=TempPt4
                Secondary_Axis=np.asarray([Secondary_P0,Secondary_P1])
                Secondary_Len=Len2
                LenFactorR=LenFactor
                DotR=PerpFactor
                CentFactorR=CentFactor
                
#Point Standardization
if Secondary_P0[0]<Secondary_P1[0]:
    temp=Secondary_P1
    Secondary_P1=Secondary_P0
    Secondary_P0=temp
    del temp

 