import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.linalg import norm
from scipy.interpolate import griddata



def Mat_To_Pandas(Raw):
    X = pd.DataFrame(Raw['Dataset'][0][1],columns = ['X']).astype(float)
    Y = pd.DataFrame(Raw['Dataset'][0][2],columns = ['Y']).astype(float)
    Z = pd.DataFrame(Raw['Dataset'][0][3],columns = ['Z']).astype(float)
    DataRaw = pd.concat([X,Y,Z],axis=1,ignore_index=False)
    return DataRaw

def Centroid_2d(X, Y):
    CentY = np.dot(X,Y)/X.sum()
    CentX = np.dot(X,Y)/Y.sum()
    Centroid = np.asarray([CentX,CentY])
    Centroid = np.reshape(Centroid,(1,2))
    Centroid = pd.DataFrame(Centroid, columns = ['X','Y'])
    return Centroid

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

    def unzip_touples(test_list):
        res = [[ i for i, j in test_list ], 
               [ j for i, j in test_list ]] 
        return res

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
    edges = unzip_touples(list(edges))
    return edges[0]

def Primary_Axis(X,Y):
    Primary_Axis = pd.DataFrame(index=['Point 1', 'Point 2'], columns=['X','Y'])
    Boundary2d = pd.concat([X,Y], axis=1)
    DistanceMatrix = squareform(pdist(Boundary2d,'euclidean'))
    # Longest Line
    LongestLine = np.unravel_index(indices=np.argmax(DistanceMatrix),shape=DistanceMatrix.shape)
    # Define Primary axis
    if Boundary2d.loc[LongestLine[0]][0] > Boundary2d.loc[LongestLine[1]][0]:
        Primary_Axis.loc['Point 1'] = Boundary2d.loc[LongestLine[0]]
        Primary_Axis.loc['Point 2'] = Boundary2d.loc[LongestLine[1]]
    else:
        Primary_Axis.loc['Point 1'] = Boundary2d.loc[LongestLine[1]]
        Primary_Axis.loc['Point 2'] = Boundary2d.loc[LongestLine[0]]
    return Primary_Axis

def Secondary_Axis(X,Y, Primary_Axis, Centroid):
    Secondary_Axis = pd.DataFrame(index=['Point 1', 'Point 2'], columns=['X','Y'])
    Metrics = pd.DataFrame(columns=['Fit','Len', 'Perp','CF'], index=['Stats'])
    Boundary2d = pd.concat([X,Y], axis=1)
    
    #Set Weights
    if len(Boundary2d)>50: #Dataset larger than 50 (best)
        wd, wl, wc = 0.50, 0.35, 0.15
    elif len(Boundary2d)<20:              #Small Dataset(x<20)
        wd, wl, wc = 0.98, 0.01, 0.01    
    else:                                #Medium Dataset (20<x<50)
        wd, wl, wc = 0.25, 0.15, 0.60

    #Initial Fit
    fit0=1
    #Define Unit Vector 1
    Primary_Vec = Primary_Axis.loc['Point 2'] - Primary_Axis.loc['Point 1']
    Primary_Norm = ((Primary_Vec[0])**2+(Primary_Vec[1])**2)**0.5
    Primary_Unit_Vector = Primary_Vec/Primary_Norm
    
    for i, point in enumerate(Boundary2d):
        SA_Temp = pd.DataFrame(index=['Point 1', 'Point 2'], columns=['X','Y'])
        SA_Temp.iloc[0][0]=Boundary2d.iloc[i][0]
        SA_Temp.iloc[0][1]=Boundary2d.iloc[i][1]

        for j in range(len(Boundary2d)):
            SA_Temp.iloc[1][0]=Boundary2d.iloc[j][0]
            SA_Temp.iloc[1][1]=Boundary2d.iloc[j][1]

            SV_Temp_Vec = SA_Temp.loc['Point 2'] - SA_Temp.loc['Point 1']
            SV_Temp_Norm = ((SV_Temp_Vec[0])**2+(SV_Temp_Vec[1])**2)**0.5
            if SV_Temp_Norm==0:
                pass
            else:
                SV_Temp_Unit_Vector = SV_Temp_Vec/SV_Temp_Norm
                #Perpendicularness Calculations
                Perp_Factor = abs(np.dot(Primary_Unit_Vector, SV_Temp_Unit_Vector))**2
                
                #Length Factoring
                Len_Factor=(SV_Temp_Norm/Primary_Norm)
                if Len_Factor>0.75:
                    Len_Factor=0.50
                    
                #Distance to Centroid Calculations
                Cent_Temp1 = SA_Temp.loc['Point 1'] - Centroid
                Cent_Temp2 = norm(np.cross(SV_Temp_Vec, Cent_Temp1))
                Cent_Temp3 = Cent_Temp2 / SV_Temp_Norm
                Cent_Factor = abs(1-Cent_Temp3)**2
            
                #Minimizing Cost Function    
                cost = wd*(1-Perp_Factor)+wl*Len_Factor+wc*Cent_Factor
                fit = 1 - cost
                if fit < fit0:
                    fit0=fit
                    if SA_Temp.iloc[0][1] > SA_Temp.iloc[1][1]:
                        Secondary_Axis.loc['Point 1'] = SA_Temp.loc['Point 1']
                        Secondary_Axis.loc['Point 2'] = SA_Temp.loc['Point 2']
                    else:
                        Secondary_Axis.loc['Point 1'] = SA_Temp.loc['Point 2']
                        Secondary_Axis.loc['Point 2'] = SA_Temp.loc['Point 1']
                    Metrics['Fit'] = fit
                    Metrics['Len'] = Len_Factor
                    Metrics['Perp'] = Perp_Factor
                    Metrics['CF'] = Cent_Factor   

    return Secondary_Axis, Metrics

def Bottom_Intercept(Prim_Axis, Sec_Axis):
    Intercept = pd.DataFrame(index=['Point 1'], columns=['X','Y'])

    Slope_Prime = (Prim_Axis.iloc[1][1] - Prim_Axis.iloc[0][1])/(Prim_Axis.iloc[1][0] - Prim_Axis.iloc[0][0]) 
    Int_Prime =  Prim_Axis.iloc[0][1]-Prim_Axis.iloc[0][0] * Slope_Prime
    Slope_Sec = (Sec_Axis.iloc[1][1] - Sec_Axis.iloc[0][1])/(Sec_Axis.iloc[1][0] - Sec_Axis.iloc[0][0]) 
    Int_Sec =  Sec_Axis.iloc[0][1]-Sec_Axis.iloc[0][0] * Slope_Sec
    
    xi = abs((Int_Prime-Int_Sec) / (Slope_Prime-Slope_Sec))
    yi = Slope_Prime * xi + Int_Prime

    Intercept['X'] = xi
    Intercept['Y'] = yi
    return Intercept

def Surface(Raw, step, xmin, xmax, ymin, ymax):
    grid_x, grid_y = np.mgrid[xmin: xmax: (xmax-xmin)/step, ymin: ymax: (ymax-ymin)/step]
    Z = griddata(Raw.drop(['Z'], axis = 1), Raw['Z'], (grid_x, grid_y), method = 'nearest', fill_value=Raw['Z'].min())
    
    fig = plt.figure(figsize=(6,2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Raw['X'], Raw['Y'], Raw['Z'], color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(Raw['X'], Raw['Y'], Raw['Z'], color='red')
    plt.show()
    plt.close()
    return Z