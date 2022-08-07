import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykrige.ok import OrdinaryKriging

# read in the data file using pandas dataframe
df = pd.read_csv('DataBoxTest.csv')

# separate the data from the data file
x = df.iloc[:,0]            # x position in meters
y = df.iloc[:,1]            # y position in meters
ID = df.iloc[:,2]           # name of the marine data survey
survey_num = df.iloc[:,3]   # integer identifier of data survey
mag = df.iloc[:,4]          # magnetic data survey value in nanotesla
uncertainty = df.iloc[:,5]  # uncertainty value for mag data in nanotelsa

# create a grid of the survey location (fill in points between lines)
grid_x = np.linspace(np.min(x)+0.01,np.max(x)-0.01,543)
grid_y = np.linspace(np.min(y)+0.01,np.max(y)-0.01,543)

# iterpolate the magnetic data to the new grid points - using gaussian distribution
OK = OrdinaryKriging(x,y,mag,variogram_model='gaussian',nlags=40)
mag1,ss1 = OK.execute('grid',grid_x,grid_y)

#Account for uncertainty
'''I think there should be an inversion (least squares) problem here to account for the uncertainty however I am not sure what the sensitivity matrix is for this problem and thus I am just matrix multiplying the magnetic data by the weighting matrix which is created from the uncertainty for the time being.'''

# set up the weighting matrix with 1/uncertainty 
W = np.eye(543)
std = 1/uncertainty
for i in range(len(mag1)):
    W[i,i] = std[i]

# include weighting matrix in final solution
final_mag = W@mag1

# plot the results
xinterp,yinterp = np.meshgrid(grid_x,grid_y)
plt.figure(1)
plt.contourf(xinterp,yinterp,final_mag)
plt.colorbar(label='MagAnomaly (nT)')
plt.title('Magnetic Survey')
plt.savefig('plots/anomaly.png')

# plot the results with the survey lines overlaid
plt.figure(2)
plt.contourf(xinterp,yinterp,final_mag)
plt.colorbar(label='MagAnomaly (nT)')
plt.scatter(x,y,color='k')
plt.title('Magnetic Survey With Survey Line')
plt.savefig('plots/anomaly_lines.png')
plt.show()
