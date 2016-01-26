
from netCDF4 import Dataset as netcdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc

from mod_LN87_v7 import fLN87, fuvtodiv, fzonalsmooth

def fncread(fn,var):
    # read a variable from a netcdf file
    fid = netcdf(fn, 'r')
    data = fid.variables[var][:]
    fid.close()
    return data

def fJJA(data):
    # get summer (JJA) climatologies
    ss = (np.nanmean(data[5::12],axis=0) + np.nanmean(data[6::12],axis=0) + np.nanmean(data[7::12],axis=0))/3.0
    return ss

#===============================================================================
# data processing

pin = '/Users/Oscar/yhy/Work/Data/ERA_Interim/data/'    # ERA-Interim data path

fin = 'ERA_T2m_197901_201505.nc'                # temperature at 2 meter (K)
print 'Reading '+pin+fin, ' ... '
ts0 = fncread(pin+fin,'t2m')[:,::-1,:]          # [time,lat,lon], s->n
lat = fncread(pin+fin,'latitude')[::-1]         # [time,lat,lon], s->n
lon = fncread(pin+fin,'longitude')

fin = 'ERA_q2m_197901_201505.nc'                # specific humidity at 2 meter (kg/kg)
print 'Reading '+pin+fin, ' ... '
qs  = fncread(pin+fin,'q2m')[:,::-1,:]          # [time,lat,lon], s->n

fin = 'ERA_wind10m_197901_201505.nc'            # wind at 10 meter (m/s)
print 'Reading '+pin+fin, ' ... '
u10 = fJJA(fncread(pin+fin,'u10')[:,::-1,:])    # [lat,lon], s->n
v10 = fJJA(fncread(pin+fin,'v10')[:,::-1,:])    # [lat,lon], s->n

fin = 'ERA_SLP_197901_201505.nc'                # sea-level pressure (Pa)
print 'Reading '+pin+fin, ' ... '
slp = fJJA(fncread(pin+fin,'msl')[:,::-1,:])    # [lat,lon], s->n


print 'Running LN87 model ...'

# virsual T (K), [lat,lon]
ts  = fJJA(ts0) * (1+0.608*fJJA(qs))

# call LN87 model
ug,vg,hg,psg = fLN87(fzonalsmooth(ts),lon,lat,ntrun=np.size(lon)/2)

# calculate the eddy component to compare with simulation
um = np.nanmean(u10,axis=1)
vm = np.nanmean(v10,axis=1)
pm = np.nanmean(slp,axis=1)
for ilat in range(np.size(lat)):
    u10[ilat,:] -= um[ilat]
    v10[ilat,:] -= vm[ilat]
    slp[ilat,:] -= pm[ilat]

# diagnostic variable: divergece (1/s)
divg  = fuvtodiv(ug,vg,lon,lat)
div10 = fuvtodiv(u10,v10,lon,lat)

#===============================================================================
# plots

ffig = 'fig_main_test.png'
print 'Plots output: ',ffig
fig  = plt.figure(1,figsize=(12,10))

# colorbar scale
clev = np.linspace(-10,10,21)

for i in range(8):
    if i==0:
        pic  = u10
        tstr = 'a) ERA_Interim: U10 ($m \ s^{-1}$) JJA'
    if i==1:
        pic  = ug
        tstr = 'b) LN87 simulation: U ($m \ s^{-1}$) JJA'
    if i==2:
        pic  = v10
        tstr = 'c) ERA_Interim: V10 ($m \ s^{-1}$) JJA'
    if i==3:
        pic  = vg
        tstr = 'd) LN87 simulation: V ($m \ s^{-1}$) JJA'
    if i==4:
        pic  = div10 * 1e6
        tstr = 'e) ERA_Interim: DIV ($10^{-6} \ s^{-1}$) JJA'
    if i==5:
        pic  = divg * 1e6
        tstr = 'f) LN87 simulation: DIV ($10^{-6} \ s^{-1}$) JJA'
    if i==6:
        pic  = slp/100.0
        tstr = 'g) ERA_Interim: SLP (hPa) JJA'
    if i==7:
        pic  = psg/100.0
        tstr = 'h) LN87 simulation: Ps (hPa) JJA'

    plt.subplot(4,2,i+1)
    plt.contourf(lon,lat,pic, clev, norm = mc.BoundaryNorm(clev, 256), cmap = cm.jet)
    plt.colorbar()
    plt.contour(lon,lat,pic, [0], colors='k')
    plt.axis([100,280,-30,30])
    plt.title(tstr,loc='left')

# output plots
plt.tight_layout()
plt.savefig(ffig)
plt.close(1)

        
