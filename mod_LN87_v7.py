
import numpy as np

def fdiff_lat_1d(x,lat):
    # One dimensional meridional gradient
    # x: 1d data
    # lat: latitude

    dx = x * 0
    dx[1:-1] = (x[2:]-x[0:-2]) / ((lat[2:]-lat[0:-2])*np.pi/180.0)
    dx[0]    = (x[1]-x[0])     / ((lat[1]-lat[0])*np.pi/180.0)
    dx[-1]   = (x[-1]-x[-2])   / ((lat[-1]-lat[-2])*np.pi/180.0)
    return dx


def fdiff_lon_1d(x,lon):
    # One dimensional zonal gradient
    # x: 1d data
    # lon: longitude

    dx = x * 0
    dx[1:-1] = (x[2:]-x[0:-2]) / ((lon[2:]-lon[0:-2])*np.pi/180.0)
    dx[0]    = (x[1]-x[0])     / ((lon[1]-lon[0]+360)*np.pi/180.0)
    dx[-1]   = (x[-1]-x[-2])   / ((lon[-1]-lon[-2]+360)*np.pi/180.0)
    return dx


def fdiff_lat(x,lat2d):
    # Two dimensional meridional gradient
    # x: 2d data
    # lat2d: 2d latitude
    dx = x * 0
    dx[1:-1,:] = (x[2:,:] - x[0:-2,:]) / ((lat2d[2:,:] - lat2d[0:-2,:]) * np.pi/180.0)
    dx[0,:]    = (x[1,:]  - x[0,:])    / ((lat2d[1,:]  - lat2d[0,:]) * np.pi/180.0)
    dx[-1,:]   = (x[-1,:] - x[-2,:])   / ((lat2d[-1,:] - lat2d[-2,:]) * np.pi/180.0)
    return dx


def fdiff_lon(x,lon2d):
    # Two dimensional zonal gradient
    # x: 2d data
    # lon2d: 2d longitude

    dx = x * 0
    dx[:,1:-1] = (x[:,2:] - x[:,0:-2]) / ((lon2d[:,2:] - lon2d[:,0:-2]) * np.pi/180.0)
    dx[:,0]    = (x[:,1]  - x[:,-1])   / ((lon2d[:,1]  - lon2d[:,-1] + 360) * np.pi/180.0)
    dx[:,-1]   = (x[:,0]  - x[:,-2])   / ((lon2d[:,0]  - lon2d[:,-2] + 360) * np.pi/180.0)
    return dx


def fuvtodiv(u,v,lon,lat):
    # Calculate the divergence div (1/s), given zonal wind u (m/s) and meridional wind v (m/s)
    # lon: 1d longitude
    # lat: 1d latitude

    nlon = np.size(lon)
    nlat = np.size(lat)
    a = 6371000.0       # Earth's radius (m)
    lat2d  = u * 0
    lon2d  = u * 0
    clat2d = u * 0      # cosine(latitude)
    for ilat in range(nlat):
        lat2d[ilat,:]  = lat[ilat]
        lon2d[ilat,:]  = lon + 0
        clat2d[ilat,:] = np.cos(np.pi*lat[ilat]/180.0)

    divlon = fdiff_lon(u,lon2d)/a/clat2d
    divlat = fdiff_lat(v*clat2d,lat2d)/a/clat2d
    div = divlon + divlat
    return div


def fzonalsmooth(x):
    # 3-point zonal smoothing on x
    z = x * 0
    z[:,1:-1] = 0.5*x[:,1:-1] + 0.25*(x[:,:-2]+x[:,2:])
    z[:,0] = 0.5*x[:,0] + 0.25*(x[:,1]+x[:,-1])
    z[:,-1] = 0.5*x[:,-1] + 0.25*(x[:,-2]+x[:,0])
    return z


def fLN87(ts, lon, lat, h0=3000, tau=30*60, ntrun=15, eps=1/(2.5*86400)):
    # linear wind balanced model, based on Lindzen and Nigam, 1987

    # -------------------  input arguments  ---------------------
    #   ts:  2d surface virsual temperture      [K]
    #   lon: 1d longitude
    #   lat: 1d latitude
    #   h0: mean top of boundary layer          [m]
    #   tau: convection adjustment time scale   [s]
    #   ntrun: truncated zonal wavenumber
    #   eps: drag coefficient                   [1/s]

    # -------------------  output arguments  ---------------------
    #   ug: eddy zonal wind             (m/s) 
    #   vg: eddy meridional wind        (m/s)
    #   hg: back-pressure field         (m)
    #   psg: eddy sea-level pressure    (Pa)


    # prevent cosine(latitude) reaching zero
    if (abs(lat[0]) - 90 < 0.01):
        lat[0] = 0.5*(lat[0] + lat[1])
    if (abs(lat[-1]) - 90 < 0.01):
        lat[-1] = 0.5*(lat[-1] + lat[-2])

    nlon = np.size(lon)
    nlat = np.size(lat)
    ntrun = min(ntrun,nlon/2)

    T0 = 288.0      # reference temperature: K
    rou0 = 1.225    # reference densisty: kg/m3
    alpha = 0.003   # vertical lapse rate of mean temperature: K/m
    gama  = 0.3     # vertical lapse rate of perturbation temperature
    omg = 7.272e-5  # Earth rotation angular velocity (1/s)
    a = 6371000.0   # Earth radius (m)
    g = 9.8         # gravitation (m/s2)
    n = 1.0 / T0
    B = g*n*h0*(1.0 - 2.0*gama/3.0)/(2.0*a)

    # specific zonal wavenumbers to match the FFT
    #   Indexes ------ [0,1,2, ..., nlon/2-1,  nlon/2,    nlon/2+1, ..., nlon-2, nlon-1]
    #   Wavenumbers -- [0,1,2, ..., nlon/2-1, -nlon/2, -(nlon/2-1), ...,     -2,     -1]
    m = np.linspace(0,nlon-1,nlon)
    m[nlon/2:nlon] = -1 * np.linspace(1,nlon/2,nlon/2)[::-1]

    clat  = np.cos(np.pi*lat/180.0)         # cosine(latitude)
    flat  = 2*omg*np.sin(np.pi*lat/180.0)   # Coriolis parameter (1/s)
    tsm   = np.mean(ts,axis=1)              # zonal mean temperature (K)
    dtsmdlat = fdiff_lat_1d(tsm,lat)        # meridional gradient of zonal mean temperature (K/m)
    Alat = g*(2.0 - n*tsm + n*alpha*h0) / a

    lat2d  = ts * 0
    lon2d  = ts * 0
    clat2d = ts * 0
    tsm2d  = ts * 0
    A2d    = ts * 0

    for ilat in range(nlat):
        lat2d[ilat,:]  = lat[ilat]
        lon2d[ilat,:]  = lon + 0
        clat2d[ilat,:] = np.cos(np.pi*lat[ilat]/180.0)
        tsm2d[ilat,:]  = np.mean(ts[ilat,:])
        A2d[ilat,:]    = g*(2.0 - n*tsm2d[ilat,:] + n*alpha*h0) / a

    tsp      = ts - tsm2d   # temperature perturbation from zonal mean (K)
    dtsdlon  = fdiff_lon(tsp,lon2d) # zonal gradient of temperature perturbation (K/m)
    dtsdlat  = fdiff_lat(tsp,lat2d) # meridional gradient of temperature perturbation (K/m)

    Flon = np.fft.fft(B*dtsdlon/clat2d,axis=1)  # forcing term 1 (in spectral space)
    Flat = np.fft.fft(B*dtsdlat,axis=1)         # forcing term 2 (in spectral space)


    # rearrange the matrix and vectors, and solve the algebra equations
    ug = 0.0 * ts
    vg = 0.0 * ts
    hg = 0.0 * ts
    psg = 0.0 * ts
    us = (0 + 0j) * ts
    vs = (0 + 0j) * ts
    hs = (0 + 0j) * ts

    x0 = np.zeros([3*nlat,3*nlat]) + (0+0j)
    y0 = np.zeros([3*nlat]) + (0+0j)
    x0[0,0] = 1
    y0[0]   = 0
    x0[nlat-1,nlat-1] = 1
    y0[nlat-1]        = 0
    x0[nlat,nlat] = 1
    y0[nlat]      = 0
    x0[2*nlat-1,2*nlat-1] = 1
    y0[2*nlat-1]          = 0
    x0[2*nlat,2*nlat] = 1
    y0[2*nlat]        = 0
    x0[3*nlat-1,3*nlat-1] = 1
    y0[3*nlat-1]          = 0

    # Specify the coefficients in the 3*3 block matrix, except for the dependent values on wavenumber m

    for ilat in range(1,nlat-1):
        # the first row, except for the coefficients of h
        i = ilat
        x0[i,i] = eps
        x0[i,i+nlat] = -flat[ilat]

    for ilat in range(1,nlat-1):
        # the second row
        i = ilat + nlat
        x0[i,i-nlat] = flat[ilat]
        x0[i,i] = eps
        x0[i,i+nlat-1] = -Alat[ilat]/((lat[ilat+1]-lat[ilat-1]) * np.pi/180.0)
        x0[i,i+nlat]   = -0.5*g*n* dtsmdlat[ilat] / a
        x0[i,i+nlat+1] =  Alat[ilat]/((lat[ilat+1]-lat[ilat-1]) * np.pi/180.0)

    for ilat in range(1,nlat-1):
        # the third row, except for the coefficient of u
        i = ilat + 2*nlat
        x0[i,i-nlat-1] = -clat[ilat-1]/((lat[ilat+1]-lat[ilat-1]) * np.pi/180.0)
        x0[i,i-nlat+1] =  clat[ilat+1]/((lat[ilat+1]-lat[ilat-1]) * np.pi/180.0)
        x0[i,i] = a*clat[ilat]/(tau*h0)
        y0[i] = 0 + 0j

    # solve the algebra equation on truncated range of wavenumber: [-M,M] without m=0
    for im in range(1,ntrun+1) + range(nlon-ntrun,nlon):        
        x = x0 + 0
        y = y0 + 0

        for ilat in range(1,nlat-1):
            # the coefficients that depends on wavenumber m (zonal gradient terms)
            x[ilat,ilat+2*nlat] = 1.0j * m[im] * Alat[ilat]/clat[ilat]
            x[ilat+2*nlat,ilat] = 1.0j * m[im]

        # forcing terms
        y[1:nlat-1] = Flon[1:nlat-1,im] + 0
        y[nlat+1:2*nlat-1] = Flat[1:nlat-1,im] + 0

        # solve the algebra equations
        temp = np.linalg.solve(x,y)

        # u,v,h in spectral space
        us[:,im] = temp[0:nlat] + 0
        vs[:,im] = temp[nlat:2*nlat] + 0
        hs[:,im] = temp[2*nlat:3*nlat] + 0

    # get the values on grid points (physical space)
    for ilat in range(nlat):
        ug[ilat,:] = np.fft.ifft(us[ilat,:]).real
        vg[ilat,:] = np.fft.ifft(vs[ilat,:]).real
        hg[ilat,:] = np.fft.ifft(hs[ilat,:]).real
        psg[ilat,:] = g*rou0*n*h0*(gama/2.0 - 1)*tsp[ilat,:] + g*rou0*(2 - n*tsm[ilat] + n*alpha*h0)*hg[ilat,:]
    
    return  ug,vg,hg,psg



