"""All the relevant packages are imported for code below"""

import custom_cmap as make_me_colors # import custom color map script 
import go_chandra_analysis_tools as gca_tools # import the defined functions to analysis Chandra data nad perfrom coordinate transformations

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
from astropy.io import ascii
from astropy.io import fits as pyfits
from astropy.time import Time
from astropy.time import TimeDelta              # add/subtract time intervals 
import astropy.units as u
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
import os
import time
from datetime import datetime, timedelta


"""Setup the font used for plotting"""
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.labelsize']=14
matplotlib.rcParams['ytick.labelsize']=14
matplotlib.rcParams['agg.path.chunksize'] = 1000000

matplotlib.rcParams['pcolor.shading'] = 'auto'

# ## Making Polar Plots of X-ray emission from North and South Pole <br>
# 
# Polar plots are created for either the full observation or feach defined time interval. The user is prompted to set the max limit for the color bar used in the plots. The pltos are saved to the same folder as the corrected event file.
AU_2_km = 1.49598E+8

obsID = '23369'

# Reading in JRM33 code (provided by Dale) to plot Io and Ganymede footprints.
jrm33_N_ftp = ascii.read('/Users/mcentees/Desktop/paper2/jrm33_N.txt')
jrm33_S_ftp = ascii.read('/Users/mcentees/Desktop/paper2/jrm33_S.txt')

jrm33_N_io_lat = jrm33_N_ftp['col4']
jrm33_N_io_lon = (jrm33_N_ftp['col5'])

jrm33_S_io_lat = jrm33_S_ftp['col4']
jrm33_S_io_lon = (jrm33_S_ftp['col5'])

jrm33_N_gan_lat = jrm33_N_ftp['col8']
jrm33_N_gan_lon = (jrm33_N_ftp['col9'])

jrm33_S_gan_lat = jrm33_S_ftp['col8']
jrm33_S_gan_lon = (jrm33_S_ftp['col9'])

# Compute the difference between successive t2 values
diffs1_N = np.diff(jrm33_N_gan_lon)
diffs2_N = np.diff(jrm33_N_io_lon)

diffs1_S = np.diff(jrm33_S_gan_lon)
diffs2_S = np.diff(jrm33_S_io_lon)

# Find the differences that are greater than pi
discont_indices_N = np.where(np.abs(diffs1_N) > 100)[0]
discont_indices2_N = np.where(np.abs(diffs2_N) > 100)[0]

discont_indices_S = np.where(np.abs(diffs1_S) > 100)[0]
discont_indices2_S = np.where(np.abs(diffs2_S) > 100)[0]

# Set those t2 values to NaN
jrm33_N_gan_lat[discont_indices_N] = np.nan
jrm33_N_gan_lon[discont_indices2_N] = np.nan

jrm33_S_gan_lat[discont_indices_S] = np.nan
jrm33_S_gan_lon[discont_indices2_S] = np.nan

jrm33_N_io_lat[discont_indices_N] = np.nan
jrm33_N_io_lon[discont_indices2_N] = np.nan

jrm33_S_io_lat[discont_indices_S] = np.nan
jrm33_S_io_lon[discont_indices2_S] = np.nan

# Read in UV main oval (provided by Ben Swithenbank Harris)
uv_me = np.load("/Users/mcentees/Desktop/Chandra/refmon.npz")
uv_lat = uv_me['lat']
uv_lon = uv_me['lon']
uv_hilat = uv_me['hilat']
uv_hilon = uv_me['hilon']

uv_hilon_pos = []
for l in uv_hilon:
    if l < 0.0:
        uv_hilon_pos.append(l + 360.0)
    else:
        uv_hilon_pos.append(l)

# Denis's uv main emission code
uv_n_lon = np.array([200.0,140.0,141.0,150.0,149.3,155.0,160.0,170.0,180.0,190.0,200.0,210.0,220.0,230.0,240.0,250.0,260.0,270.0,280.0,290.0,201.0])
uv_n_lat = np.array([89.30,84.94,81.00,73.46,66.00,59.00,56.32,54.65,55.43,57.44,60.11,63.19,66.21,69.24,72.47,75.27,77.66,80.00,82.64,86.78,89.30])

uv_s_lon = np.array([20.00,30.00,40.00,50.00,60.00,70.00,80.00,90.00,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,210.0,220.0,230.0,240.0,250.0,260.0,270.0,280.0,290.0,300.0,310.0,320.0,330.0,340.0,350.0,360.0,21.00])
uv_s_lat = -1.*np.array([68.28,67.80,67.40,66.95,67.08,68.04,69.66,71.58,73.48,75.17,76.67,77.93,78.92,79.77,80.41,80.90,81.27,81.49,81.61,81.61,81.51,81.30,80.98,80.57,80.02,79.36,78.62,77.73,76.78,75.72,74.59,73.46,72.28,71.11,70.04,68.28])

phi_n = np.deg2rad(360 - uv_n_lon)
theta_n = np.deg2rad(90 - uv_n_lat)

phi_s = np.deg2rad(360 - uv_s_lon)
theta_s = np.deg2rad(90 - uv_s_lat)

orig_x_n = np.sin(theta_n) * np.cos(phi_n)
orig_y_n = np.sin(theta_n) * np.sin(phi_n)
orig_z_n = np.cos(theta_n)

orig_x_s = np.sin(theta_s) * np.cos(phi_s)
orig_y_s = np.sin(theta_s) * np.sin(phi_s)
orig_z_s = np.cos(theta_s)

sizefactor = 5.
orig_arr_n = np.arange(len(uv_n_lon))
out_arr_n = np.arange((len(uv_n_lon)-1)*sizefactor+1)/sizefactor

orig_arr_s = np.arange(len(uv_s_lon))
out_arr_s = np.arange((len(uv_s_lon)-1)*sizefactor+1)/sizefactor

splined_x_n = interpolate.interp1d(orig_arr_n, orig_x_n, kind='quadratic')(out_arr_n)
splined_y_n = interpolate.interp1d(orig_arr_n, orig_y_n, kind='quadratic')(out_arr_n)
splined_z_n = interpolate.interp1d(orig_arr_n, orig_z_n, kind='quadratic')(out_arr_n)

splined_x_s = interpolate.interp1d(orig_arr_s, orig_x_s, kind='quadratic')(out_arr_s)
splined_y_s = interpolate.interp1d(orig_arr_s, orig_y_s, kind='quadratic')(out_arr_s)
splined_z_s = interpolate.interp1d(orig_arr_s, orig_z_s, kind='quadratic')(out_arr_s)

splined_phi_n = np.arctan2(splined_y_n, splined_x_n)
splined_theta_n = np.arccos(splined_z_n)

splined_phi_s = np.arctan2(splined_y_s, splined_x_s)
splined_theta_s = np.arccos(splined_z_s)

splined_lon_n = -1. * (np.rad2deg(splined_phi_n) - 360.)
splined_lon_n = splined_lon_n % 360

splined_lat_n = -1. * (np.rad2deg(splined_theta_n) - 90.)

splined_lon_s = -1. * (np.rad2deg(splined_phi_s) - 360.)
splined_lon_s = splined_lon_s % 360

splined_lat_s = -1. * (np.rad2deg(splined_theta_s) - 90.)

# Read in fits file from correct path
folder_path = '/Users/mcentees/Desktop/Chandra/' + str(obsID) + '/primary'

r_np_ticks = np.arange(90,0,-10)
r_np_ticks = np.ma.masked_where(r_np_ticks < 40, r_np_ticks)
r_np_ticks = np.ma.masked_where(r_np_ticks == 90, r_np_ticks)

r_sp_ticks = np.arange(-90,0,10)
r_sp_ticks = np.ma.masked_where(r_sp_ticks > -40, r_sp_ticks)
r_sp_ticks = np.ma.masked_where(r_sp_ticks == -90, r_sp_ticks)

# Creating the custom color map for polar plots
c = colors.ColorConverter().to_rgb
custom_map = make_me_colors.make_cmap([c('white'), c('cyan'), 0.10, c('cyan'), c('blue'), 0.50, c('blue'), c('lime'), 0.90, c('lime')])
# The color strings can be changed to what the user desires! Default: color map used in the Weigt et al. 2020

# convert to X-ray brightness in Rayleighs - assuimg Aef = 40cm^2 (appropriate for 300eV X-rays)
conf = 4.0 * np.pi * 206264.806**2 / 1E6 / 1000 # convert flux -> Rayleighs (brightness)

rad_eq_0 = 71492.0 # radius of equator in km
rad_pole_0 = 66854.0 # radius of poles in km

ratio = rad_pole_0/rad_eq_0 # ratio of polar radius to equatorial radius

# Defining azimuth angle and distance in polar plot
azimuth = np.deg2rad(np.arange(0,361)) # azimuth = S3 longitude in this system
# R deifned for both North and South poles using latitude. North: (0,90), South: (0, -90) 
R_np = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(np.arange(0,91))-np.pi/2))))**2 + 1))
R_sp = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(np.arange(-90,1))-np.pi/2))))**2 + 1))

R_np = np.arctan(np.tan(R_np)/ratio**2)
R_sp = np.arctan(np.tan(R_sp)/ratio**2)

R_np_grid = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(np.arange(0,91,10))-np.pi/2))))**2 + 1))
R_np_grid = np.arctan(np.tan(R_np_grid)/ratio**2)
R_sp_grid = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(np.arange(-90,1,10))-np.pi/2))))**2 + 1))
R_sp_grid = np.arctan(np.tan(R_sp_grid)/ratio**2)

# Assumptions used for mapping:
scale = 0.13175 # scale used when observing Jupiter using Chandra - in units of arcsec/pixel
fwhm = 0.8 # FWHM of the HRC-I point spread function (PSF) - in units of arcsec
psfsize = 25 # size of PSF used - in units of arcsec
alt = 400 # altitude where X-ray emission is assumed to occur in Jupiter's ionosphere - in units of km

# Reading in photon list (after PI filter)
PI_file = pd.read_csv(str(folder_path) + f'/{obsID}_photonlist_PI_filter_Jup_full_10_250.txt')
PI_file.columns = PI_file.columns.str.replace("# ", "")
PI_tevts = PI_file['t(s)']
PI_xevts = PI_file['x(arcsec)']
PI_yevts = PI_file['y(arcsec)']
sup_lat_list = np.array(PI_file['lat (deg)'] + 90)
sup_lon_list = np.array(PI_file['SIII_lon (deg)'])

# Reading in Chandra Event fits file to extract start and end times of the observation.
hdulist = pyfits.open(f'/Users/mcentees/Desktop/Will_code/hrcf{obsID}_pytest_evt2_change.fits', dtype=float)
img_head = hdulist[1].header
cxo_tstart = Time(img_head['DATE-OBS']).iso[0:-4]
cxo_tend = Time(img_head['DATE-END']).iso[0:-4]
tstart_evt = img_head['TSTART']
hdulist.close()

evt_date = pd.to_datetime(cxo_tstart) #... and coverted to datetiem format to allow the relevant information to be read to...
evt_hour = evt_date.hour
evt_doy = evt_date.strftime('%j')
evt_mins = evt_date.minute
evt_secs = evt_date.second
evt_DOYFRAC = gca_tools.doy_frac(float(evt_doy), float(evt_hour), float(evt_mins), float(evt_secs))

"""Brad's horizons code to extract the ephemeris file"""

from astropy.time import Time                   #convert between different time coordinates
from astropy.time import TimeDelta              #add/subtract time intervals 
from astroquery.jplhorizons import Horizons     #automatically download ephemeris 

# The start and end times are taken from the horizons file.
dt = TimeDelta(0.125, format='jd')

# Below sets the parameters of what observer the ephemeris file is generated form. For example, '500' = centre of the Earth, '500@-151' = CXO
obj = Horizons(id=599,location='500@-151',epochs={'start':cxo_tstart, 'stop':(Time(cxo_tend)+dt).iso, 'step':'1m'}, id_type='majorbody')
eph_jup = obj.ephemerides()

# Extracts relevent information needed from ephermeris file
cml_spline_jup = scipy.interpolate.UnivariateSpline(eph_jup['datetime_jd'], eph_jup['PDObsLon'],k=1)
lt_jup = eph_jup['lighttime']
sub_obs_lon_jup = eph_jup['PDObsLon']
sub_obs_lat_jup = eph_jup['PDObsLat']

# Adding angular diameter from JPL Horizons to use later to define radius of circular region within which photons are kept
ang_diam = max(eph_jup['ang_width'])

# Also adding tilt angle of Jupiter with respect to true North Pole
tilt_ang = np.mean(eph_jup['NPole_ang'])

eph_dates = pd.to_datetime(eph_jup['datetime_str'])
eph_dates = pd.DatetimeIndex(eph_dates)
eph_doy = np.array(eph_dates.strftime('%j')).astype(int)
eph_hours = eph_dates.hour
eph_minutes = eph_dates.minute
eph_seconds = eph_dates.second

eph_DOYFRAC_jup = gca_tools.doy_frac(eph_doy, eph_hours, eph_minutes, eph_seconds) # DOY fraction from ephermeris data

jup_time = (eph_DOYFRAC_jup - evt_DOYFRAC)*86400.0 + tstart_evt # local tiem of Jupiter
j_rotate = np.rad2deg(1.758533641E-4) # Jupiter's rotation period

delta_mins = 595.5
PI_tevts_datetime = Time(PI_tevts, format='cxcsec').datetime

# from the start end end time of the photons detected, the time interval of dt minutes is created
obs_start_times = PI_tevts_datetime.min()
obs_end_times = PI_tevts_datetime.max()

time_interval = [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in gca_tools.datetime_range(obs_start_times,obs_end_times,timedelta(minutes=delta_mins))]
time_interval.append(obs_end_times.strftime('%Y-%m-%dT%H:%M:%S'))

time_interval_isot = Time(time_interval, format='isot')
time_interval_cxo = time_interval_isot.cxcsec

time_int_plot = Time(time_interval_isot, format='iso', out_subfmt='date_hm')
lt_jj = (TimeDelta((0) * u.min)).datetime
time_int_minusLT = Time((Time(time_int_plot) - lt_jj), format='iso', out_subfmt='date_hm')

label_left = ['(a)', '(c)', '(e)', '(g)']
label_right = ['(b)', '(d)', '(f)', '(h)']
title_posn = [0.89, 0.69, 0.49, 0.29]

fig = plt.figure(figsize=(15,20))
for m in range(len(time_interval_cxo) - 1):
    int_indx = np.where((PI_tevts >= time_interval_cxo[m]) & (PI_tevts <= time_interval_cxo[m+1]))[0]
    PI_tevts_int = np.array(PI_tevts[int_indx])
    PI_xevts_int = np.array(PI_xevts[int_indx])
    PI_yevts_int = np.array(PI_yevts[int_indx])
    sup_lat_list_int = np.array(sup_lat_list[int_indx])
    sup_lon_list_int = np.array(sup_lon_list[int_indx])

    """CODING THE SIII COORD TRANSFORMATION"""
    # define the local time and central meridian latitude (CML) during the observation  
    jup_cml_0 = float(eph_jup['PDObsLon'][0]) + j_rotate * (jup_time - jup_time[0])
    interpfunc_cml = interpolate.interp1d(jup_time, jup_cml_0)
    jup_cml = interpfunc_cml(PI_tevts_int)
    jup_cml = np.deg2rad(jup_cml % 360)
    # find the distance between Jupiter and Chandra throughout the observation, convert to km
    interpfunc_dist = interpolate.interp1d(jup_time, (eph_jup['delta'].astype(float))*AU_2_km)
    jup_dist = interpfunc_dist(PI_tevts_int)
    dist = sum(jup_dist)/len(jup_dist)
    kmtoarc = np.rad2deg(1.0/dist)*3.6E3 # convert from km to arc
    kmtopixels = kmtoarc/scale # convert from km to pixels using defined scale
    rad_eq_0 = 71492.0 # radius of equator in km
    rad_pole_0 = 66854.0 # radius of poles in km
    ecc = np.sqrt(1.0-(rad_pole_0/rad_eq_0)**2) # oblateness of Jupiter 
    rad_eq = rad_eq_0 * kmtopixels
    rad_pole = rad_pole_0 * kmtopixels # convert both radii form km -> pixels
    alt0 = alt * kmtopixels # altitude at which we think emission occurs - agreed in Southampton Nov 15th 2017

    # find sublat of Jupiter during each Chandra time interval
    interpfunc_sublat = interpolate.interp1d(jup_time, (sub_obs_lat_jup.astype(float)))
    jup_sublat = interpfunc_sublat(PI_tevts_int)
    # define the planetocentric S3 coordinates of Jupiter 
    phi1 = np.deg2rad(sum(jup_sublat)/len(jup_sublat))
    nn1 = rad_eq/np.sqrt(1.0 - (ecc*np.sin(phi1))**2)
    p = dist/rad_eq
    phig = phi1 - np.arcsin(nn1 * ecc**2 * np.sin(phi1)*np.cos(phi1)/p/rad_eq)
    h = p * rad_eq *np.cos(phig)/np.cos(phi1) - nn1
    interpfunc_nppa = interpolate.interp1d(jup_time, (eph_jup['NPole_ang'].astype(float)))
    jup_nppa = interpfunc_nppa(PI_tevts_int)
    gamma = np.deg2rad(sum(jup_nppa)/len(jup_nppa))
    omega = 0.0
    Del = 1.0

    # Define latitude and longitude for entire surface
    lat_test = np.zeros(360 * 181)
    lng_test = np.zeros(360 * 181)
    j_test = np.arange(181)

    for i in range(360):
        lat_test[j_test * 360 + i] = j_test - 90
        lng_test[j_test * 360 + i] = i


    # perform coordinate transfromation from planetocentric -> planetographic (taking into account the oblateness of Jupiter
    # when defining the surface features)
    coord_transfo_test = gca_tools.ltln2xy(alt=alt0, re0=rad_eq_0, rp0=rad_pole_0, r=rad_eq, e=ecc, h=h, phi1=phi1, phig=phig, lambda0=0.0, p=p, d=dist, gamma=gamma,            omega=omega, latc=np.deg2rad(lat_test), lon=np.deg2rad(lng_test))

    # Assign the corrected transformed position of the X-ray emission
    xt_test = coord_transfo_test[0]
    yt_test = coord_transfo_test[1]
    cosc_test = coord_transfo_test[2]
    condition_test = coord_transfo_test[3]
    count_test = coord_transfo_test[4]

    # Find latiutde and lonfitude of the surface features
    laton_test = lat_test[condition_test] + 90
    lngon_test = lng_test[condition_test]

    # Creating 2D array of the properties and time properties
    props_test = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    timeprops_test = np.zeros((int(360) // int(Del), int(180) // int(Del) + int(1)))
    num_test = len(PI_tevts_int)
    # define a Gaussian PSF for the instrument
    psfn = np.pi*(fwhm / (2.0 * np.sqrt(np.log(2.0))))**2
    # create a grid for the position of the properties

    # Equations for defining ellipse region
    tilt_ang_rad = np.deg2rad(tilt_ang)
    R_eq_as = (ang_diam/2.)/np.cos(tilt_ang_rad) # equatorial radius of Jupiter in arcsecs
    R_pol_as = R_eq_as * np.sqrt(1 - ecc**2) # polar radius of Jupiter in arcsecs

    for k in range(0,num_test):

        # convert (x,y) position to pixels
        xpi_test = (PI_xevts_int[k]/scale)
        ypi_test = (PI_yevts_int[k]/scale)


        cmlpi_test = (np.rad2deg(jup_cml[k]))#.astype(int)

        xtj_test = xt_test[condition_test]
        ytj_test = yt_test[condition_test]
        latj_test = (laton_test.astype(int)) % 180
        lonj_test = ((lngon_test + cmlpi_test.astype(int) + 360.0).astype(int)) % 360
        dd_test = np.sqrt((xpi_test - xtj_test)**2 + (ypi_test - ytj_test)**2) * scale
        psfdd_test = np.exp(-(dd_test/ (fwhm / (2.0 * np.sqrt(np.log(2.0)))))**2) / psfn # define PSF of instrument

        psf_max_cond_test = np.where(psfdd_test == max(psfdd_test))[0] # finds the max PSF over each point in the grid
        count_mx_test = np.count_nonzero(psf_max_cond_test)
        if count_mx_test != 1: # ignore points where there are 2 cases of the same max PSF
            # print('2 cases with same max psf')
            continue
        else:

            props_test[lonj_test,latj_test] = props_test[lonj_test,latj_test] + psfdd_test # assign the 2D PSF to the each point in the grid

    # effectively, do the same idea except for exposure time

    # interval = Time(obs_end_times).cxcsec - Time(obs_start_times).cxcsec
    interval = time_interval_cxo[m+1] - time_interval_cxo[m] 
    # interval = delta_mins * 60 

    if interval > 1000.0:
        step = interval/100.0
    elif interval > 100.0:
        step = interval/10.0
    else:
        step = interval/2.0

    time_vals_test = np.arange(round(int(interval/step)))*step + step/2 + Time(obs_start_times).cxcsec

    interpfunc_time_cml_test = interpolate.interp1d(jup_time,jup_cml_0)
    time_cml_test = interpfunc_time_cml_test(time_vals_test)

    for j in range(0, len(time_vals_test)):
        timeprops_test[((lngon_test + time_cml_test[j].astype(int))%360).astype(int),laton_test.astype(int)] = timeprops_test[((lngon_test + time_cml_test[j].astype(int))%360).astype(int),laton_test.astype(int)] + step


    # applying the conversion from a flux -> Rayleighs for the 2D PSFs
    # make grids first
    polar_props = props_test.T
    sp_polar_props = polar_props[1:91] # -90 latitude gives error
    np_polar_props = polar_props[90:-1]

    time_props = timeprops_test.T
    sp_time_props = time_props[1:91]
    np_time_props = time_props[90:-1]

    bright_props = polar_props/(time_props + 0.001) * conf
    sp_bright_props = bright_props[1:91]
    np_bright_props = bright_props[90:-1]

    # print(f'North pole max brightness: {np.round(np.nanmax(np_bright_props), 2)}')
    # print(f'South pole max brightness: {np.round(np.nanmax(sp_bright_props), 2)}')

    # max_brightness = np.round(np.max([np.max(np_bright_props), np.max(sp_bright_props)]), 1)
    max_brightness = 0.8

    # split into brightnesses for North and South pole
    north_photons_pos = np.where(sup_lat_list_int >= 90)[0]
    south_photons_pos = np.where(sup_lat_list_int <= 90)[0]
    # perfoming coordinate transformation on the photon position to be plotted on polar plot
    az_scat = np.deg2rad(sup_lon_list_int)
    R_scat_np = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(sup_lat_list_int[north_photons_pos])-np.pi))))**2 + 1))
    R_scat_np = np.arctan(np.tan(R_scat_np)/ratio**2)
    R_scat_sp = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(sup_lat_list_int[south_photons_pos])-np.pi))))**2 + 1))
    R_scat_sp = np.arctan(np.tan(R_scat_sp)/ratio**2)

    # Including UV main oval
    # uv_scat = np.deg2rad(uv_hilon_pos)
    # R_uv = np.sqrt(1/(1/(ratio*np.tan((np.deg2rad(uv_hilat)-np.pi)))**2 + 1))
    uv_scat = np.deg2rad(uv_hilon) # + 180.0)
    R_uv = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(uv_hilat) - np.pi/2))))**2 + 1))

    # Including Denis's northern main oval
    uv_n_scat = np.deg2rad(uv_n_lon)
    R_n_uv = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(uv_n_lat) - np.pi/2))))**2 + 1))

    uv_n_scat_spl = np.deg2rad(splined_lon_n)
    R_n_uv_spl = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(splined_lat_n) - np.pi/2))))**2 + 1))

    # Including Denis's suthern main oval
    uv_s_scat = np.deg2rad(uv_s_lon)
    R_s_uv = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(uv_s_lat) - np.pi/2))))**2 + 1))

    uv_s_scat_spl = np.deg2rad(splined_lon_s)
    R_s_uv_spl = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(splined_lat_s) - np.pi/2))))**2 + 1))

    # Including Io and Ganymede footprints in north and south
    io_N_scat = np.deg2rad(jrm33_N_io_lon)
    R_N_io = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(jrm33_N_io_lat) - np.pi/2))))**2 + 1))

    io_S_scat = np.deg2rad(jrm33_S_io_lon)
    R_S_io = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(jrm33_S_io_lat) - np.pi/2))))**2 + 1))

    gan_N_scat = np.deg2rad(jrm33_N_gan_lon)
    R_N_gan = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(jrm33_N_gan_lat) - np.pi/2))))**2 + 1))

    gan_S_scat = np.deg2rad(jrm33_S_gan_lon)
    R_S_gan = np.sqrt(1/(1/(ratio*np.tan(((np.deg2rad(jrm33_S_gan_lat) - np.pi/2))))**2 + 1))

    # creating figure for North and South polar plots
    # fig = plt.figure(figsize=(10,10))
    # left polar plot is North pole
    # ax1 = plt.subplot(121, projection="polar")
    ax1 = plt.subplot(len(time_interval_cxo)-1, 2, 2*m+1, projection="polar")
    # creating the 2D polar plot with transformed values and custom color map...
    mesh_np = plt.pcolormesh(azimuth, R_np, np_bright_props, cmap = custom_map, norm=colors.PowerNorm(gamma=0.7, vmin=0, vmax=max_brightness))
    #...with a scatter plot of the photon positions
    # ax1.scatter(az_scat[north_photons_pos], R_scat_np, s=5, color='black', alpha = 0.2)
    # editing polar plot to show 180 degrees at the top for North, setting the appropriate limits and other plotting functions
    ax1.set_theta_direction(-1)
    ax1.set_theta_offset(-np.pi/2)
    R_lim = np.quantile(R_np, 0.6)
    ax1.set_ylim(0, R_lim)
    # ax1.set_ylim(0, max(R_np))
    #ax1.set_yticks(np.arange(0, 91, 10))
    #ax1.set_yticklabels(ax1.get_yticks()[::-1])
    #ax1.set_yticklabels([])
    # ax1.set_title('North Pole', size = 16, y=1.08)
    # ax1.plot(uv_n_scat, R_n_uv, 'k', zorder=10)
    # ax1.plot(uv_n_scat_spl, R_n_uv_spl, 'k', zorder=10)
    ax1.plot(io_N_scat, R_N_io, 'k--', zorder=10)
    ax1.plot(gan_N_scat, R_N_gan, 'k', zorder=10)
    ax1.set_yticklabels([])
    for kk in range(len(R_np_grid)):
        ax1.plot(2*np.pi - np.linspace(0, 2*np.pi, 100), np.full(100,R_np_grid[kk]), color='k', alpha=0.125)
    for jj in range(len(r_np_ticks)):
        ax1.text(np.deg2rad(45), np.full(100,sorted(R_np_grid)[jj])[0]+0.02, r_np_ticks[jj], alpha=0.5)
    ax1.grid(axis='x', alpha=0.375)

    # South pole is axis on the right
    ax2 = plt.subplot(len(time_interval_cxo)-1, 2, 2*m+2, projection="polar")
    # ax2 = plt.subplot(122, projection="polar")
    mesh = plt.pcolormesh(azimuth, R_sp, sp_bright_props, cmap=custom_map, norm=colors.PowerNorm(gamma=0.7, vmin=0, vmax=max_brightness))
    # ax2.scatter(az_scat[south_photons_pos], R_scat_sp, s=5, color='black', alpha=0.2)
    #... except 0 degress is pointed at the top for South pole
    # ax2.plot(uv_s_scat, R_s_uv, 'k', zorder=10)
    # ax2.plot(uv_s_scat_spl, R_s_uv_spl, 'k', zorder=10)
    io_l, = ax2.plot(io_S_scat, R_S_io, 'k--', zorder=10, label='Io footprint')
    gan_l, = ax2.plot(gan_S_scat, R_S_gan, 'k', zorder=10, label='Ganymede footprint')
    ax2.set_theta_offset(+np.pi/2)
    ax2.set_ylim(0, R_lim)
    # ax2.set_ylim(0,max(R_sp))
    # ax2.set_yticks(np.arange(0, 91, 10))
    #ax2.set_yticklabels(ax2.get_yticks()[::-1])
    #  ax2.set_yticklabels([])
    # text_arr = ' ' * 68
    text_arr = ' ' * 100
    # ax3 = plt.subplot(len(time_interval_cxo)-1, 1, m+1)
    if m == 0:
        # ax3.set_title(f'({time_int_minusLT[m]} - {time_int_minusLT[m+1]})', size=16, y=1.1)
        ax1.set_title('\nNorth Pole', size=16, y=1.1)
        ax2.set_title('\nSouth Pole', size=16, y=1.1)
        # plt.legend([io_l, gan_l], ['Io footprint', 'Ganymede footprint'], loc='upper center', fontsize=12, ncols=2, handlelength=0.5, bbox_to_anchor=(0.51, 0.93))
    # else:
        # ax3.set_title(f'({time_int_minusLT[m]} - {time_int_minusLT[m+1]})', size=16, y=1.10)
    fig.text(x=0.51, y=title_posn[m], s=f'JR{m+1}\n({time_int_minusLT[m]} - {time_int_minusLT[m+1]})', size=16, ha='center', va='center', color='darkviolet',weight='bold')
    fig.text(x=0.2, y=title_posn[m], s=label_left[m], size=16, ha='center', va='center')
    fig.text(x=0.83, y=title_posn[m], s=label_right[m], size=16, ha='center', va='center')
    # ax2.set_title('South Pole', size = 16, y=1.08)
    ax2.set_yticklabels([])
    for kk in range(len(R_sp_grid)):
        ax2.plot(2*np.pi - np.linspace(0, 2*np.pi, 100), np.full(100,R_sp_grid[kk]), color='k', alpha=0.125)
        # ax2.plot(2*np.pi - np.linspace(0, 2*np.pi, 100), np.full(100,R_sp_grid[kk]), color='k', alpha=0.075)
    for jj in range(len(r_sp_ticks)):
        ax2.text(np.deg2rad(135), np.full(100,sorted(R_sp_grid)[jj])[0]+0.02, r_sp_ticks[jj], alpha=0.5)
    ax2.grid(axis='x', alpha=0.375)
    # plt.tight_layout()
    # creating and formatting the color bar at the bottom of the plot
    if m == 3:
        # fig.subplots_adjust()
        cbar_ax = fig.add_axes([0.20, 0.05, 0.63, 0.025])
        cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal",fraction=0.2)
        cbar.set_label('Brightness (R)', labelpad=2, size=16)

        # creating the title for the polar plots
        fig.subplots_adjust(hspace=0.3)
        # fig.subplots_adjust(top=1.3)
        # fig.suptitle(f'Chandra X-ray Jupiter Polar Maps - ObsID: {obsID} \n({cxo_tstart} - {cxo_tend})', size=16)
        # fig.suptitle(f'Chandra X-ray Jupiter Polar Maps - ObsID: {obsID} \n\n({time_int_plot[0]} - {time_int_plot[1]})', size=16, y=0.95)
        # fig.suptitle(f'Chandra X-ray Jupiter Polar Maps - ObsID: {obsID}', size=16, x=0.5, y=0.95, ha='center', va='center')
        # plt.suptitle(f'Chandra X-ray Jupiter Polar Maps - ObsID: {obsID}', size=16, x=0.51, y=0.95, ha='center', va='center')
        # save polar plots to same folder as event file
        # plt.savefig(str(folder_path) + f'/{obsID}_polar_plot_PI_10_250_JR{m+1}.png', bbox_inches='tight', dpi=500)
        # plt.savefig(f'/Users/mcentees/Desktop/paper2/{obsID}_polar_plot_PI_10_250_8_panel_plot_test.png', dpi=500)
        # plt.savefig(f'/Users/mcentees/Desktop/paper2/{obsID}_polar_plot_PI_10_250_8_panel_plot_lt_no_title_labels.png', bbox_inches='tight', dpi=500)
        plt.savefig(f'/Users/mcentees/Desktop/paper2/{obsID}_polar_plot_PI_10_250_8_panel_plot_lt_no_title_labels_ftp.png', bbox_inches='tight', dpi=500)
plt.show()






