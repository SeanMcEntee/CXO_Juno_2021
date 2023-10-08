import numpy as np
import datetime
import scipy
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import matplotlib.dates as mdates
from astropy.time import Time
import astropy.units as u
from astropy.time import TimeDelta              # add/subtract time intervals 

import pywt

from astroquery.jplhorizons import Horizons     # automatically download ephemeris 
# Need to do this to fix astroquery bug, otherwise it won't find the ephemeris data
from astroquery.jplhorizons import conf

import statsmodels
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf, pacf
import scipy.stats as ss

# Tell Python where to find your data
obsID = '23369'

indir = f"/Users/mcentees/Desktop/Chandra/{obsID}/primary"
# Tell Python where to save your plots
outdir = "/Users/mcentees/Desktop/paper2"


# Find and read in North and South Photon lists
alldata = pd.read_csv(indir + f'/{obsID}_photonlist_PI_filter_Jup_full_10_250.txt')
alldata.columns = alldata.columns.str.replace("# ", "")
northdata = pd.read_csv(indir + f'/{obsID}_photonlist_PI_filter_North.txt')
northdata.columns = northdata.columns.str.replace("# ", "")
southdata = pd.read_csv(indir + f'/{obsID}_photonlist_PI_filter_South.txt')
southdata.columns = southdata.columns.str.replace("# ", "")
# Write an observation date for plot titles
# Choose a bin size
binsizeinseconds = 120

# Pull start and end times from first photon in list
obs_start = min(alldata['t(s)'])
obs_end = max(alldata['t(s)'])
# obs_start = min(min(northdata['t(s)']), min(southdata['t(s)']))
# obs_end = max(max(northdata['t(s)']), max(southdata['t(s)']))
tstart = Time(obs_start, format='cxcsec')
tstop = Time(obs_end, format='cxcsec')

# create lists of binned data points
ncounts = []
scounts = []
times = []
ndata = northdata['t(s)'].values
sdata = southdata['t(s)'].values

for t in range(int(obs_start), int(obs_end), binsizeinseconds):
    ntemptimes = ndata[(t < ndata) & (ndata < (t+binsizeinseconds))]
    stemptimes = sdata[(t < sdata) & (sdata < (t+binsizeinseconds))]
    times.append((t+t+binsizeinseconds)/2)
    ncounts.append(len(ntemptimes))
    scounts.append(len(stemptimes))

ncountserrors = np.sqrt(np.array(ncounts))
scountserrors = np.sqrt(np.array(scounts))

# Define start and stop times for ephemeris data; since jpl does not accept seconds, 

# all times are in YY:MM:DD hh:mm format;dt is added to stop time to ensure ephemeris 
# data range extends beyond exposure time 

# Jupiter's ID is 599, XMM's location is'500@-125989', planet's are classed as a majorbody

eph_tstart = Time(tstart)
dt = TimeDelta(0.125, format='jd')
eph_tstop = Time(tstop + dt)

obj = Horizons(id=599, location='500@-151', epochs={'start': eph_tstart.iso, 'stop': eph_tstop.iso, 'step': '1m'}, id_type='majorbody')
obj_jj = Horizons(id=599, location='500@-61', epochs={'start': eph_tstart.iso, 'stop': eph_tstop.iso, 'step': '1m'}, id_type='majorbody')

eph = obj.ephemerides()
eph_jj = obj_jj.ephemerides()

# pull the mean light travel time
lighttravel_jup = eph['lighttime'].mean()
lighttravel_jj = eph_jj['lighttime'].mean()

lighttravel = lighttravel_jup - lighttravel_jj

# lighttravel = 34.2678894435478

# read in NE photon lists to get time window boundaries
bounds_NE = []

for i in range(4):
    NE_data = pd.read_csv(indir + f'/{obsID}_photonlist_PI_filter_NE{i+1}.txt')
    NE_data.columns = NE_data.columns.str.replace("# ", "")
    NE_time = np.array(NE_data['t(s)'])
    bounds_NE.append(NE_time[0])
    bounds_NE.append(NE_time[-1])

bounds_NE_arr = np.array(bounds_NE)
bounds_NE_datetime = [Time(x, format='cxcsec').datetime for x in bounds_NE_arr]
bounds_NE_minusLT = [Time(x - lighttravel*60, format='cxcsec').datetime for x in bounds_NE_arr]

# Converting times from Chandra/XMM time to yr,m,d,h,m
np_times = np.array(times)
timeincxo = Time(np_times, format='cxcsec')
timeindatetime = timeincxo.datetime
np_TminusLT = np.array(times) - (lighttravel*60)
tminusLTincxo = Time(np_TminusLT, format='cxcsec')
tminusLTindatetime = tminusLTincxo.datetime

# Define a function for CML throughout the observation using a spline (can equally interpolate etc.)
cml_spline = scipy.interpolate.UnivariateSpline(eph['datetime_jd'], eph['PDObsLon'], k=1)
cmlsspline = cml_spline(timeincxo.jd)
lt_spline = scipy.interpolate.UnivariateSpline(eph['datetime_jd'], eph['lighttime'], k=1)
ltspline = lt_spline(timeincxo.jd)

# this block takes some CML values and calculates their associated times
# so they can be plotted onto the top axis of the lightcurve
chosencmls = [60, 180, 300]
cmlrangetimes = []
cmlrangedatetimes = []
cmlsoltimesplus1 = []
givencmls = []
for cml in chosencmls:
    # define a spline for each chosen CML from which we can solve for the times at which they occur
    cml_splinesolver = scipy.interpolate.UnivariateSpline(eph['datetime_jd'], eph['PDObsLon']-cml, k=3)
    cml_splinesolverplus1 = scipy.interpolate.UnivariateSpline(eph['datetime_jd'], eph['PDObsLon']-(cml+1), k=3)
    # derive the roots, however there will be one found because of decrease from 360 to 0 (see plot)
    # we don't want that one, but know that a legitimate CML time will always increase
    # so we can choose to only include those that are followed by a CML/time that is larger
    solutiontimes = cml_splinesolver.roots()
    solutiontimesplus1 = cml_splinesolverplus1.roots()
    for i in range(0, len(solutiontimesplus1)):
        # we choose to store a JD value and a datetime value for convenience      
        if solutiontimesplus1[i] > solutiontimes[i]:
            cmlrangetimes.append(solutiontimes[i])
            givencmls.append(cml)
            # cmlrangedatetimes.append([Time(solutiontimes[i],format='jd').datetime, cml])
            cmlrangedatetimes.append([Time(Time(solutiontimes[i],format='jd').unix - lighttravel*60, format='unix').datetime, cml])

CMLxlines = []
for i in solutiontimes:
    CMLxlines.append(Time(i, format='jd').datetime)

cmlrangedates = (Time(Time(cmlrangetimes, format='jd').unix - lighttravel*60, format='unix').datetime)

# plotting figures as panels of one figure rather than 3 individual plots
timesfromobsstart = []
for t in times:
    newt = (t-times[0])
    timesfromobsstart.append(newt+1)

# 'Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 'Reverse biorthogonal',
# 'Discrete Meyer (FIR Approximation)', 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet',
# 'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets'

# 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus','mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
# prefer shan

def plot_xrayctwavelet(time, signal1, scales, datetime, cbarscale, lc_bounds,
                 waveletname='shan',
                 cmap=plt.cm.seismic,
                 ylabel='Period (min)'):

    # making figure
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_axes([0.1, 1.4, 1.2, 0.4])
    ax2 = fig.add_axes([0.1, 0.7, 1.2, 0.7])

    ax1.get_xaxis().set_visible(False)

    # plot the lightcurve
    # ax1.plot(datetime, signal1, label='Northern Aurora')
    ax1.plot(datetime, signal1, color='k', lw=1.0)
    ax1.set_xlim([datetime[0], datetime[-1]])
    ax1.set_ylim(-0.5, 12.5)

    jov_rot = (TimeDelta((595.5) * u.min)).datetime
    jr1 = (Time(datetime[0]) + jov_rot).datetime
    jr2 = (Time(jr1) + jov_rot).datetime
    jr3 = (Time(jr2) + jov_rot).datetime
    jr4 = (Time(jr3) + jov_rot).datetime

    jr1_text = (Time(datetime[0]) + jov_rot/2).datetime
    jr2_text = (Time(jr1) + jov_rot/2).datetime
    jr3_text = (Time(jr2) + jov_rot/2).datetime
    jr4_text = (Time(jr3) + jov_rot/2).datetime
    jr_text = [jr1_text, jr2_text, jr3_text, jr4_text]

    # text_col = 'lightslategrey'
    text_col = 'darkviolet'
    fill_col = 'darkgray'
    fs_text = 12

    jr_y = [10.5, 11.5, 10.5, 11.5, 10.5]
    jrs = [datetime[0], jr1, jr2, jr3, jr4]

    for i in range(4):
        ax1.plot([jrs[i], jrs[i+1]], [jr_y[i], jr_y[i]], lw=1.5, color=text_col)
        if jr_y[i] == 10.5:
            ax1.text(jr_text[i], jr_y[i]+1. , f'JR{i+1}', horizontalalignment='center', verticalalignment='center', color=text_col, weight='bold', fontsize=fs_text)
        else:
            ax1.text(jr_text[i], jr_y[i]-1.0 , f'JR{i+1}', horizontalalignment='center', verticalalignment='center', color=text_col, weight='bold', fontsize=fs_text)
        # ax1.axvline(jrs[i], ymin=jr_y[i]-0.5, ymax=jr_y[i]+0.5, lw=1.5, color='darkviolet')
    [ax1.vlines(jrs[i], ymin=jr_y[i]-0.25, ymax=jr_y[i]+0.25, lw=1.5, color=text_col) for i in range(len(jrs))]
    [ax1.vlines(jrs[i], ymin=jr_y[i+1]-0.25, ymax=jr_y[i+1]+0.25, lw=1.5, color=text_col) for i in range(1,4)]



    ax1.set_ylabel('Northern X-ray Counts')
    # ax1.set_title('X-ray Lightcurve')
    ax1.set_xlabel(str(datetime[0])+' to '+str(datetime[-1]), fontsize=18)
    for i in range(4):
        text_x = Time((Time(cmlrangedates[i]).unix + Time(cmlrangedates[i+4]).unix)/2, format='unix').datetime
        ax1.text(text_x, ax1.get_ylim()[1]+0.7, f'NE{i+1}', horizontalalignment='center', verticalalignment='center', color=fill_col, weight='bold', fontsize=fs_text)

    ax1a = ax1.twiny()
    ax1a.set_xlim(ax1.get_xlim())
    ax1a.set_xticks(cmlrangedates)
    ax1a.set_xticklabels(givencmls)
    # ax1a.text(cmlrangedates[4], 20, 'NE1', color='orange')
    ax1a.set_xlabel("CML (deg)")
    # ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # ax1.legend()

    # wavelet for North:

    # define time steps  
    dt = time[1] - time[0]
    # calculate coefficients and frequencies
    [coefficients, frequencies] = pywt.cwt(signal1, scales, waveletname, dt)
    # calculate power from coefficients and period from frequencies
    power = (abs(coefficients)) ** 2
    # produce period in minutes from frequency in Hz
    period = (1 / frequencies)/60

    # significances with contour levels set levels = [0.03125,0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,16]
    levels = cbarscale
    contourlevels = np.log2(levels)

    # make wavelet for North plot:
    im = ax2.contourf(datetime, np.log(period), np.log(power), contourlevels, extend='both', cmap=cmap, zorder=0)

    ax2.set_ylabel('North '+ylabel)
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax2.set_yticks(np.log(yticks))
    ax2.set_yticklabels(yticks)
    ax2.invert_yaxis()
    ylim = ax2.get_ylim()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel(str(datetime[0].replace(second=0, microsecond=0))+' to '+str(datetime[-1].replace(second=0, microsecond=0)))

    cbar_ax = fig.add_axes([1.35, 0.75, 0.03, 1])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label='PSD')

    # adding vertical lines to define boundaries
    power_T = power.T
    log_power_T = np.log(power_T)
    max_log_power = [max(log_power_T[i]) for i in range(len(datetime))]

    time_bins = np.array(time)
    time_inv = time_bins[np.where(np.array(max_log_power) > 2.5)[0]]
    datetime_res = datetime[np.where(np.array(max_log_power) > 2.5)[0]]
    chandra_times = [Time(datetime[i]).cxcsec for i in range(len(datetime_res))]

    time_diff = time_inv[1:] - time_inv[:-1]
    time_diff_min = time_diff/60
    ints = np.where(time_diff_min > 8)[0]

    datetime_ints_st = np.sort([datetime_res[i + 1] for i in ints])
    datetime_ints_end = np.sort([datetime_res[i] for i in ints])

    qpp_st = np.append(datetime_ints_st, datetime_res[0])
    qpp_start_ch = np.sort([Time(x).cxcsec for x in qpp_st])

    qpp_end = np.append(datetime_ints_end, datetime_res[-1])
    qpp_end_ch = np.sort([Time(x).cxcsec for x in qpp_end])

    # ax1a.axvspan(65, 280, alpha=0.25, color='orange', zorder=5)
    for i in range(len(lc_bounds)):
        ax1.axvline(lc_bounds[i], 0, 10, color=fill_col, lw=2, zorder=10)
        if (i%2 == 0):
            ax1.axvspan(lc_bounds[i], lc_bounds[i+1], alpha=0.25, color=fill_col, zorder=5)

    # ax1.axvline(datetime_res[0], 0, 10, color='magenta', lw=2, zorder=10)
    # ax1.axvline(datetime_res[-1], 0, 10, color='limegreen', lw=2, zorder=10)
    # ax2.axvline(datetime_res[0], 0, 1, color='magenta', lw=2, zorder=100)
    # ax2.axvline(datetime_res[-1], 0, 1, color='limegreen', lw=2, zorder=10)
    # ax2.axvline(datetime_ints_st[0], 0, 1, color='magenta', lw=2, zorder=100)
    # ax2.axvline(datetime_ints_end[0], 0, 1, color='limegreen', lw=2, zorder=10)

    # for i in range(len(datetime_ints_st) - 1):
    for i in range(len(qpp_start_ch)):
        if (qpp_end_ch[i] - qpp_start_ch[i]) > 30 * 60:
            # ax1.axvline(datetime_ints_st[i+1], color='limegreen', lw=2, zorder=10)
            # ax1.axvline(datetime_ints_end[i], color='limegreen', lw=2, zorder=10)
            if i==0:
                ax2.axvline(Time(qpp_start_ch[i], format='cxcsec').datetime, 0, 1, color='yellow', lw=2, zorder=10)
                ax2.axvline(Time(qpp_end_ch[i], format='cxcsec').datetime, 0, 1, color='yellow', lw=2, zorder=10)
            else:
                ax2.axvline(Time(qpp_start_ch[i], format='cxcsec').datetime, 0, 1, color='limegreen', lw=2, zorder=10)
                ax2.axvline(Time(qpp_end_ch[i], format='cxcsec').datetime, 0, 1, color='limegreen', lw=2, zorder=10)


    return power, period, coefficients, datetime_res, ints, datetime_ints_st, datetime_ints_end, time_diff_min


os.chdir(outdir)
# scales controls the period resolution and range, smaller steps changes the resolution and a larger range changes the range of frequencies explored
scales = np.arange(1, 64/(binsizeinseconds/60), 1)

# Normalised counts
norm_ncounts = (ncounts - np.mean(ncounts))/np.std(ncounts)
norm_scounts = (scounts - np.mean(scounts))/np.std(scounts)


cbarscale = [0.5, 1, 2, 4, 8, 16, 32, 64]
# wavelet plotting function applied with arguments of: numbered time bins, counts per bin for North, scales (frequency resolution and range), timestamps for the x-axis of lightcurve, titles for each plot,
# power, period, coeffs, datetime_res, ints, datetime_ints_st, datetime_ints_end = plot_xrayctwavelet(timesfromobsstart, ncounts, scales, timeindatetime, cbarscale)
power, period, coeffs, datetime_res, ints, datetime_ints_st, datetime_ints_end, time_diff_min = plot_xrayctwavelet(timesfromobsstart, ncounts, scales, tminusLTindatetime, cbarscale, bounds_NE_minusLT)

qpp_st = np.append(datetime_ints_st, datetime_res[0])
qpp_start_ch = np.sort([Time(x).cxcsec for x in qpp_st])

qpp_end = np.append(datetime_ints_end, datetime_res[-1])
qpp_end_ch = np.sort([Time(x).cxcsec for x in qpp_end])

np.savetxt(outdir + f'/{obsID}_qpp_times_threshold_2p5_lt.txt', np.c_[qpp_start_ch, qpp_end_ch], delimiter=',', header='QPP start time (cxcsec),QPP end time (cxcsec)', fmt='%s')

# plot_xrayctwavelet(timesfromobsstart, norm_ncounts, norm_scounts, scales, tminusLTindatetime, 'North Aurora Wavelet Power Spectrum','South Aurora Wavelet Power Spectrum', cbarscale)

# plt.savefig(outdir + f'/{obsID}_lc_wl_edit_and_vline_3_thr_w_NE_bounds.png', dpi=500, bbox_inches='tight')
# plt.savefig(outdir + f'/{obsID}_lc_wl_edit_and_vline_2p5_thr_w_NE_bounds.png', dpi=500, bbox_inches='tight')
# plt.savefig(outdir + f'/{obsID}_fig2_smaller_lt.png', dpi=500, bbox_inches='tight')
plt.savefig(outdir + f'/{obsID}_si_lt_vline_yellow.png', dpi=500, bbox_inches='tight')
plt.show()




