import numpy as np
import datetime
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd
import re
import matplotlib.dates as mdates
from astropy.time import Time
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
obsdate = "September 2021"
# Choose a bin size
binsizeinseconds = [30, 90, 300, 600]
# binsizeinseconds = [120, 60]
# binsizeinseconds = [60, 120]

# Pull start and end times from first photon in list
obs_start = min(alldata['t(s)'])
obs_end = max(alldata['t(s)'])
tstart = Time(obs_start, format='cxcsec')
tstop = Time(obs_end, format='cxcsec')

ndata = northdata['t(s)'].values
sdata = southdata['t(s)'].values

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

# Converting times from Chandra/XMM time to yr,m,d,h,m

# making figure
fig = plt.figure(figsize=(12,6))

count = 0
for x in binsizeinseconds:
    # create lists of binned data points
    ncounts = []
    scounts = []
    times = []
    for t in range(int(obs_start), int(obs_end), x):
        ntemptimes = ndata[(t < ndata) & (ndata < (t+x))]
        stemptimes = sdata[(t < sdata) & (sdata < (t+x))]
        times.append((t+t+x)/2)
        ncounts.append(len(ntemptimes))
        scounts.append(len(stemptimes))

    ncountserrors = np.sqrt(np.array(ncounts))
    scountserrors = np.sqrt(np.array(scounts))

    np_times = np.array(times)
    timeincxo = Time(np_times, format='cxcsec')
    timeindatetime = timeincxo.datetime
    np_TminusLT = np.array(times) - (lighttravel*60)
    tminusLTincxo = Time(np_TminusLT, format='cxcsec')
    tminusLTindatetime = tminusLTincxo.datetime

    # Define a function for CML throughout the observation using a spline (can equally interpolate etc.)
    '''cml_spline = scipy.interpolate.UnivariateSpline(eph['datetime_jd'], eph['PDObsLon'], k=1)
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
                cmlrangedatetimes.append([Time(solutiontimes[i],format='jd').datetime, cml])

    CMLxlines = []
    for i in solutiontimes:
        CMLxlines.append(Time(i, format='jd').datetime)

    cmlrangedates = (Time(cmlrangetimes, format='jd')).datetime'''

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

    def plot_xrayctwavelet(time, signal1, scales, datetime, cbarscale, wavelet, waveletname, label,
                     cmap=plt.cm.seismic,
                     ylabel='Period (min)',
                     xlabel=obsdate):


        ax = fig.add_subplot(int(len(binsizeinseconds)/2), 2, count+1)
        divider = make_axes_locatable(ax)
        # define time steps  
        dt = time[1] - time[0]
        # calculate coefficients and frequencies
        [coefficients, frequencies] = pywt.cwt(signal1, scales, wavelet, dt)
        # calculate power from coefficients and period from frequencies
        power = (abs(coefficients)) ** 2
        # produce period in minutes from frequency in Hz
        period = (1 / frequencies)/60

        # significances with contour levels set levels = [0.03125,0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,16]
        levels = cbarscale
        contourlevels = np.log2(levels)

        # make wavelet for North plot:
        im = ax.contourf(datetime, np.log(period), np.log(power), contourlevels, extend='both', cmap=cmap, zorder=0)

        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)
        # yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        yticks = 2**np.arange(0, 9)
        ax.set_yticks(np.log(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        print(min(np.log(period)), max(np.log(period)))
        # min_shan = 0.948039430188735
        # max_shan = 4.38202663467388
        min_all = -0.4382549309311552
        max_all = 4.40593215552743
        ax.set_ylim(max_all, min_all)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if (count==len(binsizeinseconds)-2) or (count==len(binsizeinseconds)-1):
            ax.set_xlabel(str(datetime[0].replace(second=0, microsecond=0))+' to '+str(datetime[-1].replace(second=0, microsecond=0)))
         # else:
            # ax.set_xticklabels([])
        cax = divider.append_axes("right", size="3%", pad=0.5)
        if count%2 != 0:
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('PSD', rotation=270, labelpad=5)
        else:
            cax.axis('off')
            ax.set_ylabel(ylabel)
        # cbar_ax = fig.add_axes([0.915, 0.2, 0.02, 0.6])
        # fig.colorbar(im, cax=cbar_ax, orientation="vertical", label='PSD')
        ax.set_title(r'$\bf{%s}$' % label[count] + f' Northern Aurora Wavelet Power Spectrum {x}s time bins')
        # ax.set_title(r'$\bf{%s}$' % label[count] + f' Wavelet Power Spectrum {x}s time bins')


    os.chdir(outdir)
    # scales controls the period resolution and range, smaller steps changes the resolution and a larger range changes the range of frequencies explored
    scales = np.arange(1, 64/(x/60), 1)
    # scales = np.arange(1, 64/(30/60), 1)

    # Normalised counts
    norm_ncounts = (ncounts - np.mean(ncounts))/np.std(ncounts)
    norm_scounts = (scounts - np.mean(scounts))/np.std(scounts)


    cbarscale = [0.5, 1, 2, 4, 8, 16, 32, 64]
    # wavelet plotting function applied with arguments of: numbered time bins, counts per bin for North, scales (frequency resolution and range), timestamps for the x-axis of lightcurve, titles for each plot,
    '''power, period, coeffs, datetime_res, ints, datetime_ints_st, datetime_ints_end = plot_xrayctwavelet(timesfromobsstart, ncounts, scales, timeindatetime, cbarscale)

    datetime_ints_st.append(datetime_res[0])
    qpp_start_ch = np.sort([Time(x).cxcsec for x in datetime_ints_st])

    datetime_ints_end.append(datetime_res[-1])
    qpp_end_ch = np.sort([Time(x).cxcsec for x in datetime_ints_end])

    np.savetxt(outdir + f'/{obsID}_qpp_times_threshold_3.txt', np.c_[qpp_start_ch, qpp_end_ch], delimiter=',', header='QPP start time (cxcsec),QPP end time (cxcsec)', fmt='%s')'''

    # plot_xrayctwavelet(timesfromobsstart, norm_ncounts, norm_scounts, scales, tminusLTindatetime, 'North Aurora Wavelet Power Spectrum','South Aurora Wavelet Power Spectrum', cbarscale)

    # 'Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal', 'Reverse biorthogonal',
    # 'Discrete Meyer (FIR Approximation)', 'Gaussian', 'Mexican hat wavelet', 'Morlet wavelet',
    # 'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets'

    # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus','mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
    # prefer shan
    # wavelets = ['gaus1', 'cgau1', 'morl', 'cmor', 'fbsp', 'mexh']
    wavelets = 'shan'
    waveltnames = ['Gaussian', 'Complex Gaussian', 'Morlet', 'Complex Morlet', 'Frequency B-Spline', 'Mexican Hat']
    labels = ['(a)', '(b)', '(c)', '(d)']

    wavelets_ex = ['gaus1', 'cgau1']
    waveltnames_ex = ['Gaussian', 'Complex Gaussian']
    plot_xrayctwavelet(timesfromobsstart, ncounts, scales, tminusLTindatetime, cbarscale, wavelets, waveltnames, labels)
    fig.tight_layout()
    count += 1

plt.savefig(outdir + f'/{obsID}_shan_diff_time_bins_lt.png', dpi=500, bbox_inches='tight')
plt.show()





