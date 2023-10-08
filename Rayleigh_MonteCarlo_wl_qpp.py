#Purpose: Read in time series. Compute the Rayleigh test. Evaluate significance from Monte Carlo simulation.
#Category: Time series analysis
#Authors: Christian Knigge, Caitriona Jackman (c.jackman@soton.ac.uk)
#Version released September 18th 2018, University of Southampton

import numpy as np
import pandas as pd
import label_maker as make_me_labels
from astropy.io import ascii
from astropy.io import fits as pyfits
from astropy.time import Time
from matplotlib import pyplot as plt
from numpy.random import poisson 
from matplotlib import gridspec
from math import ceil
import os
import time
from packaging.version import Version
import scipy
from scipy import interpolate
from scipy.interpolate import UnivariateSpline as spline

try:
    import numba
except ImportError:
    use_numba = False
    print()
    print('Numba is not installed; falling back on standard python')
else:
    if Version(numba.__version__) > Version('0.38'):
        use_numba = True
        print()
        print('Numba is installed; using jit to speed up code')
        @numba.jit(nopython=True)
        def rayleigh_power(omega, tf):
            s2 = 0.0*omega
            c2 = 0.0*omega
            pow = 0.0*omega
            nfreq = omega.size
            for i in range(nfreq):
                s2[i] = np.sum(np.sin(omega[i]*tf))**2.0
                c2[i] = np.sum(np.cos(omega[i]*tf))**2.0
                pow = (s2+c2)/float(tf.size)
            return pow
    else:
        use_numba = False
        print()
        print('Numba is installed, but version is old; falling back to standard python')

#start of main program    
start_clock = time.perf_counter()

# Lists to make output file later
bestperiod_list = []
maxpower_list = []
pvalue_list = []

#Read in unbinned, time-tagged data
# obsIDs = ['23370', '23371', '23372', '23374', '23375']
# obsIDs = ['23370', '23371']
obsIDs = ['23369']
# thr ='2p5'
thr ='3'
# constraint = 'PSD > 2.5'
constraint = 'PSD > 3.0'

start_times_2p5 = ['2021-09-15 19:10', '2021-09-15 20:54', '2021-09-15 22:10', '2021-09-16 15:22', '2021-09-17 00:40']
end_times_2p5 = ['2021-09-15 20:38', '2021-09-15 21:52', '2021-09-15 23:18', '2021-09-16 16:58', '2021-09-17 03:32']
dur_2p5 = ['88', '58', '68', '96', '172']
cts_2p5 = ['124', '79', '71', '118', '290']

start_times_3 = ['2021-09-15 19:18', '2021-09-17 00:48']
end_times_3 = ['2021-09-15 20:32', '2021-09-17 03:22']
dur_3 = ['74', '154']
cts_3 = ['101', '276']

qpp = 2

# Accounting for different filepaths of ObsIDs that originlly had SAMP values and others that did not.
for obsID in obsIDs:
    # Read in fits file from correct path
    folder_path = '/Users/mcentees/Desktop/Chandra/' + str(obsID) + '/primary'
    hdulist = pyfits.open(f'/Users/mcentees/Desktop/Will_code/hrcf{obsID}_pytest_evt2_change.fits', dtype=float)
    img_head = hdulist[1].header
    img_events = hdulist['EVENTS'].data
    date = img_head['DATE-OBS'] # Start date of observation
    date_end = img_head['DATE-END'] # End date of observation
    tstart_cxo = img_head['TSTART']
    tstop_cxo = img_head['TSTOP']
    hdulist.close()
    tstop_min = (tstop_cxo - tstart_cxo)/60

    plotit = True      #create an interactive plot of the power spectrum?
    verify = False      #if True, use a second method to calculate all power spectra and test for discrepancies
                        #this is a debugging tool and is slow, obviously...
    ntest = 10000       #set the number of runs for Monte Carlo testing
    startp = 2.0      #smallest period to search for (using Rayleigh test)
    endp = 80.0      #largest period to search for (using Rayleigh test)
    nstep = 1500       #number of frequency bins to search over (using Rayleigh test)
    time_convert=True   #if you want to convert the input time from seconds into minutes
    time_restrict=False#if you want to restrict the time to a manual selection (e.g. X-ray hot spot)
    tstart = 0.0      #the start time for N Hot spot 1 for ObsID 2519 from Jackman paper

    all = ascii.read('/Users/mcentees/Desktop/paper2' + f"/{obsID}_photonlist_PI_filter_qpp{qpp}_threshold_{thr}_North_lt.txt", header_start=0, data_start=0) # updated from Christian Knigge email, previously was data_start=1

    t_all = np.array(all['t(s)'])
    lat = np.array(all['lat (deg)'])
    lon = np.array(all['SIII_lon (deg)'])

    tend = float(ceil((t_all[-1] - t_all[0])/60))

    method = 2          #Method 1 has MC simulations start with Poissonian distribution of points.
                        #Method 2 has number of points for MC sim (nfake) is just same as input t

    # t = t_all[np.where((lat >= 60) & (lat <= 66) & (lon >= 162) & (lon <= 171))[0]]

    # t = t - t_all[0]    #Gets time from start of array (in seconds). *This assumes your input array is increasing in time.
    t = t_all - t_all[0]    #Gets time from start of array (in seconds). *This assumes your input array is increasing in time.
    if time_convert:
        t = t / 60.0    #converts the above time into minutes
    if time_restrict:
        t=t[(t>=tstart)&(t<=tend)]  #restricts to a specific hot spot start and end time

    #Run Rayleigh test on input data to extract highest power and best period

    startf = 1./endp
    endf = 1./startp
    f = np.linspace(startf, endf, nstep)    #equal linearly spaced frequency grid
    #Note: Be very careful when selecting the number of steps and the range of the frequency grid: Avoid undersampling, and ensure each spectral peak is fully resolved with at least 2 points
    p = 1./f #convert to period...
    omega = 2.0*np.pi*f #...and angular frequency

    if (use_numba):
        pow = rayleigh_power(omega, t)
    else:
        # s2 = np.asarray([(np.sum(np.sin(omega_i*t)))**2.0 for omega_i in omega],float,omega.size)
        # c2 = np.asarray([(np.sum(np.cos(omega_i*t)))**2.0 for omega_i in omega],float,omega.size)
        s2 = np.asarray([(np.sum(np.sin(omega_i*t)))**2.0 for omega_i in omega],float)
        c2 = np.asarray([(np.sum(np.cos(omega_i*t)))**2.0 for omega_i in omega],float)
        pow = (s2+c2)/float(t.size)


    if (verify):
        #use alternative method to check (this is slow, obviously)
        #
        pow2 = np.asarray(map(lambda omega_i: (np.sum(np.cos(omega_i*t))**2.0 + \
                                               np.sum(np.sin(omega_i*t))**2.0) / float(t.size), omega))
        test = pow-pow2
        if (np.abs(test).max() > 1.e-10):
            print(f'Max power difference between methods is in excess of 1e-10:  {np.abs(test).max()}')

    fbest = f[np.argmax(pow)]
    bestperiod_data = p[np.argmax(pow)]
    maxpower_data =pow.max()

    print()
    print(f'Period corresponding to highest power = {bestperiod_data} min')
    bestperiod_list.append(bestperiod_data)
    print(f'Max Power =  {maxpower_data}')
    maxpower_list.append(maxpower_data)
    print()

    #now we run Monte Carlo simulations to estimate significance
    dt = tend-tstart #this is the interval of time over which we randomly spread the photons for MC simulation

    maxpow_fake = np.zeros(ntest)
    pbest_fake = np.zeros(ntest)

    for ii in np.arange(ntest):

        if (method == 1):   #number of events drawn from Poisson distribution
            lam = float(t.size)
            nfake = poisson(lam, size=ntest)
            #each of the fake time arrays for the MC sim has a slightly different number of entries
        if (method == 2):  #fixed number of events
            nfake = np.full(ntest, t.size, dtype=int)
            #each of the fake time arrays for the MC sim has the exact same number of entries

        if (ii%1000 == 0) :
            print(f'Completed {ii} of {ntest} iterations')

        tf = tstart + np.sort(dt * np.random.uniform(low=0.0, high=1.0, size=nfake[ii]))
        #create the mock time series:
        #   nfake randomly (uniformly) distributed events across the same time interval as the actual data

        if (use_numba):
            powf = rayleigh_power(omega, tf)
        else:
            s2 = np.asarray([ (np.sum(np.sin(omega_i*tf)))**2.0 for omega_i in omega],float)
            c2 = np.asarray([ (np.sum(np.cos(omega_i*tf)))**2.0 for omega_i in omega],float)
            powf = (s2+c2)/float(np.size(tf))

        if (verify):
            #use alternative method to check (this is slow, obviously)
            #
            powf2 = np.asarray(map(lambda omega_i: (np.sum(np.cos(omega_i*tf))**2.0 + \
                                                   np.sum(np.sin(omega_i*tf))**2.0) / float(tf.size), omega))
            test = powf-powf2
            if (np.abs(test).max() > 1.e-10):
                print(f'Max power difference between methods is in excess of 1e-10:  {np.abs(test).max()}')

        #save the highest peak and the associated period from each iteration
        maxpow_fake[ii] = powf.max()
        pbest_fake[ii]  = p[np.argmax(powf)]

    fake_succ = 0
    for ii in range(ntest):
        if (maxpow_fake[ii] > maxpower_data):
            fake_succ = fake_succ + 1

    pvalue = float(fake_succ)/float(ntest)

    print(f'Number of Fake Data sets with Power > Real Power =  {fake_succ}')
    print (f'P-value (Fake Power >= Real Power) = {pvalue}')
    pvalue_list.append(pvalue)
    print()
    print()

    stop_clock = time.perf_counter()
    timer = stop_clock - start_clock
    print(f'Code Execution Time:  {np.round(timer/60., 2)}  min')

    if (plotit):
        figsize=(10,10)
        cols = 2
        rows = 2
        gs = gridspec.GridSpec(rows,cols,wspace=0.5, hspace=0.5)
        ax = np.empty((rows,cols), dtype=object)
        full_fig, ax=plt.subplots(rows,cols,figsize=figsize)

        bins = np.linspace(t[0], t[-1],ceil(t[-1] - t[0]))
        counts = np.histogram(t, bins)[0]
        dummy_poisson = np.random.poisson(lam=np.nanmean(maxpow_fake), size=ntest)

        s2 = np.asarray([(np.sum(np.sin(omega_i*t)))**2.0 for omega_i in omega],float)
        c2 = np.asarray([(np.sum(np.cos(omega_i*t)))**2.0 for omega_i in omega],float)
        ray_test = (s2+c2)/float(t.size)
        # ray_test = rayleigh_power(omega,t)
        binwidth=1

        mc_bins=np.arange(min(dummy_poisson), max(dummy_poisson) + binwidth, binwidth)
        values, base = np.histogram(dummy_poisson, bins=mc_bins)
        cumlative = np.cumsum(values)
        cumlative_percentage = (cumlative/ntest)*100
        try:
            xnew = np.linspace(mc_bins[:-1].min(), mc_bins[:-1].max(),3000)
        except ValueError:
            pass
        #cumlative_smooth = spline(mc_bins[:-1], cumlative_percentage, xnew)
        smooth_func = spline(mc_bins[:-1], cumlative_percentage)# xnew)
        smooth_func.set_smoothing_factor(0.01)
        cumlative_smooth = smooth_func(xnew)

        ax[0,0].hist(t, bins, color='black', linewidth=0.5)
        ax[0,0].set_xlabel('Time (min)', size = 14)
        ax[0,0].set_ylabel('Counts/min', size = 14)
        #ax[0,0].set_title('%s'%obs_id[indx])

        ax[0,1].plot(p,pow,'-k', lw=1)
        ax[0,1].set_ylabel('Rayleigh Power', size = 14)
        ax[0,1].set_ylim(0,30)
        ax[0,1].set_xlim(0)
        #ax2.axvline(x = p[pow == max(pow)],ymin= max(pow)/20, ymax=20, color='blue', linestyle='dashed', linewidth=3.0)
        ax[0,1].axvline(x = p[pow == max(pow)], color='blue', linestyle='dashed', linewidth=2.0)
        ax[0,1].axhline(y=np.percentile(dummy_poisson,99), color='black', linestyle='--')
        ax[0,1].set_xlabel('Period (min)', size = 14)
        #           ax[0,1].text(-15,21.23,'b',size=20)

        ax4 = ax[1,1].twinx()
        ax[1,1].hist(dummy_poisson, bins=np.arange(min(dummy_poisson), max(dummy_poisson) + binwidth, binwidth),histtype = 'step',\
                     fill = None, edgecolor='black', density=False)
        ax[1,1].axvline(max(ray_test), color='blue', linestyle='dashed', linewidth = 2)
        ax[1,1].set_xlabel('Max. Power', size = 14)
        ax[1,1].set_ylabel('# of events', size = 14)
        #ax[1,0].text(-2.5,1900,'c',size=20)

        ax4.plot(xnew,cumlative_smooth, color='red')
        #ax2.plot(xnew, cumlative_smooth, color='red')
        ax4.set_ylim(0,100)
        ax4.set_xlim(0,30)
        ax4.set_ylabel('Cumulative %', size = 14, rotation=270,labelpad=10)

        pow_indx = np.where(ray_test == max(ray_test))[0]
        ax[1,0].loglog(f,pow,'-k',lw=1)
        # m, c = np.polyfit(np.log10(f), np.log10(pow), 1) # fit log(y) = m*log(x) + c
        # #y_fit = np.exp(m*np.log10(f) + c) # calculate the fitted values of y 
        # test = np.poly1d(m,c)
        # y_fit = test(np.log10(f))
        ax[1,0].axvline(max(f[pow_indx]), color='blue', linestyle='dashed', linewidth = 2.0)
        #plt.axhline(y=np.percentile(bins,99), color='black', linestyle='--')
        ax[1,0].axhline(y=np.percentile(dummy_poisson,99), color='black', linestyle='--')
        ax[1,0].set_ylabel('Rayleigh Power', size=14)
        ax[1,0].set_xlabel('Frequency (1/min)',size=14)
        #ax[1,1].title('North All', size=24)
        ax[1,0].grid(True,which="both",axis='x',ls="-",alpha=0.5)
        ax[1,0].grid(True,which="major",axis='y',ls="-",alpha=0.5)
        # ax[1,0].set_ylim(0,100)
        # ax[1,0].set_xlim(0,1)

        # full_fig.suptitle(f'ObsID: {obsID} - Northern viewing',size =20, y=0.95)
        if thr == '2p5':
            full_fig.suptitle(f'Constraint: {constraint}\n{start_times_2p5[qpp-1]} - {end_times_2p5[qpp-1]} ({dur_2p5[qpp-1]} min)',size =16, y=0.95, va='center')
        else:
            full_fig.suptitle(f'Constraint: {constraint}\n{start_times_3[qpp-1]} - {end_times_3[qpp-1]} ({dur_3[qpp-1]} min)',size =16, y=0.95, va='center')
        # full_fig.suptitle(f'ObsID: {obsID} - Southern envelope viewing',size =20, y=0.95)
        # plt.title(f'ObsID: {obsID}',size =20)
        make_me_labels.draw_labels(ax[ax != None])
        # plt.tight_layout()
        # full_fig.savefig(r'NHS_rayleigh\ahsnuc\full_RT_results\%i_step_logspace_full_RT_%s_NHS%i_AHSNuc_100000_v1.png' %(nstep,int(final_all_spat_id[indx]),i+1), bbox_inches='tight',dpi=200)
        full_fig.savefig('/Users/mcentees/Desktop/paper2/RT_results_lt/Output_plot' + f'/{obsID}_qpp{qpp}_{ntest}_runs_threshold_{thr}_North_title.png', dpi=500, bbox_inches='tight')
output = {'ObsID': obsIDs, 'Period Corresponding to Max Power': np.round(bestperiod_list, 3), 'Max Power': np.round(maxpower_list, 3), 'P-value': pvalue_list}

out_df = pd.DataFrame(data=output)
out_df.to_excel(f'/Users/mcentees/Desktop/paper2/RT_results_lt/Output_data/{obsID}_qpp{qpp}_{ntest}_runs_threshold_{thr}_North_title.xlsx')
plt.show()

