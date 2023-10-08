import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from lobe_field_strength import load_lobe_field_strength
from read_juno_ephemeris_from_amda import *
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates

from scipy.io import readsav
from h5py import File
from glob import glob
import time as t
import json
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.io import fits as pyfits
from astropy.io import ascii
from astroquery.jplhorizons import Horizons     # automatically download ephemeris 
from os import path
from tfcat import TFCat
# from datetime import datetime


def get_polygons(polygon_fp, start, end):
    unix_start = t.mktime(start.utctimetuple())
    unix_end = t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array = []
    emission_type = []
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
                time_points = np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
                if any(time_points <= unix_end) and any(time_points >= unix_start):
                    polygon_array.append(np.array(catalogue._data.features[i]['geometry']['coordinates'][0]))
                    emission_type.append(catalogue._data.features[i]['properties']['feature_type'])
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval         
    return polygon_array, emission_type

def juno_ephemeris_from_uiowa(file):
    data_file = ascii.read(file)
    scet_ut = data_file['SCET (UT)']
    doy = [scet_ut[i].split(' ')[1] for i in range(len(scet_ut))]
    date = [datetime.datetime.strptime(scet_ut[i], "%Y %j %H %M %S.%f") for i in range(len(scet_ut))]

    lon = data_file['W Lon (deg)']
    mlat = data_file['MLat (deg)']
    mlt = data_file['MLT (hrs)']
    r_j = data_file['R (Rj)']
    l_shell = data_file['L']
    io_phase = data_file['Io phase (deg)']

    return date, doy, lon, mlat, mlt, r_j, l_shell, io_phase

def ephemeris_labels(dtime, file):
    time_arr, doy_arr, lon_arr, mlat_arr, mlt_arr, r_arr, l_arr, io_arr = juno_ephemeris_from_uiowa(file)
    index = np.where(Time(time_arr).unix == Time(dtime).unix)[0][0]

    r_index = str("%.2f" % np.round(r_arr[index], 2))
    lon_index = str("%.2f" % np.round(lon_arr[index], 2))
    mlat_index = str("%.2f" % np.round(mlat_arr[index], 2))
    mlt_index = str("%.2f" % np.round(mlt_arr[index], 2))
    l_index = str("%.2f" % np.round(l_arr[index], 1))
    io_index = str("%.2f" % np.round(io_arr[index], 1))

    eph_strs = [str(x) for x in [r_index, lon_index, mlat_index, mlt_index, l_index, io_index]]

    return eph_strs

@mticker.FuncFormatter
def ephemeris_fmt(tick_val, _):
    """
    Call with eg

        ax.xaxis.set_major_formatter(plt.FuncFormatter(ephemeris_fmt))

    or, if decorator @matplotlib.ticker.FuncFormatter used

        ax.xaxis.set_major_formatter(ephemeris_fmt)

    """

    # Convert matplotlib datetime float to date

    tick_dt = mdates.num2date(tick_val)
    tick_dt = tick_dt.replace(tzinfo=None)
    tick_str = datetime.datetime.strftime(tick_dt, ('%Y-%m-%d'))
    # this returns corresponding radial dist, gse_lat, gse_lt for the tick
    # as strings in a list
    eph_str = ephemeris_labels(tick_dt, iowa_ephem)
    eph_str = [tick_str] + eph_str
    tick_str = '\n'.join(eph_str)

    return tick_str



iowa_ephem = '/Users/mcentees/Desktop/paper2/juno_ephemeris_plus_mag_data/iowa_ephem.txt'
date_ephem, doy_ephem, lon_ephem, mlat_ephem, mlt_ephem, r_ephem, l_ephem, io_ephem = juno_ephemeris_from_uiowa(iowa_ephem)

data_start = pd.Timestamp('2021-09-11')
data_end = pd.Timestamp('2021-09-19')
polygon_path = '/Users/mcentees/catalog_xray_08092021_18092021.json'

saved_polys, poly_labels = get_polygons(polygon_path, data_start, data_end)

# reading in fits data
hdulist = pyfits.open(f'/Users/mcentees/Desktop/Will_code/hrcf23369_pytest_evt2_change.fits', dtype=float)
img_head = hdulist[1].header
date = img_head['DATE-OBS']  # Start date of observation
date_end = img_head['DATE-END']  # Start date of observation
tstart = img_head['TSTART']
tstop = img_head['TSTOP']
hdulist.close()

alldata = pd.read_csv('/Users/mcentees/Desktop/Chandra/23369/primary/23369_photonlist_PI_filter_Jup_full_10_250.txt')
alldata.columns = alldata.columns.str.replace("# ", "")

data_start = min(alldata['t(s)'])
data_end = max(alldata['t(s)'])

# calculating light travel time
'''eph_tstart = Time(tstart, format='cxcsec')
dt = TimeDelta(0.125, format='jd')
tstop_time = Time(tstop, format='cxcsec')
eph_tstop = Time(tstop_time + dt)

obj = Horizons(id=599, location='500@-151', epochs={'start': eph_tstart.iso, 'stop': eph_tstop.iso, 'step': '1m'}, id_type='majorbody')
eph = obj.ephemerides()

# pull the mean light travel time
lighttravel = eph['lighttime'].mean()'''

lighttravel = 33.9115185511787

tstartminusLT = data_start - lighttravel * 60
tstart_minusLT_datetime = Time(tstartminusLT, format='cxcsec').datetime

tstopminusLT = data_end - lighttravel * 60
tstop_minusLT_datetime = Time(tstopminusLT, format='cxcsec').datetime

time_window = TimeDelta(0 * u.hour).datetime

# calculate lobe magnetic field plot from Kivelson & Khurana (2002) model
time_start = datetime.datetime(2021, 9, 11)
time_end = datetime.datetime(2021, 9, 19)
(date_plot, Blobe, Blobe_err_plus, Blobe_err_minus) = load_lobe_field_strength(time_start, time_end)

# reading in mag data
mag_data = ascii.read('/Users/mcentees/Desktop/paper2/juno_ephemeris_plus_mag_data/mag_data_processed_long_23369.txt')

time_mag = mag_data['SCET(UTC)']
mag_amp = mag_data['Magnitude(nT)']


# reading in JunoWaves data
juno_file = '/Users/mcentees/full_JunoWaves_spdyn_short_freq_range_processed_23369.sav'
# juno_file = '/Users/mcentees/Desktop/autoplot_files/juno_waves_23369_interp.sav'

file_sav = readsav(juno_file, python_dict=True)
# time = np.array(Time(file_sav['epoch'], format='unix').datetime)
time = np.array(file_sav['epoch'])
freq = np.array(file_sav['frequency'])
flux = np.array(file_sav['data'])

# rescaling data to have shorter time window of 8 days starting on 11-Sep-2021
'''time = time[np.where(time >= float(Time('2021-9-11').unix))[0]]
freq= freq[np.where(time >= float(Time('2021-9-11').unix))[0]]
flux = flux[np.where(time >= float(Time('2021-9-11').unix))[0]]'''

freq_cutoff = np.where(freq > 150)[0][0]
freq_min = np.where(freq > 20)[0][0]

# freq_range = freq[0:freq_cutoff + 1]
# flux_range = flux.T[0:freq_cutoff + 1]
freq_range = freq[freq_min - 1:freq_cutoff + 1]
flux_range = flux.T[freq_min - 1:freq_cutoff + 1]

freq_bins = len(freq_range) * 4

# rescaling freq array
freq_rescaled = 10 ** (numpy.linspace(start=numpy.log10(freq_range[0]), stop=numpy.log10(freq_range[-1]), num=freq_bins , dtype=float))

flux_new = numpy.zeros((len(time), len(freq_rescaled)), dtype=float)

# interpolate frequency
for i in range(len(time)):
    flux_new[i, :] = numpy.interp(x=freq_rescaled, xp=freq_range, fp=flux_range.T[i, :])


# Paramaters for colourbar
# clrmap = 'Spectral_r'
clrmap = 'viridis'

vmin = np.quantile(flux_range[flux_range > 0.], 0.00)
# vmax = np.quantile(flux_range[flux_range > 0.], 1.00)
# vmax = 1.e-14
vmax = 1.e-15

# Make figure
fs_labels = 10
fs_title = 12
fs_ticks = 10
bounds_st = '--'
bounds_c = 'gray'
bounds_lt = 2


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(211)
# im = ax.pcolormesh(time, freq_range, flux_range, cmap=clrmap, norm=LogNorm(vmin=vmin, vmax=vmax), zorder=0)
im = ax.pcolormesh(time, freq_rescaled, flux_new.T, cmap=clrmap, norm=LogNorm(vmin=vmin, vmax=vmax), zorder=0)
ax.axvline(Time(tstartminusLT, format='cxcsec').unix, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=15)
ax.axvline(Time(tstopminusLT, format='cxcsec').unix, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=15)
ax.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
ax.set_ylabel('Frequency (kHz)', fontsize=fs_labels)
ax.set_title('Juno Waves data - Electric Field Flux Density', fontsize=fs_title)
# ax.set_xlabel('Time', fontsize=fs_labels)
# ax.set_xlim(time[0], (Time(time[-1]) + 1*u.s).value)
ax.set_xlim(float(Time('2021-9-11').unix), time[-1] + 1)
# ax.set_ylim(freq[0], 140)
# ax.set_ylim(9, 140)
ax.set_ylim(20, 140)
# ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
# ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
for i, shape in enumerate(saved_polys):
    # label_x = np.mean(shape[:, 0])
    # label_y = np.mean(shape[:, 1])
    label_x = np.quantile(shape[:, 0], 0.5)
    label_y = np.quantile(shape[:, 1], 0.5)
    # if (label_x > ax.get_xlim()[0]) and (label_y > ax.get_ylim()[0]):
        # ax.text(label_x, label_y, poly_labels[i], fontsize=10, zorder=30, horizontalalignment='center', verticalalignment='center', color='tomato')
    # ax.add_patch(Polygon(shape, color='white', zorder=10, fill=False, alpha=0.8))
    # ax.add_patch(Polygon(shape, color='white', zorder=10, fill=True, alpha=0.5))
    ax.add_patch(Polygon(shape, color='white', zorder=10, fill=True))

# Formatting colourbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
# cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
cb = fig.colorbar(im, cax=cax)
cb.set_label('Estimated flux density', fontsize=fs_labels)
cb.ax.tick_params(labelsize=fs_ticks)
cb.ax.set_title(r'$\mathrm{W/m^2/Hz}$', fontsize=fs_labels)
ax.set_xticklabels([])


ax2 = fig.add_subplot(212)
ax2.plot(Time(time_mag).datetime, mag_amp, c='cornflowerblue', lw=0.5, zorder=10, label='Juno MAG')
ax2.plot(date_plot, Blobe, 'k', lw=0.5, zorder=0, label='K & K (2002) lobe B-field model')
ax2.plot(date_plot, Blobe_err_plus, 'k--', lw=0.5, zorder=0)
ax2.plot(date_plot, Blobe_err_minus, 'k--', lw=0.5, zorder=0)
# ax2.set_xlabel(f'2021-09-09 ({doy_ephem[0]}) through 2021-09-18 ({int(doy_ephem[-1]) - 1})')
# ax2.set_xlabel(f'2021-09-11 ({doy_ephem[0]}) through 2021-09-18 ({int(doy_ephem[-1]) - 1})')
entries_per_day = 24*60/5
ax2.set_xlabel(f'2021-09-11 ({doy_ephem[int(entries_per_day*2)]}) through 2021-09-18 ({int(doy_ephem[-1]) - 1})')
ax2.set_ylabel('Magnetic field amplitude (nT)', fontsize=fs_labels)
ax2.set_ylim(4, 12.9)
# ax2.set_xlim((Time(time_mag[0]) - 5 * u.minute).datetime, (Time(time_mag[-1]) + 5 * u.minute - 1 * u.day).datetime)
# ax2.set_title('Magnetic Field from FGM in Payload Coordinates (10 minute Averages)', fontsize=fs_title)
ax2.axvline(Time(tstartminusLT, format='cxcsec').datetime, 0.0, 130.0, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax2.axvline(Time(tstopminusLT, format='cxcsec').datetime, 0.0, 130.0, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax2.legend(loc='lower left', handlelength=1.0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.10)
cax2.axis('off')
ax2.set_xlim(Time('2021-09-11').datetime, Time('2021-09-19').datetime)

seconds_per_day = 60*60*24
# major_xticks_mag = Time(np.arange(time[0], Time('2021-09-19').unix + 16, step=seconds_per_day), format='unix').datetime
# minor_xticks_mag = Time(np.arange(time[0], Time('2021-09-19').unix + 16, step=seconds_per_day/4), format='unix').datetime
major_xticks_mag = Time(np.arange(Time('2021-09-12').unix, Time('2021-09-19').unix, step=seconds_per_day*2), format='unix').datetime
minor_xticks_mag = Time(np.arange(Time('2021-9-11').unix, Time('2021-09-19').unix + 16, step=seconds_per_day/2), format='unix').datetime

# major_xticks_waves = np.arange(time[0], Time('2021-09-19').unix + 16, step=seconds_per_day)
# minor_xticks_waves = np.arange(time[0], Time('2021-09-19').unix + 16, step=seconds_per_day/4)
major_xticks_waves = np.arange(Time('2021-09-12').unix, Time('2021-09-19').unix + 16, step=seconds_per_day)
minor_xticks_waves = np.arange(Time('2021-9-11').unix, Time('2021-09-19').unix + 16, step=seconds_per_day/2)

ax2.set_xticks(major_xticks_mag)
ax.set_xticks(major_xticks_waves)

ax2.set_xticks(minor_xticks_mag, minor=True)
ax.set_xticks(minor_xticks_waves, minor=True)

ax2.tick_params(axis='x', which='major', length=10, top=True, direction='in', bottom=False)
ax2.tick_params(axis='x', which='major', length=10, bottom=True, direction='out')
# ax.tick_params(axis='x', which='major', length=10, labelbottom=False)

ax2.tick_params(axis='x', which='minor', length=5, top=True, direction='in', bottom=False)
ax2.tick_params(axis='x', which='minor', length=5, bottom=True, direction='out')
# ax.tick_params(axis='x', which='minor', length=5, labelbottom=False)


ax2.xaxis.set_major_formatter(plt.FuncFormatter(ephemeris_fmt))

eph_titles = '\n'.join(['\n', r'$\mathrm{R_J}$', r'$\mathrm{Lon_{III}}$', r'$\mathrm{MLat_{JRM09}}$', 'MLT', 'L', 'Io Phase'])
kwargs = {'xycoords': 'figure fraction', 'fontsize': fs_ticks-0.5}
# kwargs['xy'] = (0.10, -0.056)
kwargs['xy'] = (0.05, 0.040)
# ax2.annotate(eph_titles, **kwargs)
ax2.annotate(eph_titles, **kwargs)

# ax2.set_xticklabels('ha=right')
fig.subplots_adjust(hspace=0)
# plt.autoscale()
# fig.tight_layout(pad=1.08, h_pad=.5, w_pad=0)

# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/JunoWaves_full_plus_khurana.png', format='png', dpi=500)
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/JunoWaves_full_plus_khurana_new_cbar_interp.png', format='png', dpi=500, bbox_inches='tight')
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/JunoWaves_full_plus_khurana_tick_labels.png', format='png', dpi=500, bbox_inches='tight')
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/JunoWaves_full_plus_khurana_tick_labels_title.png', format='png', dpi=500, bbox_inches='tight')
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/JunoWaves_full_plus_khurana_tick_labels_title.png', format='png', dpi=500, bbox_inches='tight')
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/fig4_will.png', format='png', dpi=500, bbox_inches='tight')
# plt.savefig('/Users/mcentees/Desktop/paper2/JunoWaves_timeseries_23369/fig4_fixed_lt.png', format='png', dpi=500, bbox_inches='tight')

plt.show()


