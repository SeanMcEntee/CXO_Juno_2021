import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates

from astropy.time import Time, TimeDelta
from astropy.io import fits as pyfits
import astropy.units as u
from astropy.io import ascii

hdulist = pyfits.open(f'/Users/mcentees/Desktop/Will_code/hrcf23369_pytest_evt2_change.fits', dtype=float)
img_head = hdulist[1].header
date = img_head['DATE-OBS'] # Start date of observation
date_end = img_head['DATE-END'] # Start date of observation
tstart = img_head['TSTART']
tstop = img_head['TSTOP']
hdulist.close()

# Find and read in North and South Photon lists
obsID = '23369'
indir = f"/Users/mcentees/Desktop/Chandra/{obsID}/primary"
alldata = pd.read_csv(indir + f'/{obsID}_photonlist_PI_filter_Jup_full_10_250.txt')
alldata.columns = alldata.columns.str.replace("# ", "")

data_start = min(alldata['t(s)'])
data_end = max(alldata['t(s)'])

lighttravel = 33.9115185511787

tstartminusLT = data_start - lighttravel * 60
tstopminusLT = data_end - lighttravel * 60

fs_labels = 10
fs_ticks = 10
bounds_st = '--'
bounds_c = 'orange'
bounds_lt = 1.5


# Adding tao model stuff
tao_data = ascii.read('/Users/mcentees/Desktop/paper2/output-jup_sw_v_jup_sw_pdyn_jup_sw_da_2021251000000000.txt')
date_tao = tao_data['col1']
sw_v_r = tao_data['col2']
sw_v_t = tao_data['col3']
sw_pdyn = tao_data['col4']
jse_angle = tao_data['col5']

jr = 595.5 * 60

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(311)

ax.plot(Time(date_tao, format='isot').datetime, sw_v_r, 'k', lw=0.5, label=r'$v_r$')
# ax.plot(Time(date_tao, format='isot').datetime, sw_v_t, 'b', lw=0.5, label=r'$v_t$')
ax.axvline(Time(tstartminusLT, format='cxcsec').datetime, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax.axvline(Time(tstopminusLT, format='cxcsec').datetime, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax.set_ylabel('Velocity (km/s)', fontsize=fs_labels) # labelpad=7)
ax.set_ylim(300.0, 400)
ax.set_xlim(Time(date_tao[47], format='isot').datetime, Time(date_tao[-1], format='isot').datetime)

ax.set_xticklabels([])
major_yticks_ax = np.arange(300, 401, 20)
minor_yticks_ax = np.arange(300, 401, 5)
ax.set_yticks(major_yticks_ax)
ax.set_yticks(minor_yticks_ax, minor=True)

# ax.legend(loc='center right', handlelength=1.0)


ax2 = fig.add_subplot(312)
ax2.plot(Time(date_tao, format='isot').datetime, sw_pdyn, 'k', lw=0.5)
ax2.axvline(Time(tstartminusLT, format='cxcsec').datetime, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax2.axvline(Time(tstopminusLT, format='cxcsec').datetime, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax2.set_ylabel('Dynamic pressure (nPa)', fontsize=fs_labels) # labelpad=7)
ax2.set_ylim(0.0, 0.5)
ax2.set_xlim(Time(date_tao[47], format='isot').datetime, Time(date_tao[-1], format='isot').datetime)
ax2.set_xticklabels([])

major_yticks_ax2 = np.arange(0, 0.51, 0.1)
minor_yticks_ax2 = np.arange(0, 0.51, 0.02)
ax2.set_yticks(major_yticks_ax2)
ax2.set_yticks(minor_yticks_ax2, minor=True)

ax3 = fig.add_subplot(313)
ax3.plot(Time(date_tao, format='isot').datetime, jse_angle, 'k', lw=0.5)
ax3.axvline(Time(tstartminusLT, format='cxcsec').datetime, ls=bounds_st, color=bounds_c, lw=bounds_lt, zorder=10)
ax3.axvline(Time(tstopminusLT, format='cxcsec').datetime, color=bounds_c, ls=bounds_st, lw=bounds_lt, zorder=10)
ax3.set_ylabel('Jup-Sun-Earth \nangle (deg)', fontsize=fs_labels) # labelpad=16)
ax3.set_ylim(-1, 9)
ax3.set_xlabel('Time (UT)')
ax3.set_xlim(Time(date_tao[47], format='isot').datetime, Time(date_tao[-1], format='isot').datetime)
major_yticks_ax3 = np.arange(0, 9, 2)
minor_yticks_ax3 = np.arange(-1, 9.1, 1)
ax3.set_yticks(major_yticks_ax3)
ax3.set_yticks(minor_yticks_ax3, minor=True)

# adding jr lines
cxo_start = Time(tstartminusLT, format='cxcsec').datetime
jov_rot = (TimeDelta((595.5) * u.min)).datetime
jr1 = (Time(cxo_start) + jov_rot).datetime
jr2 = (Time(jr1) + jov_rot).datetime
jr3 = (Time(jr2) + jov_rot).datetime
jr4 = (Time(jr3) + jov_rot).datetime

jr1_text = (Time(cxo_start) + jov_rot/2).datetime
jr2_text = (Time(jr1) + jov_rot/2).datetime
jr3_text = (Time(jr2) + jov_rot/2).datetime
jr4_text = (Time(jr3) + jov_rot/2).datetime
jr_text = [jr1_text, jr2_text, jr3_text, jr4_text]

# text_col = 'lightslategrey'
text_col = 'darkviolet'
fs_text = 12

jr_y_ax = [310, 320, 310, 320, 310]
jr_y_ax2 = [0.35, 0.4, 0.35, 0.4, 0.35]
jr_y_ax3 = [0, 1, 0, 1, 0]
jrs = [cxo_start, jr1, jr2, jr3, jr4]

for axis in [ax, ax2, ax3]:
    axis.axvline(jr1, lw=1.5, color='darkviolet', ls='dashed')
    axis.axvline(jr2, lw=1.5, color='darkviolet', ls='dashed')
    axis.axvline(jr3, lw=1.5, color='darkviolet', ls='dashed')

for i in range(len(jr_text)):
    ax.text(jr_text[i], ax.get_ylim()[1] + 5, f'JR{i+1}', ha='center', va='center', color='darkviolet')

seconds_per_day = 60*60*24
major_xticks = Time(np.arange(Time('2021-09-11').unix, Time('2021-09-19').unix + 60, step=seconds_per_day), format='unix').datetime
minor_xticks = Time(np.arange(Time('2021-09-11').unix, Time('2021-09-19').unix + 60, step=seconds_per_day/2), format='unix').datetime

ax3.set_xticks(major_xticks)
ax2.set_xticks(major_xticks)
ax.set_xticks(major_xticks)

ax3.set_xticks(minor_xticks, minor=True)
ax2.set_xticks(minor_xticks, minor=True)
ax.set_xticks(minor_xticks, minor=True)

ax3.tick_params(axis='both', which='major', length=10)
ax2.tick_params(axis='both', which='major', length=10)
ax.tick_params(axis='both', which='major', length=10)

ax3.tick_params(axis='both', which='minor', length=5)
ax2.tick_params(axis='both', which='minor', length=5)
ax.tick_params(axis='both', which='minor', length=5)

fig.tight_layout()
plt.savefig('/Users/mcentees/Desktop/paper2/Tao_lt.png', format='png', dpi=500)
plt.show()

