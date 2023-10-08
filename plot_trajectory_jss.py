import datetime
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.time import Time, TimeDelta
from astropy.io import fits as pyfits
from astropy.io import ascii

import pandas as pd
from pdyn_to_ms_boundaries import *
from read_juno_ephemeris_from_amda import *
from read_juno_ephemeris_from_webgeocalc import *
# from read_juno_ephemeris_from_uiowa import *

'''file_ephem = "/Users/mcentees/Desktop/ephemeris/juno_jup_xyz_jso_2016_2021.txt"
(date_ephem,x_coord,y_coord,z_coord) = juno_ephemeris_from_amda(file_ephem)

file_ephem_uiowa = "/Users/mcentees/Desktop/ephemeris/juno_jup_xyz_jso_iowa_2022.txt"
(date_ephem_uiowa,x_coord_uiowa,y_coord_uiowa,z_coord_uiowa) = juno_ephemeris_from_uiowa(file_ephem_uiowa)

date_ephem = numpy.append(date_ephem,date_ephem_uiowa)
x_coord = numpy.append(x_coord,x_coord_uiowa)
y_coord = numpy.append(y_coord,y_coord_uiowa)
z_coord = numpy.append(z_coord,z_coord_uiowa)'''

file_ephem = "/Users/mcentees/Desktop/paper2/jupiter_magnetosphere_boundaries-main/juno_jup_xyz_jss_2016_2022.txt"
(date_ephem, x_coord, y_coord, z_coord) = juno_ephemeris_from_webgeocalc(file_ephem, planetary_radius = 71492.)

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection="3d")

# draw sphere
u, v = numpy.mgrid[0:2*numpy.pi:50j, 0:numpy.pi:50j]
r = 1.
c = [0.,0.,0.]
x = r*numpy.cos(u)*numpy.sin(v)
y = r*numpy.sin(u)*numpy.sin(v)
z = r*numpy.cos(v)

ax.plot_surface(x-c[0], y-c[1], z-c[2], color=('k'))

ax.set_xlabel("x (JSS)")
ax.set_ylabel("y (JSS)")
ax.set_zlabel("z (JSS)")


directory_path_out = "/Users/mcentees/Desktop/JUNO_trajectory_plots/"
filename_out = directory_path_out+'Juno_trajectory'

date_beg = datetime.datetime(2016,6,20)
date_end = datetime.datetime(2022,12,21)

x_plot = x_coord[(date_ephem >= date_beg) & (date_ephem <= date_end)]
y_plot = y_coord[(date_ephem >= date_beg) & (date_ephem <= date_end)]
z_plot = z_coord[(date_ephem >= date_beg) & (date_ephem <= date_end)]

ax.plot3D(x_plot, y_plot, z_plot, color = 'k', lw=1.0)

hdulist = pyfits.open(f'/Users/mcentees/Desktop/Will_code/hrcf23369_pytest_evt2_change.fits', dtype=float)
img_head = hdulist[1].header
date = img_head['DATE-OBS'] # Start date of observation
date_end = img_head['DATE-END'] # Start date of observation
tstart = img_head['TSTART']
tstop = img_head['TSTOP']
hdulist.close()

alldata = pd.read_csv('/Users/mcentees/Desktop/Chandra/23369/primary/23369_photonlist_PI_filter_Jup_full_10_250.txt')
alldata.columns = alldata.columns.str.replace("# ", "")

data_start = min(alldata['t(s)'])
data_end = max(alldata['t(s)'])

lighttravel = 33.9115185511787
tstartminusLT = data_start - lighttravel * 60
tstopminusLT = data_end - lighttravel * 60

date_chandra_obs_beg = [Time(tstartminusLT, format='cxcsec').datetime]
date_chandra_obs_end = [Time(tstopminusLT, format='cxcsec').datetime]

color = 'm'
marker = '.'
color_marker = 'm.'

for i_crossing in range(len(date_chandra_obs_beg)):
    x_cross = x_coord[(date_ephem >= date_chandra_obs_beg[i_crossing]) & (date_ephem <= date_chandra_obs_end[i_crossing])]
    y_cross = y_coord[(date_ephem >= date_chandra_obs_beg[i_crossing]) & (date_ephem <= date_chandra_obs_end[i_crossing])]
    z_cross = z_coord[(date_ephem >= date_chandra_obs_beg[i_crossing]) & (date_ephem <= date_chandra_obs_end[i_crossing])]
    # ax.plot3D(x_cross, y_cross, z_cross, c = 'g', zorder=10, lw=2.5)
    ax.plot3D(x_cross, y_cross, z_cross, c = 'g', marker=marker, zorder=10)


# for i_azim in range(0,360,90):
#     ax.view_init(elev = 10,azim=i_azim)
#     plt.tight_layout()
#     plt.savefig(filename_out+"%d.png" %i_azim, dpi = 500, bbox_inches='tight')

ax.view_init(elev = 10, azim = 60)
plt.tight_layout()
plt.savefig(filename_out+"60_JSS_lt.png", dpi = 500, bbox_inches='tight')

plt.show()

