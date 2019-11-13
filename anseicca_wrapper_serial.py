#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

# Custom module imports
use_reald=False
import anseicca_utils1 as u1
import anseicca_utils2 as u2
import hans2013_serial as h13

########################################################### ESTABLISH THE STATION/RECEIVER NETWORK ########################################################

nrecs=289
origx = 406; origy = 8504 # coordinate origin
coordfile="EXAMPLES/coordinates_receivers_h13format.csv"

################################################################ DEFINE COMPUTATIONAL DOMAIN ##############################################################
hlbox_outer = 60.
# half-length of outer box OR length of (side of) inner box, in km
ngp_inner = 241
# number of grid points in inner box (half the number of grid points in outer box)
d_xy=hlbox_outer/(ngp_inner-1)
#*** grid location error threshold
Delta_thresh = 5 #km
glerr_thresh = 0.0008*Delta_thresh
wspeed = 3.0 
# wavespeed everywhere in model (km/s)
map_plots=True

stno, stid, stx, sty = u1.read_station_file(coordfile)
smdo=u2.setup_modelling_domain(stno,stid,stx,sty,origx,origy,d_xy,glerr_thresh,map_plots)

############################################################### DATA AND/OR DATA CHARACTERISTICS ##########################################################
sig_char = u1.SignalParameters()

if not use_reald:
# SYNTHETIC TEST CASE
	sig_char.dt = 0.2
	sig_char.nsam = 250
	sig_char.cf = 0.3
	sig_char.lf = None
	sig_char.hf = None
	sig_char.altukey = None

	obsdata=None
	obsdata_info=None
else:
# REAL DATA CASE
	data_format = {0: 'python_binary_archive'}
	infile="EXAMPLES/stack_manypairs_99_zz.npz"

	rdo=u2.cc_data.Read(infile,data_format[0])
	mdo=u2.cc_data.MatrixForm(stno,stx,sty,smdo.num_chosen,smdo.chosen_st_no)
	pdo=u2.cc_data.Process(wspeed,smdo.num_chosen)
	azstep=90
	edo=u2.cc_data.Errors(smdo.num_chosen,azstep)

	sig_char.dt = pdo.dt
	sig_char.nsam = pdo.nsam
	sig_char.lf = rdo.fpb[0]
	sig_char.hf = rdo.fpb[1]
	sig_char.cf = (rdo.fpb[0]+rdo.fpb[1])/2
	sig_char.altukey = rdo.fpb[2]

	obsdata = pdo.use_data
	obsdata_info = (mdo.act_dist_rp, pdo.snr, edo.DelE)

####################################################################### RUN CORE CODE ######################################################################

ratio_boxes=2
rad_ring=25
w_ring=75
do_inv=True
iter_only1=False

dummy1 = (ratio_boxes*(ngp_inner-1) + 1)**2 * ((sig_char.nsam)/2+1) * 16 * 1e-9
dummy2 = sig_char.nsam * (smdo.num_chosen**2) * 16 *1e-9
print "Memory requirement for besselmat array will be %f GB" %(dummy1)
print "Memory requirement for syncross array will be %f GB" %(dummy2)

icao = h13.inv_cc_amp(hlbox_outer,ngp_inner,ratio_boxes,smdo.num_chosen,smdo.rchosenx_igp,smdo.rchoseny_igp,wspeed,sig_char,rad_ring,w_ring,do_inv,iter_only1,obsdata,obsdata_info)

######################################################################### POST RUN #########################################################################

try:
	icao
except NameError:
	# icao is not defined
	print "h13 module not used. Showing plot(s)..."
	plt.show()
	# for plots that are made before running the h13 module
else:
	if do_inv:
	# Inversion has been run (at least one iteration)
		print "\nStoring the result..."
		u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,0)
	else:
	# Models have been setup but inversion has NOT been run
		u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,1)
		plt.show()
