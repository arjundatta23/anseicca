#!/usr/bin/python

"""
Code by Arjun Datta at Tata Institute of Fundamental Research, Mumbai, India

SERIAL CODE

Author Notes: All the user settings in this code wrapper fall into two categories, which I call Type 1 and Type 2
		Type 1: those parameters whose values are typically inspired by the user's data set, even when doing synthetic tests only
			 (e.g. geometry of the problem, wavelengths of interest etc.)
		Type 2: parameters whose values are the user's personal choice, independent of data/problem at hand

	      As far as possible I have tried to indicate (in comments), the Type of all parameters in this wrapper, for the benefit of new users.
	      Please also read the accompanying README file.

"""

###########################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

use_reald=False #Type 2
# this variable is seen by module u2 at load time which is why it must be defined before importing the module.

# Custom module imports
import anseicca_utils1 as u1
import anseicca_utils2 as u2
import hans2013_serial as h13

########################################################### ESTABLISH THE STATION/RECEIVER NETWORK ########################################################

nrecs=289
# no. of stations/receivers in the network; Type 1
origx = 406; origy = 8504
# coordinate origin of the modelling domain (km units); ballpark values chosen to have the origin roughly at the network centre; Type 1
coordfile="EXAMPLES/coordinates_receivers_h13format.csv"

################################################################ DEFINE COMPUTATIONAL DOMAIN ##############################################################
hlbox_outer = 60.
# size of modelling domain (km units); length of (side of) inner box OR half-length of outer box; Type 1
ngp_inner = 241
# number of grid points in inner box (half the number of grid points in outer box); determines the grid density and therefore frequency resolution; Type 1
d_xy=hlbox_outer/(ngp_inner-1)
Delta_thresh = 5 #km
glerr_thresh = 0.0008*Delta_thresh
# grid location error threshold; Type 2
wspeed = 3.0 
# wavespeed everywhere in model (km/s); Type 1 (but could be Type 2 if doing synthtics)
map_plots=True

stno, stid, stx, sty = u1.read_station_file(coordfile)
smdo=u2.setup_modelling_domain(stno,stid,stx,sty,origx,origy,d_xy,glerr_thresh,map_plots)

############################################################### DATA AND/OR DATA CHARACTERISTICS ##########################################################
sig_char = u1.SignalParameters()

if not use_reald:
# SYNTHETIC TEST CASE
	sig_char.dt = 0.2 # sampling interval
	sig_char.nsam = 250 # number of samples
	sig_char.cf = 0.3 # central frequency of noise sources
	sig_char.lf = None # lower bound frequency
	sig_char.hf = None # upper bound frequency
	sig_char.altukey = None # alpha parameter for Tukey window in spectral domain from lf to hf

	# all of these refer to modelled cross-correlation waveforms; all Type 1 if you're doing synthetic tests that match your real data scenario

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

ratio_boxes=2 # parameter relevant for synthetic tests only; controls the size of the modelling domain; see h13 module for details; Type 2
rad_ring=25 # radius (km) of the ring of sources - model parameterization as in Datta et al. (2019); Type 1/2
w_ring=75 # gaussian width of ring of sources (grid spacing units); Type 2
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
