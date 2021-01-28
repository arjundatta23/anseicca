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

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/code_general/modules.python'))
# path to the "SW1D_earthsr" set of modules

################################################### USER CHOICES MODELLING TYPE ################################################

#-------------------------------- SCALAR vs. ELASTIC and tensor component (if elastic) ----------------------------------
if len(sys.argv)>1:
	scalar=False
	elastic=True
	mod1dfile=sys.argv[1]
	egn_ray=sys.argv[2] # eigenfunctions Rayleigh
	disp_ray=sys.argv[3] # dispersion Rayleigh
	try:
		egn_lov=sys.argv[4] # eigenfunctions Love
		disp_lov=sys.argv[5] # dispersion Love
	except IndexError:
		pass
	cctc = 3
    # cctc -> cross-correlation_tensor_component(s)
else:
	scalar=True
	elastic=False
	cctc = 3 # DO NOT EDIT!
	# the code is written so as to implement the vector (3-D) and scalar (2-D) cases in a consistent manner,
	# so scalar quantities are treated as the 'z-component' of a 3-D cartesian system.

tensor_comp_ray = {0:'RR', 1:'RZ', 2:'ZR', 3:'ZZ'}
rtz_xyz = {'R': 'x', 'T': 'y', 'Z': 'z'} # valid ONLY for receivers on the x-axis!

comp_p = rtz_xyz[tensor_comp_ray[cctc][0]]
comp_q = rtz_xyz[tensor_comp_ray[cctc][1]]

#-------------------------------- Synthetic or real data ----------------------------------
use_reald=False #Type 2
# this variable is seen by module u2 at load time which is why it must be defined before importing the module.

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
glerr_thresh = 0.0008*Delta_thresh # 0.0008 for 5, 0.0016 for 16, 0.0025 for 25, 0.00281 for 32, 0.0051 for 64
# grid location error threshold; Type 2
map_plots=True

# Custom modules: set 1
import SW1D_earthsr.utils_pre_code as u0

#-------------------------------- Model reading (elastic case) ----------------------------------
if elastic:
# read input depth-dependent model and fix/extract necessary parameters
    upreo = u0.model_1D(mod1dfile)
    dep_pts_mod = upreo.deps_all
    hif_mod = upreo.mod_hif
    upreo.fix_max_depth(dep_pts_mod[1])
    dep_pts_use = upreo.deps_tomax
    # hif_mod_use = hif_mod[hif_mod<=config.dmax]
    print("Layer interfaces in model: ", hif_mod, hif_mod.size)
    print("Depth points to be used in code: ", dep_pts_use, dep_pts_use.size)
    nz = dep_pts_use.size
    wspeed=3.0 # NEED TO CHANGE, SHOULD BE BASED ON VELOCITIES FROM DISP FILE
else:
    # nz = None
    wspeed = 3.0
    # wavespeed everywhere in model (km/s); Type 1 (but could be Type 2 if doing synthetics)
#-------------------------------- End model reading (elastic case) ----------------------------------

# Custom modules: set 2
import anseicca_utils1 as u1
import anseicca_utils2 as u2
import hans2013_serial as h13

stno, stid, stx, sty = u1.read_station_file(coordfile)
smdo=u2.setup_modelling_domain(stno, stid, stx, sty, origx, origy, d_xy, glerr_thresh, map_plots)

############################################################### DATA AND/OR DATA CHARACTERISTICS ##########################################################

#-------------------------------- Temporal signal characteristics ----------------------------------

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
iter_only1=True

dummy1 = (ratio_boxes*(ngp_inner-1) + 1)**2 * ((sig_char.nsam)/2+1) * 16 * 1e-9
dummy2 = sig_char.nsam * (smdo.num_chosen**2) * 16 *1e-9
print("Memory requirement for Green array will be %f GB" %(dummy1))
print("Memory requirement for syncross array will be %f GB" %(dummy2))

icao = h13.inv_cc_amp(hlbox_outer,ngp_inner,ratio_boxes,smdo.num_chosen,smdo.rchosenx_igp,\
smdo.rchoseny_igp,wspeed,sig_char,rad_ring,w_ring,do_inv,iter_only1,obsdata,obsdata_info)

######################################################################### POST RUN #########################################################################

try:
	icao
except NameError:
	# icao is not defined
	print("h13 module not used. Showing plot(s)...")
	plt.show()
	# for plots that are made before running the h13 module
else:
	if do_inv:
	# Inversion has been run (at least one iteration)
		print("\nStoring the result...")
		u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,0)
	else:
	# Models have been setup but inversion has NOT been run
		u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,1)
		plt.show()
