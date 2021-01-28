#!/usr/bin/python

"""
Code by Arjun Datta at Tata Institute of Fundamental Research, Mumbai, India

PARALLEL CODE

Author Notes: All the user settings in this code wrapper fall into two categories, which I call Type 1 and Type 2
		Type 1: those parameters whose values are typically inspired by the user's data set, even when doing synthetic tests only
			 (e.g. geometry of the problem, wavelengths of interest etc.)
		Type 2: parameters whose values are the user's personal choice, independent of data/problem at hand

	      Refer to serial version of code for detailed comments including parameter Types.
"""

###############################################################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

sys.path.append(os.path.expanduser('~/code_general/modules.python'))
# path to the "SW1D_earthsr" set of modules

###################################################### PART 1 (SETUP)  #########################################################

#***************************************** USER CHOICES MODELLING TYPE *****************************************

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

#----------------------------------------- Synthetic or real data ----------------------------------------------
use_reald=False
# this variable is required on all processors, in Part 1 (SETUP phase), and it is seen by module u2 at load time
# which is why it must be defined before importing the module.

#----------------------------------------- Initialize MPI process -----------------------------------------------
comm_out = MPI.COMM_WORLD
rank_out = comm_out.Get_rank()
numproc_out = comm_out.Get_size()

#------------------------------------- Outline of computational domain ------------------------------------------

hlbox_outer = 60.
# size of modelling domain (km units); length of (side of) inner box OR half-length of outer box
ngp_inner = 241
# number of grid points in inner box (half the number of grid points in outer box)
d_xy=hlbox_outer/(ngp_inner-1)
# wspeed = 3.0
# # wavespeed everywhere in model (km/s)

# Custom modules: set 1
import SW1D_earthsr.utils_pre_code as u0

#-------------------------------- Model reading (elastic case) ----------------------------------
if elastic:
    wspeed=3.0 # NEED TO CHANGE, SHOULD BE BASED ON VELOCITIES FROM DISP FILE
    # read input depth-dependent model and fix/extract necessary parameters
    if rank_out==0:
        upreo = u0.model_1D(mod1dfile)
        dep_pts_mod = upreo.deps_all
        hif_mod = upreo.mod_hif
        upreo.fix_max_depth(dep_pts_mod[1])
        dep_pts_use = upreo.deps_tomax
        # hif_mod_use = hif_mod[hif_mod<=config.dmax]
        print("Layer interfaces in model: ", hif_mod, hif_mod.size)
        print("Depth points to be used in code: ", dep_pts_use, dep_pts_use.size)
        nz = dep_pts_use.size
else:
    # nz = None
    wspeed = 3.0
    # wavespeed everywhere in model (km/s); Type 1 (but could be Type 2 if doing synthetics)
#-------------------------------- End model reading (elastic case) ----------------------------------

# Custom modules: set 2
import anseicca_utils1 as u1
import anseicca_utils2 as u2
import hans2013_parallel as h13

if rank_out==0:

    #--------------------------------- Establish the station/receiver network --------------------------------
    nrecs=289

    coordfile=os.path.expanduser('~/anseicca/EXAMPLES/coordinates_receivers_h13format.csv')
    origx = 406; origy = 8504 # coordinate origin
    stno, stid, stx, sty = u1.read_station_file(coordfile)

    #-------------------------------- Complete setup of computational domain ---------------------------------

    # grid location error threshold
    Delta_thresh = 5 #km
    glerr_thresh = 0.0016*Delta_thresh # 0.0008 for 5, 0.0016 for 16, 0.0025 for 25, 0.00281 for 32, 0.0051 for 64, 0.008 for 100, 0.0226 for 256
    map_plots=False

    smdo=u2.setup_modelling_domain(stno,stid,stx,sty,origx,origy,d_xy,glerr_thresh,map_plots)
    if smdo.num_chosen != numproc_out:
    	raise SystemExit("Quitting - number of receivers incompatible with number of processors.")

    #---------------------------------- Data and/or data characteristics -------------------------------------
    # Temporal signal characteristics
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
    	azstep=4
    	edo=u2.cc_data.Errors(smdo.num_chosen,azstep)

    	sig_char.dt = pdo.dt
    	sig_char.nsam = pdo.nsam
    	sig_char.lf = rdo.fpb[0]
    	sig_char.hf = rdo.fpb[1]
    	sig_char.cf = (rdo.fpb[0]+rdo.fpb[1])/2
    	sig_char.altukey = rdo.fpb[2]

    	obsdata = pdo.use_data
    	obsdata_info = (mdo.act_dist_rp, pdo.snr, edo.DelE)

    #--------------------- store into simple variables those quantities that need to be broadcasted ---------------
    num_chosen = smdo.num_chosen
    rchosenx_igp = smdo.rchosenx_igp
    rchoseny_igp = smdo.rchoseny_igp

else:
    num_chosen=None
    rchosenx_igp=None
    rchoseny_igp=None
    sig_char=None
    act_dist_rp=None
    obsdata=None
    obsdata_info=None

################################################## PART 2 (CORE CODE) ##########################################################

# variables required on all processors
ratio_boxes=2
rad_ring=25
w_ring=75
do_inv=True
iter_only1=False

# Broadcast variables computed on master processor but required on all processors
nr = comm_out.bcast(num_chosen,root=0)
sigchar = comm_out.bcast(sig_char, root=0)
rc_xp = comm_out.bcast(rchosenx_igp, root=0)
rc_yp = comm_out.bcast(rchoseny_igp, root=0)
odata = comm_out.bcast(obsdata, root=0)
odata_info = comm_out.bcast(obsdata_info, root=0)

# a little check of memory requirements
if rank_out==0:
	dummy1=(ratio_boxes*(ngp_inner-1) + 1)**2 * ((sig_char.nsam)/2+1) * 16 * 1e-9
	dummy2 = sig_char.nsam * (smdo.num_chosen**2) * 16 *1e-9
	print("Memory requirement for Green array will be %f GB" %(dummy1))
	print("Memory requirement for syncross array will be %f GB" %(dummy2))

icao = h13.inv_cc_amp(hlbox_outer,ngp_inner,ratio_boxes,nr,\
rc_xp,rc_yp,wspeed,sigchar,rad_ring,w_ring,do_inv,iter_only1,odata,odata_info)

################################################## PART 3 (POST RUN) ############################################################

try:
	icao
except NameError:
	# icao is not defined
	print("h13 module not used. Showing plot(s)...")
	plt.show()
	# for plots that are made before running the h13 module
else:
	if rank_out==0:
		if do_inv:
		# Inversion has been run (at least one iteration)
			print("\nStoring the result...")
			u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,0)
		else:
		# Models have been setup but inversion has NOT been run
			u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,1)
			plt.show()
