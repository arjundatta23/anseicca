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

##################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

use_reald=False
# this variable is required on all processors, in Part 1 (SETUP phase), and it is seen by module u2 at load time
# which is why it must be defined before importing the module.

# Custom module imports
import anseicca_utils1 as u1
import anseicca_utils2 as u2
import hans2013_parallel as h13

#***************************************** Initialize MPI process **********************************************
comm_out = MPI.COMM_WORLD
rank_out = comm_out.Get_rank()
numproc_out = comm_out.Get_size()

############################################## PART 1 (SETUP)  ##################################################

#------------------------------------- Outline of computational domain ------------------------------------------

hlbox_outer = 60.
# half-length of outer box OR length of (side of) inner box, in km
ngp_inner = 241
# number of grid points in inner box (half the number of grid points in outer box)
d_xy=hlbox_outer/(ngp_inner-1)
wspeed = 3.0 
# wavespeed everywhere in model (km/s)

if rank_out==0:

	#--------------------------------- Establish the station/receiver network --------------------------------
	nrecs=289

	coordfile="EXAMPLES/coordinates_receivers_h13format.csv"
	origx = 406; origy = 8504 # coordinate origin
	stno, stid, stx, sty = u1.read_station_file(coordfile)

	#-------------------------------- Complete setup of computational domain ---------------------------------

	# grid location error threshold
	Delta_thresh = 5 #km
	glerr_thresh = 0.0226*Delta_thresh
	map_plots=True
	
	smdo=u2.setup_modelling_domain(stno,stid,stx,sty,origx,origy,d_xy,glerr_thresh,map_plots)
	if smdo.num_chosen != numproc_out:
		raise SystemExit("Quitting - number of receivers incompatible with number of processors.")

	#---------------------------------- Data and/or data characteristics -------------------------------------

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

############################################## PART 2 (CORE CODE) ######################################################

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
	print "Memory requirement for besselmat array will be %f GB" %(dummy1)
	print "Memory requirement for syncross array will be %f GB" %(dummy2)

icao = h13.inv_cc_amp(comm_out,rank_out,hlbox_outer,ngp_inner,ratio_boxes,nr,rc_xp,rc_yp,wspeed,sigchar,rad_ring,w_ring,do_inv,iter_only1,odata,odata_info)

############################################## PART 3 (POST RUN) ##########################################################

try:
	icao
except NameError:
	# icao is not defined
	print "h13 module not used. Showing plot(s)..."
	plt.show()
	# for plots that are made before running the h13 module
else:
	if rank_out==0:
		if do_inv:
		# Inversion has been run (at least one iteration)
			print "\nStoring the result..."
			u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,0)
		else:
		# Models have been setup but inversion has NOT been run
			u2.post_run(wspeed,smdo,icao,sig_char,rad_ring,w_ring,1)
			plt.show()
