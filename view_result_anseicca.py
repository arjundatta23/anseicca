#!/usr/bin/python

# General purpose modules
import os
import sys
import gzip
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Modules written by me
import anseicca_utils1 as u1

####################################### coordinates file for receiver distances #################################################

coordfile="/home/arjun/postdoc/Shell_data_Albania/Supporting_Documentation/coordinates_receivers_h13format.csv"
# this is on the desktop machine
if not os.path.isfile(coordfile):
	# this is on the cluster
	coordfile="/home/arjun/backup_desktop/Shell_data_Albania/Supporting_Documentation/coordinates_receivers_h13format.csv"

rnum, rid, xloc, yloc = u1.read_station_file(coordfile)

#cfh=open(coordfile,'r')
#entire=cfh.readlines()
#rnum=np.array(map(lambda p: int(p.split()[0]), entire))
#xloc=np.array(map(lambda p: float(p.split()[2])/1e3, entire))
#yloc=np.array(map(lambda p: float(p.split()[3])/1e3, entire))
#
#cfh.close()
#del cfh

#################################################################################################################################

def plot_kernels():

	fig1=plt.figure()
	ax1=fig1.add_subplot(111)
	ax1.set_title("Initial misfit kernel: positive branch")
	cax1=ax1.pcolor(kcao_gx,kcao_gy,kcao_mfkp,cmap=plt.cm.jet)
	ax1.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	if len(rc_xp)<13:
	# plot the station numbers as seen by the h13 module
		for i in range(len(rc_xp)):
			ax1.annotate(i, xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]))
			#pass
	ax1.tick_params(axis='both', labelsize=14)
	fig1.colorbar(cax1)

	fig2=plt.figure()
	ax2=fig2.add_subplot(111)
	ax2.set_title("Initial misfit kernel: negative branch")
	cax2=ax2.pcolor(kcao_gx,kcao_gy,kcao_mfkn,cmap=plt.cm.jet)
	ax2.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	ax2.tick_params(axis='both', labelsize=14)
	# plot the actual station numbers of the real data set
	if len(rc_xp)<13:
		for i in range(len(rc_xp)):
			ax2.annotate(rc[i], xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]))
	fig2.colorbar(cax2)

################################################################################################################################

def plot_models():

	diff_sizes=False
	if not reald:
		fig4=plt.figure()
		ax4=fig4.add_subplot(111)
		ax4.set_title("True model")

		if kcao_sdtrue.shape[0]==kcao_sdstart.shape[0]:
			xpts_true = kcao_gx
			ypts_true = kcao_gy
		else:
			hlbox_outer = kcao_dx*(kcao_gx.shape[0]-1)
			ngp_outer = 2*kcao_gx.shape[0] - 1
			xobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
			yobox=np.linspace(-hlbox_outer,hlbox_outer,ngp_outer)
			xpts_true, ypts_true = np.meshgrid(xobox, yobox)
			diff_sizes = True
			
		cax4=ax4.pcolor(xpts_true,ypts_true,kcao_sdtrue,cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
		ax4.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
		ax4.tick_params(axis='both', labelsize=14)
		#for i in range(len(rc_xp)):
		#	ax4.annotate(i, xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]), color='green')
		plt.colorbar(cax4,ax=ax4)
		print "Min and max values in True model: ", np.amin(kcao_sdtrue), np.amax(kcao_sdtrue)

	fig3=plt.figure()
	ax3=fig3.add_subplot(111)
	ax3.set_title("Starting model")
	if diff_sizes:
		ax3.set_xlim(-hlbox_outer,hlbox_outer)
		ax3.set_ylim(-hlbox_outer,hlbox_outer)
	try:
		cax3=ax3.pcolor(kcao_gx,kcao_gy,kcao_sdstart,cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
	except NameError:
		cax3=ax3.pcolor(kcao_gx,kcao_gy,kcao_sdstart,cmap=plt.cm.jet)
	ax3.tick_params(axis='both', labelsize=14)
	ax3.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	for i in range(len(rc_xp)):
		ax3.annotate(i, xy=(kcao_dx*rc_xp[i],kcao_dx*rc_yp[i]), color='white')
	plt.colorbar(cax3,ax=ax3)

	fig5=plt.figure()
	ax5=fig5.add_subplot(111) #, aspect='equal')
	ax5.set_title("Inversion result")
	#if diff_sizes:
	#	ax5.set_xlim(-hlbox_outer,hlbox_outer)
	#	ax5.set_ylim(-hlbox_outer,hlbox_outer)
	print "Min and max values in inverted result: ", np.amin(kcao_sdinv), np.amax(kcao_sdinv)
	try:
		cax5=ax5.pcolor(kcao_gx,kcao_gy,kcao_sdinv,cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
		#cax5=ax5.pcolor(kcao_gx,kcao_gy,kcao_sdinv,cmap=plt.cm.jet,vmin=0.0,vmax=1.25)
		#cax5=ax5.pcolor(kcao_gx,kcao_gy,kcao_sdinv,cmap=plt.cm.jet)
	except NameError:
		cax5=ax5.pcolor(kcao_gx,kcao_gy,kcao_sdinv,cmap=plt.cm.jet)
	ax5.tick_params(axis='both', labelsize=14)
	ax5.plot(kcao_dx*rc_xp, kcao_dx*rc_yp, 'wd', markerfacecolor="None")
	plt.colorbar(cax5,ax=ax5) #, orientation='horizontal',fraction=0.04)

###############################################################################################################

def plot_inversion_progress(bs):

	def deltad_iter():

		# determine appropriate histogram bins such that "0" is a bin center
		mv1=max(max(kcao_flit_indmis_p[0]),max(kcao_flit_indmis_n[0]))
		mv2=min(min(kcao_flit_indmis_p[0]),min(kcao_flit_indmis_n[0]))
		maxval=np.round(max(abs(mv1),abs(mv2)))
		print "Max val for histograms is: ", maxval
		hbe_p=np.arange(bs/2,maxval+bs,bs)
		hbe_n=-1*hbe_p[::-1]
		hbe = np.hstack((hbe_n,hbe_p))
		#hbe -> histogram_bin_edges
		#print hbe

		ntot="$N_{tot}$\n= %d" %(npairs)

		#fig = plt.figure(figsize=(5.5,10))
		#axh_p, axh_n = fig.subplots(2,1,sharex=True,sharey=True)
		fig = plt.figure()
		ncen="$N_{cen}$\n= %d" %(good_after_p.size)
		axh_p, axh_n = fig.subplots(1,2,sharex=True,sharey=True)
		axh_p.hist(kcao_flit_indmis_p[0],bins=hbe,edgecolor='black')
		axh_p.hist(kcao_flit_indmis_p[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_p.set_xlabel(r'$\Delta d$')
		axh_p.set_xlabel(r'$\ln \left( A^{obs}/A^{syn} \right)$', fontsize=16)
		#axh_p.set_ylabel("No. of pairs")#, fontsize=14)
		axh_p.set_title("Positive branch")#, fontsize=18)
		axh_p.tick_params(labelsize=14)
		axh_p.text(0.7, 0.8, ntot, transform=axh_p.transAxes)
		axh_p.text(0.7, 0.7, ncen, transform=axh_p.transAxes)

		ncen="$N_{cen}$\n= %d" %(good_after_n.size)
		axh_n.hist(kcao_flit_indmis_n[0],bins=hbe,edgecolor='black')
		axh_n.hist(kcao_flit_indmis_n[-1],bins=hbe,histtype='step',linewidth='1.5')
		#axh_n.set_xlabel(r'$\Delta d$')#, fontsize=18)
		axh_n.set_xlabel(r'$\ln \left( A^{obs}/A^{syn} \right)$', fontsize=16)
		#axh_n.set_ylabel("No. of pairs")#, fontsize=14)
		axh_n.set_title("Negative branch")#, fontsize=18)
		axh_n.tick_params(labelsize=14)
		axh_n.text(0.1, 0.8, ntot, transform=axh_n.transAxes)
		axh_n.text(0.1, 0.7, ncen, transform=axh_n.transAxes)

	def chi_iter():

		nchid = kcao_allit_misfit/np.amax(kcao_allit_misfit)

		its=range(numit)
		fig=plt.figure()
		ax=fig.add_subplot(111)
		#ax.set_title("Inversion progress: total misfit", fontsize=18)
		try:
			ax.plot(its,nchid,'-o')
		except ValueError:
			ax.plot(its[:-1],nchid,'-o')
		ax.xaxis.set_ticks(its)
		ax.ticklabel_format(axis='y',style='scientific',scilimits=(-2,2))
		ax.set_ylabel(r"$\chi_d(m_k)$", fontsize=14)
		ax.set_xlabel("k, iteration number", fontsize=14)
		ax.tick_params(labelsize=14)
		#plt.xticks(fontsize=14)

	def mod_iter():

		fig=plt.figure()
		#pub = numit-1
		#for p in range(pub):
		for p in range(numit):
			#it=p+1
			#spname = "k=%d" %(it)
			spname = "k=%d" %(p)
			axsp=fig.add_subplot(3,3,p+1) #,aspect='equal')
			cax=axsp.pcolor(kcao_gx,kcao_gy,kcao_sditer[p,:,:],cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
			axsp.text(0.8,0.85,spname,transform=axsp.transAxes,color='white')
			#axsp.set_title(spname)
			#if p==pub-1:
			#	plt.colorbar(cax,ax=axsp,orientation="horizontal")

	chi_iter()
	deltad_iter()
	mod_iter()

###############################################################################################################

def plot_Lcurve():
	print "Model norm: ", sorted_modnorm
	print "Misfit: ", sorted_misfit
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(sorted_modnorm,sorted_misfit)
	for p in range(len(filelist)):
		gval = np.sort(gamma_files)[p]
		if gval>=1 or gval==0:
			plabel=r"$\gamma = %d$" %(gval)
		else:
			dp=abs(int(np.floor(np.log10(gval))))
			plabel=r"$\gamma = %.*f$" %(dp,gval)
		ax.plot(sorted_modnorm[p],sorted_misfit[p],'o',label=plabel)
	ax.legend()
	ax.set_xlabel("Model norm (relative)")
	ax.set_ylabel("Misfit")
	

################################################ Main program ############################################################

if __name__ == '__main__':

	inarg=sys.argv[1]
	if os.path.isdir(inarg):
		filelist=[os.path.join(inarg,n) for n in os.listdir(inarg) if n.endswith('.pckl')]
		nrecs_files=np.zeros(len(filelist))
		gamma_files=np.zeros(len(filelist))
		misfit_files=np.zeros(len(filelist))
		modnorm_files=np.zeros(len(filelist))
	elif os.path.isfile(inarg):
		filelist=[inarg]

	#********************************* Read the pickle file(s) **************************************************
	for p, pfile in enumerate(filelist):

		jar=gzip.open(pfile)
		print "Reading ", pfile
		reald=pickle.load(jar)
		rc=pickle.load(jar)
		rc_xp=pickle.load(jar)
		rc_yp=pickle.load(jar)
		dc=pickle.load(jar)
		kcao_fhz=pickle.load(jar)
		kcao_pss=pickle.load(jar)
		kcao_dx=pickle.load(jar)
		kcao_gx=pickle.load(jar)
		kcao_gy=pickle.load(jar)
		kcao_rr=pickle.load(jar)
		kcao_rgw=pickle.load(jar)
		kcao_mfkp=pickle.load(jar)
		kcao_mfkn=pickle.load(jar)
		#kcao_syncross=pickle.load(jar)
		#kcao_obscross=pickle.load(jar)
		kcao_gamma=pickle.load(jar)
		kcao_allit_mc=pickle.load(jar)
		kcao_allit_misfit=pickle.load(jar)
		kcao_flit_indmis_p=pickle.load(jar)
		kcao_flit_indmis_n=pickle.load(jar)

		num_points = kcao_gx.shape[0]
		if not reald:
			trumod = pickle.load(jar)
			if len(trumod.shape)==1:
			# implies basis coefficients are stored
				mc_true=trumod
				kcao_sdtrue = np.zeros((num_points,num_points))
			elif len(trumod.shape)==2:
			# implies the complete model itself is stored (not specified in terms of basis)
				kcao_sdtrue=trumod
		jar.close()

		if len(filelist)>1:
			nrecs_files[p]=len(rc_xp)
			gamma_files[p]=kcao_gamma
			misfit_files[p]=kcao_allit_misfit[-1]
			#mod_norm = np.sum(np.square(kcao_allit_mc[-1]))
			mod_norm = np.sum(np.square(kcao_allit_mc[-1]))/np.sum(np.square(kcao_allit_mc[0]))
			modnorm_files[p]=mod_norm
		else:
			print "gamma value is: ", kcao_gamma
			print "Number of iterations: ", len(kcao_allit_misfit), len(kcao_allit_mc)

	#************************************** Finished reading file(s) **********************************************

	if len(filelist)>1:
		if len(np.unique(nrecs_files))>1:
			sys.exit("The different pickles have different number of receivers - script terminated.")

		sortind = np.argsort(gamma_files)
		sorted_misfit = misfit_files[sortind]
		sorted_modnorm = modnorm_files[sortind]
		print "Sorted gamma values: ", np.sort(gamma_files)

		plot_Lcurve()
	else:
		misfit_start = 0.5*np.sum(np.square(kcao_flit_indmis_p[0]) + np.square(kcao_flit_indmis_n[0]))
		misfit_end = 0.5*np.sum(np.square(kcao_flit_indmis_p[-1]) + np.square(kcao_flit_indmis_n[-1]))
		if np.round(misfit_start,6) != np.round(kcao_allit_misfit[0],6) or np.round(misfit_end,6) != np.round(kcao_allit_misfit[-1],6):
			print "Initial misfit with and without errors: ", kcao_allit_misfit[0], misfit_start
			print "Final misfit with and without errors: ", kcao_allit_misfit[-1], misfit_end

		nrecs = len(rc_xp)

		#********************************** compute interstation distances ********************************************

		# NB: these are ACTUAL distances, based on the coordinates file, NOT effective distances from kcao

		adrp=np.zeros((nrecs,nrecs))
		# adrp -> actual_distance_receiver_pairs

		for b,brec in enumerate(rc[:-1]):
			urecs=rc[b+1:]
			x1=xloc[np.searchsorted(rnum,brec)]
			y1=yloc[np.searchsorted(rnum,brec)]
			x2=xloc[np.searchsorted(rnum,urecs)]
			y2=yloc[np.searchsorted(rnum,urecs)]
			adrp[b+1:,b]=np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
			adrp[b,b+1:]=adrp[b+1:,b]
		
		#******************* get all necessary quantities in vector (1-D array) form **********************************
		npairs=nrecs*(nrecs-1)/2
		rp_orig=range(npairs)
		rp_kcao=range(npairs)
		rp_dist=range(npairs)
		cp=0
		for x in itertools.combinations(range(nrecs),2):
			rp_kcao[cp]=x[::-1]
			# reversal of the pair ordering is required to match what is done in h13 (lower triangular matrix)
			rp_orig[cp]=(rc[x[1]],rc[x[0]])
			rp_dist[cp]=adrp[x]
			cp+=1

		#******** Determine the paths (receiver pairs) for which waveform fits have improved through inversion ********
		
		binsize=0.5 #2.5 #0.5
		# this is the bin size for histogram of misfits (see func "plot_inversion_progress" -> "deltad_iter")

		bad_thresh = 1.3*binsize
		good_thresh = binsize/2 # DO NOT CHANGE THIS. CHANGE ONLY THE BINSIZE

		bad_before_p = np.where(abs(kcao_flit_indmis_p[0])>bad_thresh)[0]
		bad_before_n = np.where(abs(kcao_flit_indmis_n[0])>bad_thresh)[0]

		try:
			good_after_p = np.where(abs(kcao_flit_indmis_p[-1])<good_thresh)[0]
		except IndexError:
			good_after_p = np.array([])

		try:
			good_after_n = np.where(abs(kcao_flit_indmis_n[-1])<good_thresh)[0]
		except IndexError:
			good_after_n = np.array([])
		
		nga_total = good_after_p.size + good_after_n.size
		good_frac = float(nga_total)/(2*npairs)
		print "Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(nga_total,100*good_frac)

		irpi_pos = np.intersect1d(bad_before_p, good_after_p)
		irpi_neg = np.intersect1d(bad_before_n, good_after_n)
		# irpi_xxx -> "index-of-receiver-pair-improved_branch"
		irpi_both = np.intersect1d(irpi_pos,irpi_neg)

		try:
			really_good_p = np.where(abs(kcao_flit_indmis_p[1])<good_thresh/8)[0]
		except IndexError:
			really_good_p = np.array([])
		try:
			really_good_n = np.where(abs(kcao_flit_indmis_n[1])<good_thresh/8)[0]
		except IndexError:
			really_good_n = np.array([])
		really_good = np.intersect1d(really_good_p,really_good_n)

		if nrecs<30:
			#spl = np.where(kcao_flit_indmis_n[0]<-0.7)[0]
			print "List of GOOD fits after inversion (regardless of good or bad BEFORE):"
			print "RP (kcao)\t\tRP(actual)\t\tInterstation distance (km)"
			for k in really_good:
				print rp_kcao[k], "\t\t", rp_orig[k], "\t\t", rp_dist[k]
			#print "List of fits IMPROVED by inversion (both branch fits were poor BEFORE inversion):"
			#print "RP (kcao)\tRP(actual)"
			#for k in irpi_both:
			#	print rp_kcao[k], "\t\t", rp_orig[k]
			#print "SPECIAL: "

		#************************* Compute the models from the stored model coefficients ******************************

		# NB: we store only the model coefficients. These need to be applied to the model basis-set to get the model.

		xpoints = kcao_gx[0,:]
		ypoints = kcao_gy[:,0]

		alltheta_deg=np.arange(0,360,10)
		alltheta=alltheta_deg*np.pi/180

		numit = len(kcao_allit_mc)

		basis = np.zeros((alltheta.size,num_points,num_points))
		kcao_sditer = np.zeros((numit,num_points,num_points))

		for k,theta in enumerate(alltheta):
			basis[k,:,:]=u1.somod.ringg(num_points,kcao_dx,xpoints,ypoints,theta,kcao_rr,kcao_rgw)
			if (not reald) and (len(trumod.shape)==1):
				kcao_sdtrue += mc_true[k]*basis[k,:,:]

		for i in range(numit):
			mc_iter = kcao_allit_mc[i]
			for k,theta in enumerate(alltheta):
				kcao_sditer[i,:,:] += mc_iter[k]*basis[k,:,:]

		kcao_sdstart = kcao_sditer[0,:,:]
		kcao_sdinv = kcao_sditer[-1,:,:]

		#************************ Determine range of values for appropriate colour scales *****************************

		if not reald:
			mod_min=min(np.amin(kcao_sdtrue),np.amin(kcao_sdstart),np.amin(kcao_sdinv))
			mod_max=max(np.amax(kcao_sdtrue),np.amax(kcao_sdstart),np.amax(kcao_sdinv))
		else:
			mod_min=min(np.amin(kcao_sdstart),np.amin(kcao_sdinv))
			mod_max=max(np.amax(kcao_sdstart),np.amax(kcao_sdinv))

		#************************************* Make the plots you want ************************************************
		
		plot_models()
		plot_kernels()
		#plot_inversion_progress(binsize)

	plt.show()
