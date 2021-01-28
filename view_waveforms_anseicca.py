#!/usr/bin/python

# General purpose modules
import os
import sys
import gzip
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

#####################################################################################################################################

def plot_waveforms_flit(a,b):
	fig = plt.figure(figsize=(14,2))
	axf, axl = fig.subplots(1,2,sharey=True)
	axes=[axf,axl]
	ptitle={0: "Before", 1: "After"}
	for i,ax in enumerate(axes):
		z=0 if i==0 else -1
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.yaxis.set_ticks([])
		ax.plot(kcao_t,kcao_obscross[:,a,b],label='Observation')
		ax.plot(kcao_t,kcao_flit_syncross[z][:,a,b],label='Synthetic')
		try:
			ax.axvline(x=kcao_t[kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
		except IndexError:
			# this happens when the window is the entire branch (usually with synthetic inversions)
			print("No lines for window..")
			pass
		ax.set_title(ptitle[i])
		if z!=0:
			plt.legend()

def plot_envelopes_flit(a,b):
	fig = plt.figure(figsize=(14,2))
	axf, axl = fig.subplots(1,2,sharey=True)
	axes=[axf,axl]
	ptitle={0: "Before", 1: "After"}
	for i,ax in enumerate(axes):
		z=0 if i==0 else -1
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.yaxis.set_ticks([])
		ax.plot(kcao_t,kcao_obsenv[:,a,b],label='Observation')
		ax.plot(kcao_t,kcao_flit_synenv[z][:,a,b],label='Synthetic')
		try:
			ax.axvline(x=kcao_t[kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
			ax.axvline(x=kcao_t[kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
		except IndexError:
			# this happens when the window is the entire branch (usually with synthetic inversions)
			pass
		#ax.set_title(ptitle[i])
		if z!=0:
			plt.legend()

def plot_waveforms_oneit(a,b,z):
	fig=plt.figure(figsize=(7.5,2.5))
	ax=fig.add_subplot(111)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.tick_params(axis='both', labelsize=14)
	ax.yaxis.set_ticks([])
	ax.plot(kcao_t,kcao_obscross[:,a,b],label='Observation')
	ax.plot(kcao_t,kcao_flit_syncross[z][:,a,b],label='Synthetic')
	try:
		ax.axvline(x=kcao_t[kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
	except IndexError:
		# this happens when the window is the entire branch (usually with synthetic inversions)
		pass
	if z!=0:
		plt.legend()

def plot_envelopes_oneit(a,b,z):
	fig=plt.figure(figsize=(7,2))
	ax=fig.add_subplot(111)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.tick_params(axis='both', labelsize=14)
	ax.yaxis.set_ticks([])
	ax.plot(kcao_t,kcao_obsenv[:,a,b],label='Observation')
	ax.plot(kcao_t,kcao_flit_synenv[z][:,a,b],label='Synthetic')
	try:
		ax.axvline(x=kcao_t[kcao_posr[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_negl[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_negr[a,b]],ls="--",color='k',alpha=0.3)
		ax.axvline(x=kcao_t[kcao_posl[a,b]],ls="--",color='k',alpha=0.3)
	except IndexError:
		# this happens when the window is the entire branch (usually with synthetic inversions)
		pass
	if z!=0:
		plt.legend()

def deltat_hist():

	# determine appropriate histogram bins such that "0" is a bin center
	mv1=max(max(deltat_pos_before),max(deltat_neg_before))
	mv2=min(min(deltat_pos_before),min(deltat_neg_before))
	maxval=np.round(max(abs(mv1),abs(mv2)))
	print("Max val for histograms is: ", maxval)
	hbe_p=np.arange(bs/2,maxval+bs,bs)
	hbe_n=-1*hbe_p[::-1]
	hbe = np.hstack((hbe_n,hbe_p))
	# hbe -> histogram_bin_edges

	print("Measurements in central histogram bins (P+N) after inversion: %d, %.2f per cent" %(cenbin_tot,100*cenbin_frac))
	ntot="$N_{tot}$\n= %d" %(npairs)
	fig = plt.figure()
	try:
		axh_p, axh_n = fig.subplots(1,2,sharex=True,sharey=True)
	except AttributeError:
		fig, axh = plt.subplots(1,2,sharex=True,sharey=True)
		axh_p = axh[0]
		axh_n = axh[1]

	ncen="$N_{cen}$\n= %d" %(cenbin_p.size)
	axh_p.hist(deltat_pos_before,bins=hbe,edgecolor='black')
	axh_p.hist(deltat_pos_after,bins=hbe,histtype='step',linewidth='1.5')
	axh_p.set_xlabel(r'$\Delta t$')
	axh_p.set_ylabel("No. of pairs")#, fontsize=14)
	axh_p.set_title("Positive branch")#, fontsize=18)
	axh_p.text(0.7, 0.8, ntot, transform=axh_p.transAxes)
	axh_p.text(0.7, 0.7, ncen, transform=axh_p.transAxes)

	ncen="$N_{cen}$\n= %d" %(cenbin_n.size)
	axh_n.hist(deltat_neg_before,bins=hbe,edgecolor='black')
	axh_n.hist(deltat_neg_after,bins=hbe,histtype='step',linewidth='1.5')
	axh_n.set_xlabel(r'$\Delta t$')#, fontsize=18)
	#axh_n.set_ylabel("No. of pairs")#, fontsize=14)
	axh_n.set_title("Negative branch")#, fontsize=18)
	#axh_n.tick_params(axis='x')#, labelsize=14)
	axh_n.text(0.1, 0.8, ntot, transform=axh_n.transAxes)
	axh_n.text(0.1, 0.7, ncen, transform=axh_n.transAxes)

################################################## Main program ###########################################################

inarg=sys.argv[1]
if os.path.isdir(inarg):
	filelist=[os.path.join(inarg,n) for n in os.listdir(inarg) if n.endswith('.npz')]
	nrecs_files=np.zeros(len(filelist))
	wspeed_files=np.zeros(len(filelist))
	pcentgood_files=np.zeros(len(filelist))
elif os.path.isfile(inarg):
	filelist=[inarg]

#********************************* Read the binary archive(s) **************************************************
for p,pfile in enumerate(filelist):
	barch=pfile #sys.argv[1]
	print("Reading ", barch)
	loaded = np.load(barch)
	# print(loaded.files)
	kcao_t=loaded['t']
	try:
		kcao_c=loaded['wsp']
	except KeyError:
		# THIS IS TEMPORARY: FOR OLD ARCHIVE FILES WHEN I DID NOT STORE THE WAVESPEED.
		kcao_c=3.0
	kcao_obscross=loaded['wobs']
	kcao_syncross_init=loaded['wsyn_i']
	kcao_syncross_final=loaded['wsyn_f']
	kcao_win_ind=loaded['win_ind']

	if (kcao_obscross.shape[1] != kcao_obscross.shape[2]) or (kcao_obscross.shape != kcao_syncross_init.shape) or (kcao_obscross.shape != kcao_syncross_final.shape):
		sys.exit("Problem with stored matrices for obs and syn waveforms")
	elif len(filelist)==1:
		print("Number of receivers is: ", kcao_obscross.shape[1])
		print("Wavespeed used is: ", kcao_c)

	kcao_negl = kcao_win_ind[0,:,:]
	kcao_negr = kcao_win_ind[1,:,:]
	kcao_posl = kcao_win_ind[2,:,:]
	kcao_posr = kcao_win_ind[3,:,:]

	# compute waveform envelopes
	kcao_obsenv=np.abs(ss.hilbert(kcao_obscross, axis=0))
	kcao_synenv_init=np.abs(ss.hilbert(kcao_syncross_init, axis=0))
	kcao_synenv_final=np.abs(ss.hilbert(kcao_syncross_final, axis=0))

	# get the traveltime discrepancies
	dt = np.round(kcao_t[1] - kcao_t[0],1)
	nrecs=kcao_obscross.shape[1]
	npairs=int(nrecs*(nrecs-1)/2)
	deltat_pos_before=np.zeros(npairs)
	deltat_neg_before=np.zeros(npairs)
	deltat_pos_after=np.zeros(npairs)
	deltat_neg_after=np.zeros(npairs)

	cp=0 # cp stands for count_pair
	for j in range(nrecs-1):
		for i in range(j+1,nrecs):
			# do positive branch
			ind_peak_obs = np.argmax(kcao_obsenv[kcao_posl[i,j]:kcao_posr[i,j],i,j])
			ind_peak_isyn = np.argmax(kcao_synenv_init[kcao_posl[i,j]:kcao_posr[i,j],i,j])
			ind_peak_fsyn = np.argmax(kcao_synenv_final[kcao_posl[i,j]:kcao_posr[i,j],i,j])
			deltat_pos_before[cp] = (ind_peak_obs - ind_peak_isyn)*dt
			deltat_pos_after[cp] = (ind_peak_obs - ind_peak_fsyn)*dt
			# do negative branch
			ind_peak_obs = np.argmax(kcao_obsenv[kcao_negl[i,j]:kcao_negr[i,j],i,j])
			ind_peak_isyn = np.argmax(kcao_synenv_init[kcao_negl[i,j]:kcao_negr[i,j],i,j])
			ind_peak_fsyn = np.argmax(kcao_synenv_final[kcao_negl[i,j]:kcao_negr[i,j],i,j])
			deltat_neg_before[cp] = (ind_peak_obs - ind_peak_isyn)*dt
			deltat_neg_after[cp] = (ind_peak_obs - ind_peak_fsyn)*dt

			cp+=1

	bs=1.0 #binsize
	good_thresh = bs/2 # DO NOT CHANGE THIS. CHANGE ONLY THE BINSIZE

	cenbin_p=np.where(abs(deltat_pos_after)<good_thresh)[0]
	cenbin_n=np.where(abs(deltat_neg_after)<good_thresh)[0]
	cenbin_tot = cenbin_p.size + cenbin_n.size
	cenbin_frac = float(cenbin_tot)/(2*npairs)

	if len(filelist)>1:
		pcentgood_files[p]=cenbin_frac
		wspeed_files[p]=kcao_c
		nrecs_files[p]=kcao_obscross.shape[1]

if len(filelist)>1:
	if len(np.unique(nrecs_files))>1:
		sys.exit("The different pickles have different number of receivers - script terminated.")

	sortind = np.argsort(wspeed_files)
	sorted_ws = wspeed_files[sortind]
	sorted_pcent = pcentgood_files[sortind]

	fig_ws=plt.figure()
	axws=fig_ws.add_subplot(111)
	axws.plot(sorted_ws,sorted_pcent,'-o')
	axws.set_xlabel("Wavespeed [km/s]")
	axws.set_ylabel(r"Good $\Delta t$ (%)")

else:
	#*************** generate lists containing synthetics for initial and final iterations ***************
	kcao_flit_syncross=[kcao_syncross_init,kcao_syncross_final]
	kcao_flit_synenv=[kcao_synenv_init,kcao_synenv_final]
	#*****************************************************************************************************

	deltat_hist()
plt.show()
