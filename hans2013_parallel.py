#!/usr/bin/python

# General purpose modules
import sys
import numpy as np
#import scipy.signal as ss
import scipy.stats as sst
import scipy.special as ssp
import scipy.optimize as sop
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Modules written by me
import anseicca_utils1 as u1

##########################################################################################################################

class inv_cc_amp:

	def __init__(self,comm,rank,hlboxo,ngpib,boxes_ratio,nrecs,rlocsx,rlocsy,cwav,signal,ring_rad,ring_w,doinv,only1_iter,dobs=None,dobs_info=None):

		""" 
		hlboxo (type 'float'): half-length of outer box
	 	ngpib (type 'int'): number of grid points in inner box
		nrecs (type 'int'): number of receivers or stations
		rlocsx (type 'type 'numpy.ndarray'): x-coordinates of all receivers (in grid-point units)
		rlocsy (type 'type 'numpy.ndarray'): y-coordinates of all receivers (in grid-point units)
		cwav (type 'float'): uniform wavespeed in model
		signal (type 'instance'): object of class "SignalParameters" containing various signal characteristics of the data
		dobs (optional, type 'numpy.ndarray'): the data (REAL DATA ONLY)
		dobs_info (optional, type 'tuple'): Tuple containing the S/N ratio and actual (non-gridded) receiver locations (REAL DATA ONLY)
		"""

		###################################### Part 1: preliminaries #############################################

		self.hlbox_outer = hlboxo
		self.ngpib = ngpib
		self.nrecs = nrecs
		self.c = cwav
		self.ring_rad = ring_rad
		self.wgauss = ring_w

		self.ntot=2*self.ngpib-1
		self.omost_fac = boxes_ratio # this integer factor must be >=2; if = 2, standard case

		self.nom = signal.nsam
		self.deltat = signal.dt
		f0 = signal.cf
		fl = signal.lf
		fh = signal.hf
		altuk = signal.altukey
		
		if not (dobs is None):
		# real data case
			self.reald=True
		else:
		# synthetic data case
			self.reald=False

		tdur=self.nom*self.deltat
		# inverse frequency spacing for DFT of time series

		self.clicksx = -1*rlocsx
		self.clicksy = -1*rlocsy
		# NB: rlocsx, rlocsy are the ACTUAL receiver locations. These are transformed to clicksx and clicksy
		# for the purpose of this class. The negative is because of the coordinate transformation between
		# "r" and "r - r_alpha" which is used in the functions compute_cc_lowerhalf and diffkernel.

		npairs=self.nrecs*(self.nrecs-1)/2
		alltheta_deg=np.arange(0,360,10)
		self.num_mparams=alltheta_deg.size
				
		if rank==0:
		# On master node
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! setting up !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			def one_by_r(x,k):
				return k*1./x

			tds=True
			iterate_master=doinv
			forced=False

			syncross_brec = np.empty((self.nom, self.nrecs), dtype='float')
			if self.reald:
			# REAL DATA CASE
				self.obscross = dobs
				obscross_aspec = np.abs(np.fft.fft(dobs,axis=0))
			else:
			# SYNTHETIC DATA CASE
				obscross_brec = np.empty((self.nom, self.nrecs), dtype='float')

			self.dvar_pos = np.ones(npairs)
			self.dvar_neg = np.ones(npairs)
			
			fhz=np.fft.fftfreq(self.nom,self.deltat)
			omega = 2*np.pi*fhz
			dom = omega[1] - omega[0] 
			# dom -> d_omega
			nom_nneg = len(fhz[fhz>=0]) #self.nom/2+1
			# number of non-negative frequency samples
			# remember, when nom is even, nom_nneg is smaller by one sample: the positive Nyquist is missing.

			if self.nom%2 != 0:
				self.t = np.arange(-(nom_nneg-1),nom_nneg)*self.deltat 
				# time series corresponding to cross-correlation lags; only required on rank 0
				# NB: the crucial advantage of building a signal as above, rather than doing np.arange(tstart,tend,deltat),
				# is that the above formulation ensures that you always get the time sample zero, regardless of deltat.
			else:
				self.t = np.arange(-nom_nneg,nom_nneg)*self.deltat
			if len(self.t) != self.nom:
				sys.exit("Quitting. Length of time array does not match number of samples. Check signal parameters.")

			#************************************** build the source characteristics ******************************************
			if not self.reald:
			# synthetic data case
				if f0==2:
					a=0.3
				elif f0==0.3:
					a=16
				elif f0==0.1:
					a=64
				elif f0==0.05:
					a=640
				# dependence of parameter "a" -- which controls rate of exponential damping and hence shape of stf -- on peak
				# frequency is implemented in an adhoc fashion for the peak frequencies of interest when using this code. The
				# criterion behind the estimated values is to obtain a meaningful power spectrum -- one with zero DC power.
				# (In the time domain this corresponds to retaining a few cycles (~ 2-3) of the cosine wave before it is damped to 0.)

				#sourcetime = np.exp(-self.t**2/(512*(0.05**2))) * np.cos(4*np.pi*self.t) # matlab code original
				sourcetime = np.exp(-self.t**2/a) * np.cos(2*np.pi*f0*self.t)
				#self.pss = np.abs(np.fft.fft(sourcetime)*self.deltat)**2
				# ARJUN: no need for multiplication by self.deltat in above
				self.pss = np.abs(np.fft.fft(sourcetime))**2
				# pss stands for power_spectrum_of_sources
			else:
			# real data case
				fhzp=fhz[fhz>=0]
				fhzn=fhz[fhz<0]
				# taking zero on the positive side ensures that both branches are of equal size, because remember that for
				# even number of samples, the positive side is missing the Nyquist term.
	
				#rvp=sst.skewnorm(a=-5,loc=0.55,scale=0.15)
				#rvn=sst.skewnorm(a=5,loc=-0.55,scale=0.15)

				rvp=sst.skewnorm(a=-3,loc=0.5,scale=0.13)
				rvn=sst.skewnorm(a=3,loc=-0.5,scale=0.13)

				self.pss = np.concatenate((rvp.pdf(fhzp),rvn.pdf(fhzn)))

			#************************************* end of source characteristics *******************************************s

			dx = self.hlbox_outer/(self.ngpib-1.0)
			nstart = self.ngpib/2 + 1
			nmid = self.ngpib # mid point

			hlbox_omost = self.omost_fac*self.hlbox_outer/2.0
			ntot_omost = self.omost_fac*(self.ngpib-1) + 1
			nstart_omost = (self.omost_fac - 2)*(nstart-1)

			# think of the "nstart" variables as the number of grid points between an outer box and the next inner box

			# grid points of outer-most box
			x3=np.linspace(-hlbox_omost,hlbox_omost,ntot_omost)
			y3=np.linspace(-hlbox_omost,hlbox_omost,ntot_omost)

			# grid points of outer box
			x2=np.linspace(-self.hlbox_outer,self.hlbox_outer,self.ntot)
			y2=np.linspace(-self.hlbox_outer,self.hlbox_outer,self.ntot)

			# grid points of inner box
			x = x2[nstart:nstart+self.ngpib]
			y = np.copy(x)

			self.dist_rp=np.zeros((self.nrecs,self.nrecs))
			# array of receiver pair distances; only required on rank 0
			for j in range(self.nrecs):
				for i in range(self.nrecs):
					self.dist_rp[i,j] = np.sqrt( (x2[self.clicksx[i]+nmid]-x2[self.clicksx[j]+nmid])**2 + (y2[self.clicksy[i]+nmid]-y2[self.clicksy[j]+nmid])**2 )
			
			dist_all = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]
			alldist = dist_all[np.argsort(dist_all)]
			# a sorted 1-D array of receiver-pair distances

			# generate the grid for plotting		
			self.gx, self.gy = np.meshgrid(x,y)
			if (self.omost_fac>2) and  (__name__ == '__main__'):
				self.gx2, self.gy2 = np.meshgrid(x2,y2)

			print "Computing distances from origin.."
			r = np.zeros((ntot_omost,ntot_omost))
			for j in range(r.shape[1]):
				r[:,j] = np.sqrt(x3[j]**2 + y3**2)
				# Note the different orientation of the matrix vis-a-vis MATLAB

			r+=0.000001
			# this is done so that r does not contain any zeros; to prevent the Hankel function
			# from blowing up at the origin (r=0)
			
			#rcent = r[self.nstart:(self.nstart + self.ngpib - 1), self.nstart:(self.nstart + self.ngpib-1)]

			self.besselmat = np.zeros((ntot_omost,ntot_omost,nom_nneg), dtype='complex')

			print "Computing Hankel functions.."
			for i in range(1,nom_nneg):
				self.besselmat[:,:,i] = ssp.hankel1(0,omega[i]*r/self.c) * 1j * 0.25

			#******************************* source distributions and observation errors ******************************************
			def mult_gauss(j,mag,xp,yp,xw,yw):

				# xp -> xpos (in dx units)
				# xw -> xwidth (in dx units)
				if self.omost_fac>2:
					usex=x2
					usey=y2
				else:
					usex=x
					usey=y

				ans = mag * np.exp( -( (usex[j] - xp*self.dx)**2/(xw*(dx**2)) + (usey - yp*dx)**2/(yw*(dx**2)) ) )
				#ans = mag * np.exp( -( (x[j] - xp*dx)**2/(xw*(dx**2)) + (y - yp*dx)**2/(yw*(dx**2)) ) )
				return ans

			sdist_type = {0: mult_gauss, 1: u1.somod.ringg, 2: u1.somod.rgring}
			alltheta=alltheta_deg*np.pi/180
			self.mc_start = np.ones(alltheta.size)
			# mc -> model_coefficients
			self.basis = np.zeros((self.num_mparams,self.ngpib,self.ngpib))
			if self.omost_fac>2:
				basis_true = np.zeros((alltheta.size,self.ntot,self.ntot))

			mag1=1
			self.mc_start *= mag1

			if not self.reald:
			# SYNTHETIC DATA CASE
				self.mc_true = np.ones(alltheta.size)
				if self.omost_fac>2:
					self.distribs_true = np.zeros((self.ntot,self.ntot), dtype='float')
					# used to generate the synthetic "data" in the absence of real data
					# hence used for testing inversions; this source distribution is NOT involved in computing source kernels
				else:
					self.distribs_true = np.zeros((self.ngpib,self.ngpib), dtype='float')

				self.distribs_start = np.zeros((self.ngpib,self.ngpib), dtype='float')
				# used to generate the synthetics for inversion; this source distribution is involved in computing source kernels

				self.mc_true *= mag1
				nperts=3
				# nperts -> the number of "regions" in the True model which are perturbed
				t1=[130,220,345]
				t2=[150,240,15]
				mag2=[8,5,5]
				if (len(t1)<nperts) or (len(t2)<nperts) or (len(mag2)<nperts):
					raise SystemExit("Problem building the True model. Please check.")

				#for col in range(self.distribs_true.shape[1]):
				#	self.distribs_true[:,col] += sdist_type[0](col,10*mag1,0,-140,2000,75) + sdist_type[0](col,10*mag1,-160,120,75,2000)
				#	#self.distribs_true[:,col] += sdist_type[0](col,5*mag1,80,0,100,100) + sdist_type[0](col,8*mag1,-160,-160,100,100) \
								#+ sdist_type[0](col,2*mag1,-100,100,100,100)

				for p in range(nperts):
					s1=np.argwhere(alltheta_deg >= t1[p])
					s2=np.argwhere(alltheta_deg <= t2[p])
					relind=np.intersect1d(s1,s2)
					if len(relind)==0:
						relind=np.union1d(s1,s2)
					self.mc_true[relind]=mag2[p]+mag1

				for k,theta in enumerate(alltheta):
					self.basis[k,:,:] = sdist_type[1](self.ngpib,dx,x,y,theta,self.ring_rad,self.wgauss)
					self.distribs_start += self.mc_start[k]*self.basis[k,:,:]
					if self.omost_fac>2:
						basis_true[k,:,:] = sdist_type[1](self.ntot,dx,x2,y2,theta,2*self.ring_rad,self.wgauss)
						self.distribs_true += self.mc_true[k]*basis_true[k,:,:]
					else:
						self.distribs_true += self.mc_true[k]*self.basis[k,:,:]
				
				# no observation errors in the synthetic case
				# nothing to do on this front.
			else:
			# REAL DATA CASE
				self.distribs_start = np.zeros((self.ngpib,self.ngpib))
				for k,theta in enumerate(alltheta):
					self.basis[k,:,:] = sdist_type[1](self.ngpib,dx,x,y,theta,self.ring_rad,self.wgauss)
					self.distribs_start += self.mc_start[k]*self.basis[k,:,:]
			

				#************************* Initial amplitudes and observation errors ******************************

				#********* Initial amplitudes (amplitudes of starting synthetics)
				
				occegy_funcf = np.square(obscross_aspec)
				# occegy_funcf -> observed_cc_power_as_a_function_of_frequency
				occ_egy = np.sum(occegy_funcf,axis=0)/self.nom
				# occ_egy -> observed_cc_energy
				egy_obs = occ_egy[np.nonzero(np.tril(self.dist_rp))]
				# this is because we are only interested in the lower triangular part of the matrix
				egy_obs = egy_obs[np.argsort(dist_all)]

				nf_dist = 0.5*self.c/fh
				# nf_dist -> near_field_distance. Using a very crude estimate: half the shortest wavelength in the data 
				sd_ind=np.argwhere(alldist<nf_dist)
				# sd_ind -> short_distance_indices
				sig_dummy = np.ones(alldist.size)
				sig_dummy[sd_ind] = 5
				# NB: sig_dummy - deliberately called "dummy" - contains basically the relative weights for the data points, NOT
				# the actual standard deviations. This is reflected in the argument "absolute_sigma=False" to scipy's curve fit.
							
				popt, pcov = sop.curve_fit(one_by_r,alldist,egy_obs,sigma=sig_dummy,absolute_sigma=False)
				oef = popt[0]/alldist
				#oef -> observed_energy_fitted

				#********* Errors Part 1: error due to SNR
				snr = dobs_info[1]
				
				self.esnrpd_ltpb = np.zeros((self.dist_rp.shape))
				# esnrpd -> error(due to)_SNR_(as a)_percentage_(of)_data
				# ltpb -> lower_triangle_positive_branch
				# (it is implied that the upper triangle of the matrix is for the negative branch)

				self.esnrpd_ltpb[np.where(snr<2)]=0.8
				self.esnrpd_ltpb[np.where((snr>2) & (snr<3))]=0.5
				self.esnrpd_ltpb[np.where(snr>3)]=0.1

				#********* Errors Part 2: error due to energy decay with distance

				delA = dobs_info[2]			

				#********************************************************************************************************************
				# NB: uncertainties in the observations contained in dobs_info need to be corrected, because the measurement for
				# the kernels involves cc energies computed in a certain window only, whereas the curve fitting above is done using
				# the energy of the entire cc branch. This correction can be made using the waveform's S/N ratio, which indirectly
				# provides a measure of the contribution of the window of interest, to the total energy of the waveform (branch).
				#********************************************************************************************************************

				# refine the error so it applies to the measurement window only
				nsr = 1./snr
				ScT = 1./(1+nsr) # 1./np.sqrt(1+nsr)
				# ScT -> signal_contribution_to_total (energy)
				delA *= ScT

				self.dvar_egy_ltpb = np.square(delA)

#				#********* Errors Part 3: from the position error due to relocation of receivers to grid points
#				origdist_rp = dobs_info[0]
#				deltapos = np.square(origdist_rp - self.dist_rp)

				#************************************* End of observation errors ******************************************

			print "Completed initial setup..."
			self.distribs_inv=np.copy(self.distribs_start)
			self.allit_mc = []
			self.allit_misfit = []
			self.flit_indmis_p = []
			self.flit_indmis_n = []
			self.flit_syncross = []
			# variables with names ending in "_inv"  contain values for current (ulimately last) iteration only
			# variables with names starting with "allit_" are lists where each element corresponds to an iteration of the inversion.
			# variables with names starting with "flit_" are two-element lists, storing first (f) and last (l) iteration values only, of certain quantities.
		
			self.allit_mc.append(np.copy(self.mc_start))		
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! End of setup !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		else:
		# On non-master nodes
			dx=None
			fhz=None
			dom=None
			tds=None
			nstart=None
			nstart_omost=None
			nom_nneg=None
			syncross_brec=None
			obscross_brec=None
			iterate_master=None
			self.pss = np.empty(self.nom, dtype='float')
			if self.nom%2!=0:
				self.besselmat = np.empty((self.omost_fac*(self.ngpib-1) + 1,self.omost_fac*(self.ngpib-1) + 1,self.nom/2+1), dtype='complex')
			else:
			# GOT TO BE CAREFUL: in case of even number of samples the positive Nyquist term is missing
				self.besselmat = np.empty((self.omost_fac*(self.ngpib-1) + 1,self.omost_fac*(self.ngpib-1) + 1,self.nom/2), dtype='complex')
			self.distribs_inv = np.empty((self.ngpib,self.ngpib), dtype='float')
			self.basis = np.empty((self.num_mparams,self.ngpib,self.ngpib))
			if self.omost_fac>2:
				self.distribs_true = np.empty((2*self.ngpib-1,2*self.ngpib-1), dtype='float')
			else:
				self.distribs_true = np.empty((self.ngpib,self.ngpib), dtype='float')
			
		######################################## End of part 1: preliminaries #####################################################
			
		# MPI-distribute the required variables
		self.dx = comm.bcast(dx,root=0)
		self.fhz = comm.bcast(fhz,root=0)
		self.dom = comm.bcast(dom,root=0)
		self.brec = comm.scatter(range(self.nrecs),root=0)
		self.nstart = comm.bcast(nstart,root=0)
		self.nstart_omost = comm.bcast(nstart_omost,root=0)
		self.nom_nneg = comm.bcast(nom_nneg,root=0)
		todo_syn = comm.bcast(tds,root=0)
		iterate = comm.bcast(iterate_master,root=0)
		comm.Bcast(self.basis, root=0)
		comm.Bcast(self.besselmat, root=0)
		comm.Bcast(self.distribs_inv, root=0)
		if not self.reald:
			comm.Bcast(self.distribs_true, root=0) 
			#print "distribs_true ok.."
		
		################################## Part 2: compute cross-correlations and measurements ######################################

		itnum=0
		
		while iterate:
			self.iter = itnum

			while todo_syn:
				comm.Bcast(self.pss, root=0)
				syncross_brec, obscross_brec = self.compute_cc_lowerhalf(rank)
				#comm.Gather(syncross_brec, self.syncross, root=0)
				syncross_all = np.asarray(comm.gather(syncross_brec, root=0))
				if not self.reald:
				# SYNTHETIC DATA CASE
					#comm.Gather(obscross_brec, self.obscross, root=0)
					obscross_all = np.asarray(comm.gather(obscross_brec, root=0))
			
				if rank==0:
					self.syncross = np.transpose(syncross_all, [1,2,0])
					# NB: need to go through this hassle of "asarray" and "transpose" only
					# because comm.Gather (for numppy arrays) is not working
					syncross_aspec=np.abs(np.fft.fft(self.syncross,axis=0))
					if self.iter==0:
					# First iteration
						sccegy_funcf = np.square(syncross_aspec)
						scc_egy = np.sum(sccegy_funcf,axis=0)/self.nom
						egy_syn = scc_egy[np.nonzero(np.tril(self.dist_rp))]
						egy_syn = egy_syn[np.argsort(dist_all)]

						try:
							popt, pcov = sop.curve_fit(one_by_r,alldist,egy_syn,sigma=sig_dummy,absolute_sigma=False)
						except UnboundLocalError:
						# variable sig_dummy does not exist in case of synthetic data
							popt, pcov = sop.curve_fit(one_by_r,alldist,egy_syn)

						sef = popt[0]/alldist
						#sef -> synthetic_energy_fitted
						
						if self.reald:
						# REAL DATA CASE
							esf = np.mean(oef/sef)
							# esf -> energy_scale_factor
							if esf > 0.9 and esf < 1.1:
								tds=False
							else:
								print "esf is %f, MULTIPLYING self.pss by %f" %(esf,np.sqrt(esf))
								self.pss *= np.sqrt(esf)

						else:
						# SYNTHETIC DATA CASE
							self.obscross = np.transpose(obscross_all, [1,2,0])
							tds=False
							# we're done, synthetics and "data" are both ready
					else:
					# Subsequent iterations
						# no need for energy fitting whether real or synthetic data; proceed with algorithm
						tds=False

				todo_syn = comm.bcast(tds,root=0)

			#************************************** End of while loop for energy fitting *********************************************

			if rank==0:
				#********* run computationally cheap functions on master ***********
				self.get_cc_upperhalf()
				self.make_measurement()
				#****** Finished execution of computationally cheap functions ******

				# inversion related variables
				self.Gmat_pos=np.zeros((npairs,self.num_mparams))
				self.Gmat_neg=np.zeros((npairs,self.num_mparams))

				self.deltad_pos=np.zeros(npairs)
				self.deltad_neg=np.zeros(npairs)
				
				self.ngrad_pos=np.empty(self.num_mparams)
				self.ngrad_neg=np.empty(self.num_mparams)

				mfit_kern_pos = np.zeros((self.ngpib, self.ngpib))
				mfit_kern_neg = np.zeros((self.ngpib, self.ngpib))
				# mfit_kern -> misfit_kernel

				print "Starting computation of source kernels for each receiver pair..."
			else:
				self.weightpos=np.empty((self.nom, self.nrecs, self.nrecs))
				self.weightneg=np.empty((self.nom, self.nrecs, self.nrecs))
				self.synamp_pos=np.empty((self.nrecs, self.nrecs))
				self.synamp_neg=np.empty((self.nrecs, self.nrecs))
				self.obsamp_pos=np.empty((self.nrecs, self.nrecs))
				self.obsamp_neg=np.empty((self.nrecs, self.nrecs))
				mfit_kern_pos = None
				mfit_kern_neg = None

			################################## End of Part 2: cross-correlations and measurements ####################################

			# variables common to all processors		
			ekern_pos = np.zeros((self.ngpib, self.ngpib))
			ekern_neg = np.zeros((self.ngpib, self.ngpib))
			Gproc_pos=np.zeros((self.nrecs-self.brec-1,self.num_mparams))
			Gproc_neg=np.zeros((self.nrecs-self.brec-1,self.num_mparams))
			ddproc_pos=np.zeros(self.nrecs-self.brec-1)
			ddproc_neg=np.zeros(self.nrecs-self.brec-1)

			# MPI-distribute the variables required for the rest of the program
			comm.Bcast(self.weightpos, root=0)
			comm.Bcast(self.weightneg, root=0)
			comm.Bcast(self.synamp_pos, root=0)
			comm.Bcast(self.synamp_neg, root=0)
			comm.Bcast(self.obsamp_pos, root=0)
			comm.Bcast(self.obsamp_neg, root=0)
			
			#################################### Part 3: compute kernels and do inversion ###########################################

			if self.brec+1 < self.nrecs:
				for cp,i in enumerate(range(self.brec+1,self.nrecs)):
					print "...(source kernel) for receivers %d-%d on processor %d " %(i,self.brec,rank)
					sker_p, sker_n = self.diffkernel(i,self.brec)
					# Computing individual source kernels (eq. 15)
					
					# build (partially) the G-matrix
					kb_prod = sker_p*self.basis
					Gproc_pos[cp,:] = np.sum(kb_prod, axis=(1,2)) * self.dx**2
					kb_prod = sker_n*self.basis
					Gproc_neg[cp,:] = np.sum(kb_prod, axis=(1,2)) * self.dx**2
		
					ddproc_pos[cp] = np.log(self.obsamp_pos[i,self.brec]/self.synamp_pos[i,self.brec])
					ddproc_neg[cp] = np.log(self.obsamp_neg[i,self.brec]/self.synamp_neg[i,self.brec])
					#print "obsamp_pos and synamp_pos: ", i, self.brec, self.obsamp_pos[i,self.brec], self.synamp_pos[i,self.brec]
					#print "obsamp_neg and synamp_neg: ", i, self.brec, self.obsamp_neg[i,self.brec], self.synamp_neg[i,self.brec]
					# Computing event kernels, i.e. eq. 30
					ekern_pos += sker_p * ddproc_pos[cp]
					ekern_neg += sker_n * ddproc_neg[cp]
			else:
				print "...(source kernel) nothing to do on processor %d" %(rank)

			#************ combine results from all processors
			comm.Reduce(ekern_pos, mfit_kern_pos, root=0)
			comm.Reduce(ekern_neg, mfit_kern_neg, root=0)
			list_Gp = comm.gather(Gproc_pos, root=0)
			list_Gn = comm.gather(Gproc_neg, root=0)
			list_ddp = comm.gather(ddproc_pos, root=0)
			list_ddn = comm.gather(ddproc_neg, root=0)

			if rank==0:
				#**************** inversion on master *****************

				Gmat = {'p': self.Gmat_pos, 'n': self.Gmat_neg}
				dd = {'p': self.deltad_pos, 'n': self.deltad_neg}
				list_Gmat = {'p': list_Gp, 'n': list_Gn}
				list_dd = {'p': list_ddp, 'n': list_ddn}
				
				# need to complete the G-matrix and deltad vector
				for br in Gmat:
				# br -> branch (positive or negative)
					startrow=0
					for litem in list_Gmat[br]:
						if litem.shape[0]>0:
							paths=litem.shape[0]
							Gmat[br][startrow:startrow+paths,:] = litem
							startrow += paths
					startrow=0
					for litem in list_dd[br]:
						if litem.shape[0]>0:
							paths=litem.shape[0]
							dd[br][startrow:startrow+paths] = litem
							startrow += paths

				#*********** things to do on first iteration
				if itnum==0:
					if self.reald:
					# complete the calculation of the data errors. NB: we consider two types of error.
					# The first one is independent of the measurements and is already computed.
					# The second is defined relative to the measurements, so we must get the absolute values here.

						dvar_snr_pos = np.square(self.esnrpd_ltpb * self.obsamp_pos)
						dvar_snr_neg = np.square(np.transpose(self.esnrpd_ltpb) * self.obsamp_neg)

						# combine different errors
						dvar_pos = dvar_snr_pos + self.dvar_egy_ltpb
						dvar_neg = dvar_snr_neg + np.transpose(self.dvar_egy_ltpb)

						# finally, convert data variance from matrix-form (2D) to vector-form (1D)
						dv_mat = {'p': dvar_pos, 'n': dvar_neg}
						dv_vec = {'p': self.dvar_pos, 'n': self.dvar_neg}
						for br in dv_mat:
							start=0
							for col in range(dv_mat[br].shape[1]):
								ntc = dv_mat[br].shape[0] - col - 1
								# ntc -> number(of pairs)_this_col
								dv_vec[br][start:start+ntc] = dv_mat[br][col+1:,col]
								start+=ntc

					# regardless of real or synthetic data, store the first-iteration values of certain quantities
					self.mfit_kern_pos = mfit_kern_pos
					self.mfit_kern_neg = mfit_kern_neg

				def record_flit():
					self.flit_indmis_p.append(self.deltad_pos)
					self.flit_indmis_n.append(self.deltad_neg)
					self.flit_syncross.append(self.syncross)

				# record inversion progress
				#wmp = self.deltad_pos * self.dvar_pos
				#wmn = self.deltad_neg * self.dvar_neg
				wmp = self.deltad_pos / np.sqrt(self.dvar_pos)
				wmn = self.deltad_neg / np.sqrt(self.dvar_neg)
				total_misfit = 0.5*(np.dot(wmp,wmp) + np.dot(wmn,wmn))
				if itnum==0:
					record_flit()
				self.allit_misfit.append(total_misfit)
	
				tds=True
				# because the iteration is over and we want to compute synthetics in the next iteration (if there is one)

				if itnum==1:
					record_flit()
					if only1_iter:
					# FORCED STOP FOR TESTING: last misfit stored will correspond to first updated model
						forced=True; iterate_master=False 

				if (itnum>0) and (not forced):
				# determine whether to terminate inversion or iterate further
					#if itnum>10:
					#	iterate_master=False
					mf_curr = self.allit_misfit[-1]
					mf_prev = self.allit_misfit[-2]
					pchange = 100*(mf_prev - mf_curr)/mf_prev
					if (pchange>0 and pchange<5) or itnum>15:
						iterate_master=False
						#inversion terminated.
						# store quantities corresponding to the final iteration model
						record_flit()

				if iterate_master:
					#********* do actual inversion (model update) ***********
					update_mod = self.inversion(mfit_kern_pos,mfit_kern_neg)
					self.distribs_inv += update_mod
					print "END OF ITERATION %d" %(itnum)

			# MPI-distribute the variables that (possibly) change through the iterations
			comm.Bcast(self.distribs_inv, root=0)
			iterate = comm.bcast(iterate_master,root=0)
			todo_syn = comm.bcast(tds,root=0)

			itnum +=1

		#*********************** End of loop over iterations *******************

	##################################### All parts completed, end of function init ########################################
	
	def inversion(self,mfk_pos,mfk_neg):

		""" Performs inversion using a standard Gauss-Newton iterative scheme """

		# NB: the data covariance matrix is assumed to be diagonal. Instead of storing and using the potentially HUGE
		# diagonal matrix, we work with just the vector of data variances.

		#**************************** fix the damping (model covarance matrix) *********************************
		self.gamma=0.1
		if not self.reald:
		# in case of synthetic data, we can get away with a diagonal model covariance matrix (covariances = 0)
			Dmat=np.identity(self.num_mparams)
			CmInv = (self.gamma**2)*Dmat
		else:
		# in case of real data, we use a banded model covariance matrix (non-zero covariances)
			Cm = np.zeros((self.num_mparams,self.num_mparams))
			cord = 3
			# cord -> correlation_distance
			for r in range(Cm.shape[0]):
				col = np.arange(float(Cm.shape[1]))
				Cm[r,:] = ((1./self.gamma)**2)*np.exp(-0.5*(((r-col)/cord)**2))
		
			CmInv = np.linalg.inv(Cm)
		#*************************************** End of damping ************************************************

		m_iter = self.allit_mc[-1]
		m_prior = self.mc_start

		G = {'p': self.Gmat_pos, 'n': self.Gmat_neg}
		dd = {'p': self.deltad_pos, 'n': self.deltad_neg}
		mfk = {'p': mfk_pos, 'n': mfk_neg}
		ng1 = {'p': self.ngrad_pos, 'n': self.ngrad_neg}
		#ng2 = {'p': self.ngrad2_pos, 'n': self.ngrad2_neg}
		dvi = {'p': 1./self.dvar_pos, 'n': 1./self.dvar_neg}

		mod_update = np.zeros((self.ngpib, self.ngpib))
		deltam = np.zeros((2,self.num_mparams))
			
		for b,br in enumerate(G):
		# br -> branch (positive or negative)

			# compute basic (unweighted, undamped) gradient
			kb_prod2 = mfk[br]*self.basis
			ng1[br][:] = np.sum(kb_prod2, axis=(1,2)) * self.dx**2

			# effect of weighting
			Gt_CdInv = (G[br].T)*dvi[br]
			ngrad = np.matmul(Gt_CdInv,dd[br])
			# effect of damping
			ngrad_use = ngrad - np.matmul(CmInv,(m_iter - m_prior))
			
			# Hessian with weighting and damping
			hess_apx = np.matmul(Gt_CdInv,G[br])
			hess_use = hess_apx + CmInv

			#********** solve the linear system for the model update
			deltam[b,:] = np.linalg.solve(hess_use,ngrad_use)
			
		# combine the results from the positive and negative branches
		deltam_use = np.mean(deltam,axis=0)
		#deltam_use = deltam[0,:]
		self.allit_mc.append(m_iter + deltam_use)
		mod_update = np.einsum('k,klm',deltam_use,self.basis)

		return mod_update

	######################################################################################################################

	def compute_cc_lowerhalf(self,rank):
		
		# this function is parallelized

		scc_brec = np.zeros((self.nom, self.nrecs), dtype='complex')
		
		if self.reald:
			occ_brec = None
		else:
			occ_brec = np.zeros((self.nom, self.nrecs), dtype='complex')
			
		if self.brec+1 < self.nrecs:
			print "Computing cross-correlations..."

			#nxst = np.array(map(lambda m: self.nstart + m -1 , self.clicksx))
			#nyst = np.array(map(lambda m: self.nstart + m -1 , self.clicksy))
			#nxfin = nxst + self.ngpib
			#nyfin = nyst + self.ngpib

			nxst = np.array(map(lambda m: self.nstart_omost + self.nstart + m -1 , self.clicksx))
			nyst = np.array(map(lambda m: self.nstart_omost + self.nstart + m -1 , self.clicksy))
			nxfin = nxst + self.ngpib
			nyfin = nyst + self.ngpib

			if self.omost_fac>2:
				nxst_om = np.array(map(lambda m: self.nstart_omost + m, self.clicksx))
				nyst_om = np.array(map(lambda m: self.nstart_omost + m, self.clicksy))
				nxfin_om = nxst_om + self.ntot
				nyfin_om = nyst_om + self.ntot
			elif self.omost_fac==2:
				nxst_om = nxst
				nyst_om = nyst
				nxfin_om = nxfin
				nyfin_om = nyfin

			# account for possible asymmetry in frequency samples (happens when self.nom is even)
			fhzp = len(self.fhz[self.fhz>0])
			fhzn = len(self.fhz[self.fhz<0])
			ssna = abs(fhzn-fhzp)
			# ssna stands for samples_to_skip_due_to_nyquist_asymmetry
			print "SSNA: ", ssna

			for j in range(self.brec+1,self.nrecs):
				# compute eq. 11
				print "...(cc) for receivers %d-%d on processor %d" %(j, self.brec, rank)
				if __name__ == '__main__':
					print nyst[self.brec],nyfin[self.brec],nxst[self.brec],nxfin[self.brec]
					print nyst[j],nyfin[j],nxst[j],nxfin[j]
				
				f_inv = np.conj(self.besselmat[nyst[self.brec]:nyfin[self.brec],nxst[self.brec]:nxfin[self.brec],1:self.nom_nneg]) * self.besselmat[nyst[j]:nyfin[j],nxst[j]:nxfin[j],1:self.nom_nneg]
				# ARJUN: note coordinate transformation here!! From position vector "r" to "r - r_alpha"

				fsyn = np.transpose(f_inv,[2,0,1]) * self.distribs_inv
				spa_int = np.sum(fsyn, axis=(1,2)) * self.dx**2
								
				# compute the cross-correlations for positive frequencies
				# Frequency-domain symmetry: calculations needed only for half the total number of frequencies.
				scc_brec[1:self.nom_nneg,j] = spa_int * self.pss[1:self.nom_nneg]
				
				# Negative frequency coefficients are complex conjugates of flipped positive coefficients.
				scc_brec[self.nom_nneg+ssna:,j] = np.flipud(np.conj(scc_brec[1:self.nom_nneg,j]))
				# June 22: BEWARE, the negative Nyquist term gets left out in case ssna>0, i.e. in case self.nom is even.
				# the same holds for obscross too.
				# this does matter of course, but it appears to make a very minor difference to the event kernels
				# so I am leaving it for the time being.

				# take care of constant factors
				ft_fac = self.dom/(2*np.pi)*self.nom
				scc_brec[:,j] *= ft_fac
				# ARJUN: why the multiplication with self.dom/(2*np.pi)*self.nom?

				# convert to time domain
				scc_brec[:,j] = np.fft.fftshift(np.fft.ifft(scc_brec[:,j]).real)

				if (not self.reald) and (self.iter==0):
					# Repeat above steps for the synthetic "observed" cross-correlations
					f_true = np.conj(self.besselmat[nyst_om[self.brec]:nyfin_om[self.brec],nxst_om[self.brec]:nxfin_om[self.brec],1:self.nom_nneg]) * self.besselmat[nyst_om[j]:nyfin_om[j],nxst_om[j]:nxfin_om[j],1:self.nom_nneg]
					fobs = np.transpose(f_true,[2,0,1]) * self.distribs_true
					occ_brec[1:self.nom_nneg,j] = np.sum(fobs, axis=(1,2)) * self.pss[1:self.nom_nneg] * self.dx**2
					occ_brec[self.nom_nneg+ssna:,j] = np.flipud(np.conj(occ_brec[1:self.nom_nneg,j]))
					occ_brec[:,j] = np.fft.fftshift(np.fft.ifft(occ_brec[:,j]).real)*ft_fac
				
		else:
			print "No cross-correlations to compute for processor %d" %(rank)
		
		scc_brec = scc_brec.real
		# this converts the entire array into a real-valued one
		if not self.reald:
			occ_brec = occ_brec.real		
				
		return scc_brec, occ_brec
				
	#######################################################################################################################

	def get_cc_upperhalf(self):								

		for k in range(self.nrecs):
			# [k,j] cross-correlation same as flipped [j,k]
			self.syncross[:,k,k+1:]=np.flipud(self.syncross[:,k+1:,k])
			self.obscross[:,k,k+1:]=np.flipud(self.obscross[:,k+1:,k])

	#######################################################################################################################

	def make_measurement(self):
		# from misfit.m
		
		print "In function make_measurement..."

		self.weightpos = np.zeros((self.nom, self.nrecs, self.nrecs))
		self.weightneg = np.zeros((self.nom, self.nrecs, self.nrecs))
		self.synamp_pos = np.zeros((self.nrecs, self.nrecs))
		self.synamp_neg = np.zeros((self.nrecs, self.nrecs))
		#if not self.reald:
		self.obsamp_pos = np.zeros((self.nrecs, self.nrecs))
		self.obsamp_neg = np.zeros((self.nrecs, self.nrecs))

		self.negl = np.zeros((self.nrecs, self.nrecs), dtype='int')
		self.negr = np.zeros((self.nrecs, self.nrecs), dtype='int')
		self.posl = np.zeros((self.nrecs, self.nrecs), dtype='int')
		self.posr = np.zeros((self.nrecs, self.nrecs), dtype='int')

		lefw = -4.0 #-1.0 #-0.25
		rigw = +4.0 #1.0 #+0.25

		#cslow = 1.2 #self.c - 1
		#cfast = 6.0 #self.c + 5

		for k in range(self.nrecs):
			for j in np.delete(np.arange(self.nrecs),k):
				#print "...(measurement) for receivers ", j,k

				if not self.reald:
				# SYNTHETIC DATA CASE
					# Simple windows suitable for synthetic data:
					# 1. Entire cross-correlation - [0:self.nom]
					# 2. Entire negative branch - [0:index of (sample 0)]
					# 3. Entire positive branch - [1 + index of (sample 0):self.nom]
				
					is0 = np.searchsorted(self.t,0)
			
					self.negl[j,k] = 0
					self.negr[j,k] = is0
					self.posl[j,k] = is0 + 1
					self.posr[j,k] = self.nom
				else:
				# REAL DATA CASE
					lef = max(0,self.dist_rp[j,k]/self.c + lefw) # left boundary of window (seconds)
					rig = self.dist_rp[j,k]/self.c + rigw # right boundary of window (seconds)

					#lef = self.dist_rp[j,k]/cfast # left boundary of window (seconds)
					#rig = self.dist_rp[j,k]/cslow # right boundary of window (seconds)

					self.negl[j,k] = np.searchsorted(self.t,-rig)
					self.negr[j,k] = np.searchsorted(self.t,-lef)
					self.posl[j,k] = np.searchsorted(self.t,lef) 
					self.posr[j,k] = np.searchsorted(self.t,rig)

				# the chosen windows (positive & negative side) should be of non-zero length, otherwise
				# the windowed cross-correlation energy, which divides the weight function, will be 0.
				# The windows can be zero-length if the arrival time for given station pair lies outside
				# the modelled time range (depending on wavespeed obviously).

				if self.negr[j,k]==0 or self.posl[j,k]==self.nom:
					print "Problem with stations ", j, k
					sys.exit("Aborted. The chosen window for computing cross-corrrelation energy \
						 lies outside the modelled time range")
				
				if __name__ == '__main__':
					print "Negative side window indices: ", self.negl[j,k], self.negr[j,k]
					print "Positive side window indices: ", self.posl[j,k], self.posr[j,k]
			
				self.weightpos[self.posl[j,k]:self.posr[j,k], j, k] = self.syncross[self.posl[j,k]:self.posr[j,k], j, k]
				self.weightneg[self.negl[j,k]:self.negr[j,k], j, k] = self.syncross[self.negl[j,k]:self.negr[j,k], j, k]
		
				self.synamp_pos[j,k] = np.sqrt(np.sum(self.weightpos[:,j,k]**2))#*self.deltat)
				#  Computing eq. 24 (numerator only), positive branch
				self.synamp_neg[j,k] = np.sqrt(np.sum(self.weightneg[:,j,k]**2))#*self.deltat)
				#  computing eq. 24 (numerator only), negative branch

				#if not self.reald:
				self.obsamp_pos[j,k] = np.sqrt(np.sum(self.obscross[self.posl[j,k]:self.posr[j,k],j,k]**2))#*self.deltat)
				self.obsamp_neg[j,k] = np.sqrt(np.sum(self.obscross[self.negl[j,k]:self.negr[j,k],j,k]**2))#*self.deltat)
				
				with np.errstate(invalid='raise'):
					try:
						self.weightpos[:,j,k] /= self.synamp_pos[j,k]**2
						self.weightneg[:,j,k] /= self.synamp_neg[j,k]**2
					except FloatingPointError as e :
						# this should never happen, none of the non-diagonal elements of self.synamp_pos or self.synamp_neg should be zero
						errargs_p=np.argwhere(self.synamp_pos==0)
						errargs_n=np.argwhere(self.synamp_neg==0)
						if not np.all(errargs_p[:,0]==errargs_p[:,1]) or not np.all(errargs_n[:,0]==errargs_n[:,1]) :
						# some non-diagonal elements of self.synamp_pos or self.synamp_neg or both are somehow zero
							print "RED FLAG!!!: ", e, errargs_p, errargs_n
							sys.exit("Problem with non-diagonal elements of cross-correlation energy matrices")

	#######################################################################################################################
	# from diffkernel.m

	def diffkernel(self, alpha, beta):
	# Computing source kernels for positive and negative branches

		#ccpos = (np.fft.fft(np.fft.ifftshift(self.weightpos[:,alpha,beta])))*self.deltat
		#ccneg = (np.fft.fft(np.fft.ifftshift(self.weightneg[:,alpha,beta])))*self.deltat
		# ARJUN: multiplication by self.deltat is only required here if it is also used in computation of synamp, obsamp

		ccpos = np.fft.fft(np.fft.ifftshift(self.weightpos[:,alpha,beta]))
		ccneg = np.fft.fft(np.fft.ifftshift(self.weightneg[:,alpha,beta]))
		  
		nxst1 = self.nstart_omost + self.nstart + self.clicksx[alpha] - 1
		nxfin1 = nxst1 + self.ngpib

		nyst1 = self.nstart_omost + self.nstart + self.clicksy[alpha] - 1
		nyfin1 = nyst1 + self.ngpib

		nxst2 = self.nstart_omost + self.nstart + self.clicksx[beta] - 1
		nxfin2 = nxst2 + self.ngpib

		nyst2 = self.nstart_omost + self.nstart + self.clicksy[beta] - 1
		nyfin2 = nyst2 + self.ngpib

		f = np.conj(self.besselmat[nyst1:nyfin1,nxst1:nxfin1,1:self.nom_nneg]) * self.besselmat[nyst2:nyfin2,nxst2:nxfin2,1:self.nom_nneg]

		#con = self.dom/(2*np.pi)
		con = 1/(2*np.pi)

		kp = 2 * (ccpos[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con
		kn = 2 * (ccneg[1:self.nom_nneg] * f * self.pss[1:self.nom_nneg]).real * con

		#kernpos = np.sum(kp, axis=2)
		#kernneg = np.sum(kn, axis=2)

		kernpos = spi.simps(kp,None,dx=self.dom,axis=2)
		kernneg = spi.simps(kn,None,dx=self.dom,axis=2)

		norm_kernpos = np.sum(kernpos*self.distribs_inv) * self.dx**2
		norm_kernneg = np.sum(kernneg*self.distribs_inv) * self.dx**2
		# kernel normalization, eq. 29

		if norm_kernpos < 0.95 or norm_kernneg < 0.95 or norm_kernpos > 1.05 or norm_kernneg > 1.05:
			raise SystemExit("Problem with normalization of source kernel for receivers %d-%d. Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg))
			#print "Problem with normalization of source kernel for receivers %d-%d. Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg)

		return kernpos, kernneg

############################## End of class "kernels_cc_amp" ###########################################################

if __name__ == '__main__':

	from mpi4py import MPI

	# Initialize MPI process
	comm_out = MPI.COMM_WORLD
	rank_out = comm_out.Get_rank()
	numproc_out = comm_out.Get_size()

	# Declare variables accessible by all processors
	class SignalParameters():
		""" defines the signal characteristics used for modelling sources (as also the cross-correlations)
		    and also, in case of real data, the processing parameters applied to the primary data before
		    cross-correlation.
		    Attributes: 
				dt: temporal sampling interval (seconds),
				nsam: number of samples,
				cf: center (peak) frequency, (Hz)
				lf: low corner frequency (real data frequency passband),
				hf: high corner frequency (real data frequency passband),
				altukey: Tukey window alpha parameter (real data post-whitening filter).
		"""

	sig_char = SignalParameters()
	sig_char.dt = 0.05 #0.2 #0.05
	sig_char.nsam = 401 #251 #401
	sig_char.cf = 2.0 #0.3 #2.0
	sig_char.lf = None
	sig_char.hf = None
	sig_char.altukey = None

	numrecs=20
	# receiver locations (rlocx and rlocy) are specified in number of grid points away from origin (along x,y axes)

	rlocx = np.array([18, 24, -23, 24, 7, -25, -14, 2, 27, 27, -21, 28, 27, -1, 18, -22, -5, 24, 17, 27])
	rlocy = np.array([9, -28, 20, 26, 10, 15, 14, -7, 9, -20, 12, -29, -14, -28, -25, 19, 11, -11, 27, -28])

	#numrecs=8 # this should be the size of rlocx and rlocy
	#rlocx=np.array([35, -35, 65, -65, 0, 0, 0, 0])
	#rlocy=np.array([0, 0, 0, 0, 35, -35, 45, -45])

	#numrecs=8
	#rlocx=np.array([6, 12, 40, 90, -6, -12, -40, -90])
	#rlocy=np.array([0, 0, 0, 0, 0, 0, 0, 0])

	#numrecs=4
	#rlocx=np.array([0, 0, -12, -90])
	#rlocy=np.array([12, 60, 0, 0])

	#numrecs=2
	#rlocx=np.array([-70, 90])
	#rlocy=np.array([70, 70])

	hlen_obox = 40 
	# half the length of side of outer box OR length of side of inner box (km)
	ngp_ibox = 341 
	# number of grid points in inner box, in either direction (half the number of grid points in outer box)
	wspeed = 3.0 
	# wavespeed everywhere in model (km/s)

	nrth=8
		
	##################### Do the actual computations on each processor #######################
		
	kcao = kernels_cc_amp(comm_out,rank_out,hlen_obox,ngp_ibox,numrecs,rlocx,rlocy,wspeed,sig_char)

	##################### Do final assimilation on master processor ##########################

	if rank_out==0:

		print "Completed program"
		print kcao.dist_rp

		# make figures as required

#		fig1=plt.figure()
#		ax1=fig1.add_subplot(111)
#		ax1.set_title("Positive branch")
#		cax1=ax1.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_pos)
#		fig1.colorbar(cax1)
#
#		fig2=plt.figure()
#		ax2=fig2.add_subplot(111)
#		ax2.set_title("Negative branch")
#		cax2=ax2.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_neg)
#		fig2.colorbar(cax2)
#
#		fig3=plt.figure()
#		ax3=fig3.add_subplot(111)
#		if kcao.omost_fac>2:
#			cax3=ax3.pcolor(kcao.gx2,kcao.gy2,kcao.distribs_true)
#		else:
#			cax3=ax3.pcolor(kcao.gx,kcao.gy,kcao.distribs_true)
#		ax3.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'd', markerfacecolor="None")
#		plt.colorbar(cax3,ax=ax3)

		def see_individual_skernels():

			lti=np.tril_indices(kcao.nrecs,k=-1)
			# lower triangular indices in numpy's default ordering
			ise=np.argsort(lti[1], kind='mergesort')
			r=lti[0][ise]
			c=lti[1][ise]
			cc_pdist=kcao.dist_rp[(r,c)]
			# now we have picked out lower triangular elements of kcao.dist_rp in the order that
			# we want them, i.e. in the order in which cross-correlations are done in this code

			fig_sk=plt.figure()
			for p in range(len(kcao.skers)):
				ax_sk=fig_sk.add_subplot(1,1,p+1, aspect='equal')
				cax_sk=ax_sk.pcolor(kcao.gx, kcao.gy, kcao.skers[p]) #, vmin=-0.1, vmax=0.1, cmap=plt.cm.jet)
				#ax_sk.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'kd', markerfacecolor="None")
				# use above line to plot all receivers on each subplot OR use below two lines to plot only the
				# relevant receiver pair on each subplot
				ax_sk.plot(kcao.dx*rlocx[r[p]], kcao.dx*rlocy[r[p]], 'kd', markerfacecolor="None")
				ax_sk.plot(kcao.dx*rlocx[c[p]], kcao.dx*rlocy[c[p]], 'kd', markerfacecolor="None")
				spname = "Distance %.2f km" %(cc_pdist[p])
				ax_sk.set_title(spname)
				plt.colorbar(cax_sk,ax=ax_sk)

		#if numrecs<nrth:
			#see_individual_skernels()

		#plt.show()
