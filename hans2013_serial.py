#!/usr/bin/python

# General purpose modules
import sys
import numpy as np
import scipy.signal as ss
import scipy.stats as sst
import scipy.special as ssp
import scipy.optimize as sop
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Modules written by me
import anseicca_utils1 as u1

##########################################################################################################################
# Documentation for reference

# 1. Curve fitting basic: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# 2. Curve fitting with weights example: https://scipython.com/book/chapter-8-scipy/examples/weighted-and-non-weighted-least-squares-fitting/

##########################################################################################################################

class inv_cc_amp:

	def __init__(self,hlboxo,ngpib,boxes_ratio,nrecs,rlocsx,rlocsy,cwav,signal,ring_rad,ring_w,iterate,only1_iter,dobs=None,dobs_info=None):

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

		# preliminaries - define global variables
		self.hlbox_outer = hlboxo
		self.ngpib = ngpib
		self.nrecs = nrecs
		self.c = cwav
		self.ring_rad = ring_rad
		self.wgauss = ring_w
		self.omost_fac = boxes_ratio

		"""
		 "omost_fac" is an integer factor that determines the size of the true model in the synthetic case
			It must be >=2; 
				if = 2, standard case (true and inverted models are of the same size)
				if = 3, the true model is the size of the "outer box" (four times the area of the inverted model)
		 Remember in the original code, all models are the size of the "inner box" (half the side length of outer box)
		"""

		self.nom = signal.nsam
		self.deltat = signal.dt
		f0 = signal.cf
		fl = signal.lf
		fh = signal.hf
		altuk = signal.altukey

		npairs=self.nrecs*(self.nrecs-1)/2

		if not (dobs is None):
		# real data case
			self.reald=True
			self.obscross = dobs
			self.obscross_aspec_temp = np.abs(np.fft.fft(dobs,axis=0))
		else:
		# synthetic data case
			self.reald=False
			self.obscross = np.zeros((self.nom, self.nrecs, self.nrecs), dtype='complex')

		self.dvar_pos = np.ones(npairs)
		self.dvar_neg = np.ones(npairs)

		self.clicksx = -1*rlocsx
		self.clicksy = -1*rlocsy
		# NB: rlocsx, rlocsy are the ACTUAL receiver locations. These are transformed to clicksx and clicksy
		# for the purpose of this class. The negative is because of the coordinate transformation between
		# "r" and "r - r_alpha" which is used in the functions compute_cc and diffkernel.

		self.setup(f0,fl,fh,altuk,dobs_info)

		self.num_mparams=self.basis.shape[0]
		self.distribs_inv=np.copy(self.distribs_start)
		self.allit_mc = []
		self.allit_misfit = []
		self.flit_indmis_p = []
		self.flit_indmis_n = []
		self.allit_synenv = []
		self.allit_syncross = []
		# variables with names ending in "_inv"  contain values for current (ulimately last) iteration only
		# variables with names starting with "allit_" are lists where each element corresponds to an iteration of the inversion.
		# variables with names starting with "flit_" are two-element lists, storing first (f) and last (l) iteration values only, of certain quantities.
		
		self.allit_mc.append(np.copy(self.mc_start))
		
		if __name__ == '__main__':
			if self.nrecs<nrth:
				self.skers=[] #range(npairs)

		itnum=0
		forced=False

		while iterate:
		
			self.iter = itnum
			self.syncross = np.zeros((self.nom, self.nrecs, self.nrecs), dtype='complex')
			self.syncross_aspec_temp = np.zeros(self.obscross.shape)

			#*************** inversion related variables
			self.Gmat_pos=np.zeros((npairs,self.num_mparams))
			self.Gmat_neg=np.zeros((npairs,self.num_mparams))
			
			self.deltad_pos=np.zeros(npairs)
			self.deltad_neg=np.zeros(npairs)

			self.ngrad1_pos=np.empty(self.num_mparams); self.ngrad2_pos=np.empty(self.num_mparams)
			self.ngrad1_neg=np.empty(self.num_mparams); self.ngrad2_neg=np.empty(self.num_mparams)
			# there are two ways of computing the gradient of chi: with and without explicit use of
			# the G-matrix. In other words: using individual kernels or using the total misfit kernel.
			# I compute the gradient in both ways (hence subscripts 1, 2 on the variables) and ensure
			# they are equal, for confidence in the calculations.
			
			mfit_kern_pos = np.zeros((self.ngpib, self.ngpib))
			mfit_kern_neg = np.zeros((self.ngpib, self.ngpib))
			# mfit_kern -> misfit_kernel

			#*************** compute synthetics and make measurements
			self.compute_cc()
			self.make_measurement()

			print "Starting computation of source kernels for each receiver pair..."
			cp=0 # cp stands for count_pair
			for j in range(self.nrecs-1): 
				for i in range(j+1,self.nrecs):
					print "...receivers ", i,j
					sker_p, sker_n = self.diffkernel(i,j)
					# Computing individual source kernels (eq. 15)

					# build the G-matrix
					kb_prod = sker_p*self.basis
					self.Gmat_pos[cp,:] = np.sum(kb_prod, axis=(1,2)) * self.dx**2
					kb_prod = sker_n*self.basis
					self.Gmat_neg[cp,:] = np.sum(kb_prod, axis=(1,2)) * self.dx**2
				
					if __name__ == '__main__': 
						if self.nrecs<nrth and itnum==0:
							self.skers.append(sker_p)
					
					self.deltad_pos[cp] = np.log(self.obsamp_pos[i,j]/self.synamp_pos[i,j])
					#print "obsamp_pos and synamp_pos for receivers ", i, j, self.obsamp_pos[i,j], self.synamp_pos[i,j]
					# Computing misfit kernels, i.e. eq. 30 (positive branch)
					mfit_kern_pos += sker_p * self.deltad_pos[cp]

					self.deltad_neg[cp] = np.log(self.obsamp_neg[i,j]/self.synamp_neg[i,j])
					#print "obsamp_neg and synamp_neg for receivers ", i, j, self.obsamp_neg[i,j], self.synamp_neg[i,j]
					# Computing misfit kernels, i.e. eq. 30 (negative branch)
					mfit_kern_neg += sker_n * self.deltad_neg[cp]

					cp+=1

			#*********** things to do on first iteration
			if itnum==0:
				if self.reald:
				# complete the calculation of the data errors. NB: we consider two types of error.
				# The first one (energy decay) is independent of the measurements and is already computed.
				# The second (SNR) is defined relative to the measurements, so we must get the absolute values here.

					dvar_snr_pos = np.square(self.esnrpd_ltpb * self.obsamp_pos)
					dvar_snr_neg = np.square(np.transpose(self.esnrpd_ltpb) * self.obsamp_neg)

					# combine different errors
					dvar_pos = dvar_snr_pos #+ self.dvar_egy_ltpb
					dvar_neg = dvar_snr_neg #+ np.transpose(self.dvar_egy_ltpb)

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
						
						print "dv_vec: ", dv_vec[br]

				# regardless of real or synthetic data, store the first-iteration values of certain quantities
				self.mfit_kern_pos = mfit_kern_pos
				self.mfit_kern_neg = mfit_kern_neg

			def record_flit():
				self.flit_indmis_p.append(self.deltad_pos)
				self.flit_indmis_n.append(self.deltad_neg)
			
		     	# record inversion progress
			#wmp = self.deltad_pos * self.dvar_pos
			#wmn = self.deltad_neg * self.dvar_neg
			wmp = self.deltad_pos / np.sqrt(self.dvar_pos)
			wmn = self.deltad_neg / np.sqrt(self.dvar_neg)
			total_misfit = 0.5*(np.dot(wmp,wmp) + np.dot(wmn,wmn))
			if itnum==0:
				record_flit()
			self.allit_misfit.append(total_misfit)
			self.allit_synenv.append(self.synenv)
			self.allit_syncross.append(self.syncross)

			if itnum==1:
				record_flit()
				if only1_iter:
				# FORCED STOP FOR TESTING: last misfit stored will correspond to first updated model
					forced=True; iterate=False 
				

			if (itnum>0) and (not forced):
			# determine whether to terminate inversion or iterate further
				mf_curr = self.allit_misfit[-1]
				mf_prev = self.allit_misfit[-2]
				pchange = 100*(mf_prev - mf_curr)/mf_prev
				if (pchange>0 and pchange<5) or itnum>15:
					iterate=False
					#inversion terminated.
					# store the individual misfits corresponding to the final iteration model
					record_flit()

			if iterate:
				#*********** do actual inversion (model update)
				update_mod = self.inversion(mfit_kern_pos,mfit_kern_neg)
				self.distribs_inv += update_mod
				itnum +=1
				print "END OF ITERATION %d" %(itnum)

		#*********************** End of loop over iterations *******************

	########################################################################################################################

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
			self.Cm = np.zeros((self.num_mparams,self.num_mparams))
			cord = 3
			# cord -> correlation_distance
			for r in range(self.Cm.shape[0]):
				col = np.arange(float(self.Cm.shape[1]))
				self.Cm[r,:] = ((1./self.gamma)**2)*np.exp(-0.5*(((r-col)/cord)**2))
		
			CmInv = np.linalg.inv(self.Cm)

			# to view the model covariance matrix (with iPython), use:
			# x=np.arange(kcao.Cm.shape[0]); y=np.arange(kcao.Cm.shape[1])
			# gx,gy=np.meshgrid(x,y)
			# plt.pcolor(gx,gy,kcao.Cm)

		#*************************************** End of damping ************************************************#
				
		m_iter = self.allit_mc[-1]
		m_prior = self.mc_start

		G = {'p': self.Gmat_pos, 'n': self.Gmat_neg}
		dd = {'p': self.deltad_pos, 'n': self.deltad_neg}
		mfk = {'p': mfk_pos, 'n': mfk_neg}
		ng1 = {'p': self.ngrad1_pos, 'n': self.ngrad1_neg}
		ng2 = {'p': self.ngrad2_pos, 'n': self.ngrad2_neg}
		dvi = {'p': 1./self.dvar_pos, 'n': 1./self.dvar_neg}
		mod_update = np.zeros((self.ngpib, self.ngpib))
		deltam = np.zeros((2,self.num_mparams))

		for b,br in enumerate(G):
		# br -> branch (positive or negative)

			# compute basic (unweighted, undamped) gradient: method 1
			kb_prod2 = mfk[br]*self.basis
			ng1[br][:] = np.sum(kb_prod2, axis=(1,2)) * self.dx**2
			# compute basic gradient: method 2
			ng2[br] = np.matmul(G[br].T,dd[br])
			if (not np.allclose(ng1[br],ng2[br],rtol=1e-03)):
				#sys.exit("Quitting. Problem computing gradient.")
				pass

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

	########################################################################################################################
	
	def setup(self,f0,lowf,highf,ptukey,dinfo):	

		# from setup.m

		self.fhz=np.fft.fftfreq(self.nom,self.deltat)
		self.omega = 2*np.pi*self.fhz
		self.dom = self.omega[1] - self.omega[0] # d_omega
		self.nom_nneg = len(self.fhz[self.fhz>=0])
		# number of non-negative frequency samples
		# remember, when self.nom is even, self.nom_nneg is smaller by one sample: the positive Nyquist is missing.
		tdur=self.nom*self.deltat
		# inverse frequency spacing for DFT of time series
		print "tdur as seen by h13 module: ", tdur
		if self.nom%2 != 0:
			self.t = np.arange(-(self.nom_nneg-1),self.nom_nneg)*self.deltat
			# time series corresponding to cross-correlation lags
			# NB: the crucial advantage of building a signal as above, rather than doing np.arange(tstart,tend,deltat),
			# is that the above formulation ensures that you always get the time sample zero, regardless of deltat.
		else:
			self.t = np.arange(-self.nom_nneg,self.nom_nneg)*self.deltat
		if len(self.t) != self.nom:
			sys.exit("Quitting. Length of time array does not match number of samples. Check signal parameters.")
			
		#************************************* build source characteristics ********************************************
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

			#self.sourcetime = np.exp(-self.t**2/(512*(0.05**2))) * np.cos(2*np.pi*f0*self.t) # matlab code original
			self.sourcetime = np.exp(-self.t**2/a) * np.cos(2*np.pi*f0*self.t)
			#self.pss = np.abs(np.fft.fft(self.sourcetime)*self.deltat)**2 
			# ARJUN: why the multiplication by self.deltat above?
			self.pss = np.abs(np.fft.fft(self.sourcetime))**2
			# pss stands for power_spectrum_of_sources
		else:
		# real data case

			max_each_rp = np.max(self.obscross_aspec_temp, axis=0)
			norm_obs_aspec = np.copy(self.obscross_aspec_temp)
			with np.errstate(invalid='raise'):
				try:
					norm_obs_aspec /= max_each_rp
				except FloatingPointError as e:
					errargs=np.argwhere(max_each_rp==0)
					if not np.all(errargs[:,0]<=errargs[:,1]):
					# only the upper triangular part + main diagonal of "max_each_rp" should be 0
						sys.exit("Problem normalizing the observed amplitude spectra (to their individual maxima)")

			self.obs_aspec_mean = np.nanmean(norm_obs_aspec,axis=(1,2))
			# NB: this spectrum is only useful for its shape. It is a DUMMY as far as amplitude is concerned.
			dummy_egy_funcf = (self.obs_aspec_mean)**2/(self.nom)
			dummy_pow = np.sum(dummy_egy_funcf,axis=0)

			fhzp=self.fhz[self.fhz>=0]
			fhzn=self.fhz[self.fhz<0]
			# taking zero on the positive side ensures that both branches are of equal size, because remember that for
			# even number of samples, the positive side is missing the Nyquist term.

			#rvp=sst.skewnorm(a=-5,loc=0.55,scale=0.15)
			#rvn=sst.skewnorm(a=5,loc=-0.55,scale=0.15)

			rvp=sst.skewnorm(a=-3,loc=0.5,scale=0.13)
			rvn=sst.skewnorm(a=3,loc=-0.5,scale=0.13)

			self.pss = np.concatenate((rvp.pdf(fhzp),rvn.pdf(fhzn)))

		#************************************* end of source characteristics *******************************************

		if __name__ == '__main__':
#
#			fig_stf=plt.figure()
#			ax_stf=fig_stf.add_subplot(111)
#			ax_stf.plot(self.t,self.sourcetime)
#			ax_stf.set_xlabel('Time [s]')
#			ax_stf.set_title('Source time function')
#
			fig_ss=plt.figure()
			ax_ss=fig_ss.add_subplot(111)
			ax_ss.plot(np.fft.fftshift(self.fhz),np.fft.fftshift(self.pss))
			ax_ss.set_xlim(0,1/(2*self.deltat))
			ax_ss.set_xlabel('Frequency [Hz]')
			ax_ss.set_title('Sources power spectrum')
		#***************************************************************************************************************

		self.ntot=2*self.ngpib-1
		self.dx = self.hlbox_outer/(self.ngpib-1.0)
		self.nstart = self.ngpib/2 + 1 #int(math.floor(self.ngpib/2)+1)
		self.nmid = self.ngpib # mid point
 
		hlbox_omost = self.omost_fac*self.hlbox_outer/2.0
		ntot_omost = self.omost_fac*(self.ngpib-1) + 1
		self.nstart_omost = (self.omost_fac - 2)*(self.nstart-1)

		# think of the "nstart" variables as the number of grid points between an outer box and the next inner box

		# grid points of outer-most box
		x3=np.linspace(-hlbox_omost,hlbox_omost,ntot_omost)
		y3=np.linspace(-hlbox_omost,hlbox_omost,ntot_omost)

		# grid points of outer box
		x2=np.linspace(-self.hlbox_outer,self.hlbox_outer,self.ntot)
		y2=np.linspace(-self.hlbox_outer,self.hlbox_outer,self.ntot)

		# grid points of inner box
		x = x2[self.nstart:self.nstart+self.ngpib]
		y = np.copy(x)

		self.dist_rp=np.zeros((self.nrecs,self.nrecs))
		for j in range(self.nrecs):
			for i in range(self.nrecs):
				self.dist_rp[i,j] = np.sqrt( (x2[self.clicksx[i]+self.nmid]-x2[self.clicksx[j]+self.nmid])**2 + (y2[self.clicksy[i]+self.nmid]-y2[self.clicksy[j]+self.nmid])**2 )

		dist_all = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]
		self.alldist = dist_all[np.argsort(dist_all)]
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

		self.besselmat = np.zeros((ntot_omost,ntot_omost,self.nom_nneg), dtype='complex')

		print "Computing Hankel functions.."
		for i in range(1,self.nom_nneg):
			self.besselmat[:,:,i] = ssp.hankel1(0,self.omega[i]*r/self.c) * 1j * 0.25

		#******************************* source distributions and observation errors ********************************************
		def mult_gauss(j,mag,xp,yp,xw,yw):

			# xp -> xpos (in dx units)
			# xw -> xwidth (in dx units)
			if self.omost_fac>2:
				usex=x2
				usey=y2
			else:
				usex=x
				usey=y

			ans = mag * np.exp( -( (usex[j] - xp*self.dx)**2/(xw*(self.dx**2)) + (usey - yp*self.dx)**2/(yw*(self.dx**2)) ) )
			return ans

		sdist_type = {0: mult_gauss, 1: u1.somod.ringg, 2: u1.somod.rgring}
		alltheta_deg=np.arange(0,360,10)
		alltheta=alltheta_deg*np.pi/180
		self.mc_start = np.ones(alltheta.size)
		# mc -> model_coefficients
		self.basis = np.zeros((alltheta.size,self.ngpib,self.ngpib))
		if self.omost_fac>2:
			basis_true = np.zeros((alltheta.size,self.ntot,self.ntot))

		mag1=1
		self.mc_start *= mag1

		if not self.reald:
		# SYNTHETIC DATA CASE
			self.mc_true = np.ones(alltheta.size)
			if self.omost_fac>2:
				self.distribs_true = np.zeros((self.ntot,self.ntot))
				# used to generate the synthetic "data" in the absence of real data
				# hence used for testing inversions; this source distribution is NOT involved in computing source kernels
			else:
				self.distribs_true = np.zeros((self.ngpib,self.ngpib))

			self.distribs_start = np.zeros((self.ngpib,self.ngpib)) 
			# used to generate the synthetics for inversion; this source distribution is involved in computing source kernels

			self.mc_true *= mag1
			nperts=4
			# nperts -> the number of "regions" in the True model which are perturbed
			#t1=[130,220,345,30]
			#t2=[150,240,15,50]
			#mag2=[8,5,5]
			t1=[75,165,255,345]
			t2=[105,195,285,15]
			mag2=[4,8,4,8]
			if (len(t1)<nperts) or (len(t2)<nperts) or (len(mag2)<nperts):
				sys.exit("Problem building the True model. Please check.")

			#for col in range(self.distribs_true.shape[1]):
			#	self.distribs_true[:,col] += sdist_type[0](col,10*mag1,0,-140,2000,75) + sdist_type[0](col,10*mag1,-160,120,75,2000)

			for p in range(nperts):
				s1=np.argwhere(alltheta_deg >= t1[p])
				s2=np.argwhere(alltheta_deg <= t2[p])
				relind=np.intersect1d(s1,s2)
				if len(relind)==0:
					relind=np.union1d(s1,s2)
				self.mc_true[relind]=mag2[p]+mag1

			for k,theta in enumerate(alltheta):
				self.basis[k,:,:] = sdist_type[1](self.ngpib,self.dx,x,y,theta,self.ring_rad,self.wgauss)
				self.distribs_start += self.mc_start[k]*self.basis[k,:,:]
				if self.omost_fac>2:
					basis_true[k,:,:] = sdist_type[1](self.ntot,self.dx,x2,y2,theta,2*self.ring_rad,self.wgauss)
					#self.distribs_true += self.mc_true[k]*basis_true[k,:,:]
				else:
					self.distribs_true += self.mc_true[k]*self.basis[k,:,:]

			# no observation errors in the synthetic case
			# nothing to do on this front.
		else:
		# REAL DATA CASE
			self.distribs_start = np.zeros((self.ngpib,self.ngpib))
			for k,theta in enumerate(alltheta):
				self.basis[k,:,:] = sdist_type[1](self.ngpib,self.dx,x,y,theta,self.ring_rad,self.wgauss)
				self.distribs_start += self.mc_start[k]*self.basis[k,:,:]
				
			########################### Initial amplitudes and observation errors ##############################

			#********* Initial amplitudes (amplitudes of starting synthetics)

			occegy_funcf = np.square(self.obscross_aspec_temp)
			# occegy_funcf -> observed_cc_power_as_a_function_of_frequency
			occ_egy = np.sum(occegy_funcf,axis=0)/self.nom
			egy_obs = occ_egy[np.nonzero(np.tril(self.dist_rp))]
			# the matrix is symmetric so it suffices to consider only its lower triangular part
			self.egy_obs = egy_obs[np.argsort(dist_all)]

			def one_by_r(x,k):
				return k*1./x

			nf_dist = 0.5*self.c/highf
			# nf_dist -> near_field_distance. Using a very crude estimate: half the shortest wavelength in the data 
			sd_ind=np.argwhere(self.alldist<nf_dist)
			# sd_ind -> short_distance_indices
			self.sig_dummy = np.ones(self.alldist.size)
			self.sig_dummy[sd_ind] = 5
			# NB: self.sig_dummy - deliberately called "dummy" - contains basically the relative weights for the data points, NOT
			# the actual standard deviations. This is reflected in the argument "absolute_sigma=False" to scipy's curve fit.
						
			popt, pcov = sop.curve_fit(one_by_r,self.alldist,self.egy_obs,sigma=self.sig_dummy,absolute_sigma=False)
			self.oef = popt[0]/self.alldist
			#oef -> observed_energy_fitted

			#********* Errors Part 1: error due to SNR
			snr = dinfo[1]
			
			self.esnrpd_ltpb = np.zeros((self.dist_rp.shape))
			# esnrpd -> error(due to)_SNR_(as a)_percentage_(of)_data
			# ltpb -> lower_triangle_positive_branch
			# (it is implied that the upper triangle of the matrix is for the negative branch)

			self.esnrpd_ltpb[np.where(snr<2)]=0.8
			self.esnrpd_ltpb[np.where((snr>2) & (snr<3))]=0.5
			self.esnrpd_ltpb[np.where(snr>3)]=0.05

			#********* Errors Part 2: error due to energy decay with distance

			delA = dinfo[2]

			#********************************************************************************************************************
			# NB: uncertainties in the observations contained in dinfo need to be corrected, because the measurement for
			# the kernels involves cc energies computed in a certain window only, whereas the curve fitting above is done using
			# the energy of the entire cc branch. This correction can be made using the waveform's S/N ratio, which indirectly
			# provides a measure of the contribution of the window of interest, to the total energy of the waveform (branch).
			#********************************************************************************************************************

			# refine the error so it applies to the measurement window only
			nsr = 1./snr
			ScT = 1./(1+nsr) # 1./np.sqrt(1+nsr)
			# ScT -> signal_contribution_to_total (energy)
			delA *= ScT

			# convert to variance			
			self.dvar_egy_ltpb = np.square(delA)

#			#********* Errors Part 3: position error due to relocation of receivers to grid points
#			origdist_rp = dinfo[0]
#			deltapos = np.square(origdist_rp - self.dist_rp)

			############################## End of observation errors ######################################
		
		print "Completed initial setup..."

	########################################################################################################################
	
	def compute_cc(self):
		# from compute_cc.m

		print "Computing cross-correlations (positive frequencies)..."

		todo_syn=True

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

		#self.pss_temp = np.zeros((self.nom_nneg-1, self.nrecs, self.nrecs))
		self.mod_spa_int_temp = np.zeros((self.nom_nneg-1, self.nrecs, self.nrecs))

		# account for possible asymmetry in frequency samples (happens when self.nom is even)
		fhzp = len(self.fhz[self.fhz>0])
		fhzn = len(self.fhz[self.fhz<0])
		ssna = abs(fhzn-fhzp)
		# ssna stands for samples_to_skip_due_to_nyquist_asymmetry
		print "SSNA: ", ssna
		#print "Frequency samples are: ", self.fhz

		while todo_syn:
			for k in range(self.nrecs-1):
				for j in range(k+1,self.nrecs):
					# compute eq. 11
					print "...cc for receivers ", j, k
					#print nyst[k],nyfin[k],nxst[k],nxfin[k]
					#print nyst[j],nyfin[j],nxst[j],nxfin[j]
					#print self.nom_nneg

					f_inv = np.conj(self.besselmat[nyst[k]:nyfin[k],nxst[k]:nxfin[k],1:self.nom_nneg]) * self.besselmat[nyst[j]:nyfin[j],nxst[j]:nxfin[j],1:self.nom_nneg]
					# ARJUN: note coordinate transformation here!! From position vector "r" to "r - r_alpha"

					fsyn = np.transpose(f_inv,[2,0,1]) * self.distribs_inv #self.distribs_start
					#print "Shape of f_inv is ", f_inv.shape
					spa_int = np.sum(fsyn, axis=(1,2)) * self.dx**2
									
					# compute the cross-correlations for positive frequencies
					# Frequency-domain symmetry: calculations needed only for half the total number of frequencies.
					if not self.reald:
						self.syncross[1:self.nom_nneg,j,k] = spa_int * self.pss[1:self.nom_nneg]
						self.mod_spa_int_temp[:,j,k] = np.abs(spa_int)
					else:
						self.mod_spa_int_temp[:,j,k] = np.abs(spa_int)
						#self.pss_temp[:,j,k] = self.mobscross_aspec[1:self.nom_nneg,j,k] / self.mod_spa_int_temp[:,j,k]
						#self.syncross[1:self.nom_nneg,j,k] = spa_int * (self.pss_temp[:,j,k])
						self.syncross[1:self.nom_nneg,j,k] = spa_int * self.pss[1:self.nom_nneg]
														
					# Negative frequency coefficients are complex conjugates of flipped positive coefficients.
					self.syncross[self.nom_nneg+ssna:,j,k] = np.flipud(np.conj(self.syncross[1:self.nom_nneg,j,k]))
					# June 22: BEWARE, the negative Nyquist term gets left out in case ssna>0, i.e. in case self.nom is even.
					# the same holds for obscross too.
					# this does matter of course, but it appears to make a very minor difference to the event kernels
					# so I am leaving it for the time being.

					# take care of constant factors
					ft_fac = self.dom/(2*np.pi)*self.nom
					self.syncross[:,j,k] *= ft_fac
					# ARJUN: why the multiplication with self.dom/(2*np.pi)*self.nom?

					self.syncross_aspec_temp[:,j,k] = np.abs(self.syncross[:,j,k])

					# convert to time domain					
					self.syncross[:,j,k] = np.fft.fftshift(np.fft.ifft(self.syncross[:,j,k]).real)

					if (not self.reald) and (self.iter==0):
					# SYNTHETIC DATA CASE
						# Compute the "data" if dealing with a synthetic problem, repeating the above steps
						f_true = np.conj(self.besselmat[nyst_om[k]:nyfin_om[k],nxst_om[k]:nxfin_om[k],1:self.nom_nneg]) * self.besselmat[nyst_om[j]:nyfin_om[j],nxst_om[j]:nxfin_om[j],1:self.nom_nneg]
						fobs = np.transpose(f_true,[2,0,1]) * self.distribs_true
						self.obscross[1:self.nom_nneg,j,k] = np.sum(fobs, axis=(1,2)) * self.pss[1:self.nom_nneg] * self.dx**2
						self.obscross[self.nom_nneg+ssna:,j,k] = np.flipud(np.conj(self.obscross[1:self.nom_nneg,j,k]))
						self.obscross[:,j,k] = np.fft.fftshift(np.fft.ifft(self.obscross[:,j,k]).real)*ft_fac

			#print np.nanmax(self.distribs_inv)
			
			if self.iter==0:
			# First iteration
				sccegy_funcf = np.square(self.syncross_aspec_temp)
				scc_egy = np.sum(sccegy_funcf,axis=0)/self.nom
				egy_syn = scc_egy[np.nonzero(np.tril(self.dist_rp))]
				da = self.dist_rp[np.nonzero(np.tril(self.dist_rp))]
				self.egy_syn = egy_syn[np.argsort(da)]

				def one_by_r(x,k):
					return k*1./x
				
				try:
					popt, pcov = sop.curve_fit(one_by_r,self.alldist,self.egy_syn,sigma=self.sig_dummy,absolute_sigma=False)
				except AttributeError:
				# Attribute sig_dummy does not exist in case of synthetic data
					popt, pcov = sop.curve_fit(one_by_r,self.alldist,self.egy_syn)

				self.sef = popt[0]/self.alldist
				#sef -> synthetic_energy_fitted

				if self.reald:
				# REAL DATA CASE
					esf = np.mean(self.oef/self.sef)
					# esf -> energy_scale_factor
					if esf > 0.9 and esf < 1.1:
						todo_syn=False
					else:
						print "esf is %f, MULTIPLYING self.pss by %f" %(esf,np.sqrt(esf))
						self.pss *= np.sqrt(esf)
				else:
				# SYNTHETIC DATA CASE
					todo_syn=False
					# we're done, synthetics have been built.
			else:
			# Subsequent iterations
				# no need for energy fitting whether real or synthetic data; proceed with algorithm
				todo_syn = False

		#***** END OF WHILE LOOP *******
								
		# convert the entire array into a real-valued one
		self.syncross = self.syncross.real
		#if not self.reald:
		self.obscross = self.obscross.real		

		for k in range(self.nrecs):
			# [k,j] cross-correlation same as flipped [j,k]
			self.syncross[:,k,k+1:]=np.flipud(self.syncross[:,k+1:,k])
			self.obscross[:,k,k+1:]=np.flipud(self.obscross[:,k+1:,k])

		self.obsenv=np.abs(ss.hilbert(self.obscross, axis=0))
		self.synenv=np.abs(ss.hilbert(self.syncross, axis=0))

		if self.iter==0:
			self.obscross_aspec_temp = np.abs(np.fft.fft(self.obscross,axis=0))
			# for the synthetic data case, this is generated here for the FIRST time; for the real data case
			# this is a recalculation but now the upper triangular part of the matrix is also filled in.

	#######################################################################################################################

	def make_measurement(self):
		# from misfit.m
		
		print "In function make_measurement..."

		self.weightpos = np.zeros((self.nom, self.nrecs, self.nrecs))
		self.weightneg = np.zeros((self.nom, self.nrecs, self.nrecs))
		self.synamp_pos = np.zeros((self.nrecs, self.nrecs))
		self.synamp_neg = np.zeros((self.nrecs, self.nrecs))
		self.obsamp_pos = np.zeros((self.nrecs, self.nrecs))
		self.obsamp_neg = np.zeros((self.nrecs, self.nrecs))

		initscal = np.zeros((self.nrecs, self.nrecs))

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

				#print "Negative side window indices: ", self.negl[j,k], self.negr[j,k]
				#print "Positive side window indices: ", self.posl[j,k], self.posr[j,k]

				# now make the measurements
		
				print "making measurement for receivers ", j,k

				self.weightpos[self.posl[j,k]:self.posr[j,k], j, k] = self.syncross[self.posl[j,k]:self.posr[j,k], j, k]
				self.weightneg[self.negl[j,k]:self.negr[j,k], j, k] = self.syncross[self.negl[j,k]:self.negr[j,k], j, k]
		
				self.synamp_pos[j,k] = np.sqrt(np.sum(self.weightpos[:,j,k]**2))#*self.deltat)
				#  Computing eq. 24 (numerator only), positive branch
				self.synamp_neg[j,k] = np.sqrt(np.sum(self.weightneg[:,j,k]**2))#*self.deltat)
				#  computing eq. 24 (numerator only), negative branch

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
							print self.syncross[self.posl[j,k]:self.posr[j,k], j, k]
							#sys.exit("Problem with non-diagonal elements of measurement matrices")

	#######################################################################################################################
	
	def diffkernel(self, alpha, beta):
	# from diffkernel.m
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
			#sys.exit("Problem with normalization of source kernel for receivers %d-%d. Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg))
			print "Problem with normalization of source kernel for receivers %d-%d. Norms (pos/neg) are: %f,%f" %(alpha,beta,norm_kernpos,norm_kernneg)

		return kernpos, kernneg

############################################ Main program ###########################################################

if __name__ == '__main__':

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
	sig_char.dt = 0.05 #0.2
	sig_char.nsam = 401 #251
	sig_char.cf = 2.0 #0.3
	sig_char.lf = None
	sig_char.hf = None
	sig_char.altukey = None

	#numrecs=20
	# receiver locations (rlocx and rlocy) are specified in number of grid points away from origin (along x,y axes)

	#rlocx = np.array([18, 24, -23, 24, 7, -25, -14, 2, 27, 27, -21, 28, 27, -1, 18, -22, -5, 24, 17, 27])
	#rlocy = np.array([9, -28, 20, 26, 10, 15, 14, -7, 9, -20, 12, -29, -14, -28, -25, 19, 11, -11, 27, -28])

	#numrecs=8 # this should be the size of rlocx and rlocy
	#rlocx=np.array([35, -35, 65, -65, 0, 0, 0, 0])
	#rlocy=np.array([0, 0, 0, 0, 35, -35, 45, -45])

	#numrecs=8
	#rlocx=np.array([6, 12, 40, 90, -6, -12, -40, -90])
	#rlocy=np.array([0, 0, 0, 0, 0, 0, 0, 0])

	#numrecs=4
	#rlocx=np.array([12, 90, -12, -90])
	#rlocy=np.array([0, 0, 0, 0])

	#numrecs=4
	#rlocx=np.array([0, 0, -12, -90])
	#rlocy=np.array([12, 60, 0, 0])

	numrecs=2
	rlocx=np.array([-20, 50])
	rlocy=np.array([70, 100])

	hlen_obox = 40 
	# half the length of side of outer box OR length of side of inner box (km)
	ngp_ibox = 341 
	# number of grid points in inner box, in either direction (half the number of grid points in outer box)
	wspeed = 3.0 
	# wavespeed everywhere in model (km/s)
	""" NB: meaning of outer and inner boxes - the receivers can be anywhere in the outer box but the sources are
		constrained to lie within the inner box 
	"""

	nrth=8
	# nrth is just a receiver number threshold below which individual source kernels are stored (and plotted)
	kcao = inv_cc_amp(hlen_obox,ngp_ibox,numrecs,rlocx,rlocy,wspeed,sig_char)

	# make figures as required

	pairs=numrecs*(numrecs-1)/2

	fig1=plt.figure()
	ax1=fig1.add_subplot(111)
	ax1.set_title("Positive branch")
	cax1=ax1.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_pos,cmap=plt.cm.jet)
	ax1.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
	fig1.colorbar(cax1)

	fig2=plt.figure()
	ax2=fig2.add_subplot(111)
	ax2.set_title("Negative branch")
	cax2=ax2.pcolor(kcao.gx,kcao.gy,kcao.mfit_kern_neg,cmap=plt.cm.jet)
	ax2.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
	fig2.colorbar(cax2)

	fig3=plt.figure()
	ax3=fig3.add_subplot(111)
	ax3.set_title("True model")
	if kcao.omost_fac>2:
		cax3=ax3.pcolor(kcao.gx2,kcao.gy2,kcao.distribs_true,cmap=plt.cm.jet)
	else:
		cax3=ax3.pcolor(kcao.gx,kcao.gy,kcao.distribs_true,cmap=plt.cm.jet)
	ax3.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'wd', markerfacecolor="None")
	plt.colorbar(cax3,ax=ax3)

	print "Inter-receiver distances: "
	print kcao.dist_rp

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
			ax_sk=fig_sk.add_subplot(3,2,p+1, aspect='equal')
			cax_sk=ax_sk.pcolor(kcao.gx, kcao.gy, kcao.skers[p],cmap=plt.cm.jet) #, vmin=-0.1, vmax=0.1)
			#ax_sk.plot(kcao.dx*rlocx, kcao.dx*rlocy, 'kd', markerfacecolor="None")
			# use above line to mark all receivers on each subplot OR use below two lines to mark only the
			# relevant receiver pair on each subplot
			ax_sk.plot(kcao.dx*rlocx[r[p]], kcao.dx*rlocy[r[p]], 'kd', markerfacecolor="None")
			ax_sk.plot(kcao.dx*rlocx[c[p]], kcao.dx*rlocy[c[p]], 'kd', markerfacecolor="None")
			spname = "Distance %.2f km" %(cc_pdist[p])
			ax_sk.set_title(spname)
			plt.colorbar(cax_sk,ax=ax_sk)

	if numrecs<nrth:
		if len(kcao.skers) == kcao.nrecs*(kcao.nrecs-1)/2:
			try:
				see_individual_skernels()
			except ValueError:
				print "Problem plotting individual source kernels. Please check number of subplots and try again next time"
		else:
			sys.exit("Problem with number of source kernels")
	plt.show()
