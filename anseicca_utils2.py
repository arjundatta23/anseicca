import os
import sys
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

reald=sys.modules['__main__'].use_reald
if reald:
	import obspy.core as oc
	# Modules written by me
	import azimuthal_anal_cc as azan	

###############################################################################################################
class setup_modelling_domain:

	def __init__(self,st_no,st_id,xrec,yrec,ox,oy,dxy,grd_err_max,make_plots):

		self.dxy=dxy

		#************** redefine receiver locations for chosen grid and coordinate origin ************
		recx_ro = xrec - ox
		recy_ro = yrec - oy
		# ro -> relative_to_origin
		recx_gp = recx_ro/dxy
		recy_gp = recy_ro/dxy
		# receiver locations in integer grid points away from coordinate origin
		recx_igp = np.asarray(np.rint(recx_gp), dtype=int)
		recy_igp = np.asarray(np.rint(recy_gp), dtype=int)

		#************** Apply selection criterion for selecting a subset of entire array *************
		ro_act = np.sqrt(recx_ro**2 + recy_ro**2)
		ro_grid = np.sqrt((recx_igp*dxy)**2 + (recy_igp*dxy)**2)
		ichosen = np.argwhere(np.abs(ro_act-ro_grid)<grd_err_max)
		self.num_chosen=len(ichosen)
		self.rchosenx_igp = recx_igp[ichosen.flatten()]
		self.rchoseny_igp = recy_igp[ichosen.flatten()]
		chosen_st_id = [st_id[s] for s in ichosen.flatten()]
		self.chosen_st_no = [st_no[s] for s in ichosen.flatten()]
		self.rnchosen = [j+1 for j in ichosen.flatten()]
		print "\nChosen %d out of %d receivers: " %(self.num_chosen,xrec.size)
		print chosen_st_id
		#print self.rnchosen

		#********************************** make plots if desired ************************************
		if make_plots:
			self.plot_absolute_actual(xrec,yrec,ichosen)
			self.plot_relative_gridded(recx_ro[ichosen],recy_ro[ichosen])

		#*********************************************************************************************

	def plot_absolute_actual(self,rx,ry,ind_in):
		ind_all = np.arange(rx.size)
		ind_out = np.setdiff1d(ind_all,ind_in)
		xry_act=plt.figure()
		axmap=xry_act.add_subplot(111,aspect='equal')
		axmap.scatter(rx[ind_out],ry[ind_out],marker='^',s=100,facecolor='r',alpha=0.1,edgecolor='k')
		axmap.scatter(rx[ind_in],ry[ind_in],marker='^',s=100,facecolor='r',edgecolor='k')
		axmap.set_xlabel("Easting [km]")
		axmap.set_ylabel("Northing [km]")
		axmap.set_title("Absolute coordinates")

	def plot_relative_gridded(self,act_recx,act_recy):
		xpos_grid=self.rchosenx_igp*self.dxy
		ypos_grid=self.rchoseny_igp*self.dxy
		fig=plt.figure()
		axgrid=fig.add_subplot(111, aspect='equal')
		axgrid.scatter(act_recx,act_recy,marker='^',facecolor='b',label="Actual")
		axgrid.scatter(xpos_grid,ypos_grid,marker='o',facecolor='r',label="On uniform grid")
		axgrid.set_xlabel('Km')
		axgrid.set_ylabel('Km')
		axgrid.set_title("Relative coordinates")
		axgrid.legend()		

########################################################################################################################################

class cc_data:

	class Read:

		def __init__(cc_data,infile,funcname):

			global si, reclen, fpband, ccl, orig_data, allstored, ctyp

			cc_data.inpfile = infile
			fDic={'python_binary_archive': cc_data.python_binary_archive()}
			fDic[funcname]
			
			si=cc_data.sami
			fpband=cc_data.fpb
			ccl=cc_data.cclags
			orig_data=cc_data.cookie
			allstored=cc_data.recarray
			reclen=2*cc_data.cclags[-1]

			print "\n\nSignal parameters of original data: "
			print "Number of samples: ", orig_data.shape[0]
			print "Sample spacing (time domain): ", si

		def python_binary_archive(cc_data):

			loaded = np.load(cc_data.inpfile)
			print "Reading ", cc_data.inpfile

			try:
				cc_data.recarray=loaded['reclist']
				cc_data.fpb=loaded['fpband']
				cc_data.sami=loaded['si'][0]
				cc_data.cclags=loaded['cclags']
				cc_data.cookie=loaded['cookie']
			except KeyError:
				raise SystemExit("Problem reading %s" %(cc_data.inpfile))

			cc_data.cclags=np.array(map(lambda x: float("%.2f" %x), cc_data.cclags))
			# making sure that ccl does not contain elements with more significant digits than are allowed by the sampling interval
		
			if len(cc_data.cookie.shape)==2:
				# same-component cross-correlations
				ctype='sc'
			elif len(cc_data.cookie.shape)==3:
				# inter-component cross-correlations
				ctype='ic'

	#*******************************************************************************************************
	
	class MatrixForm:

		def __init__(self,rn_all,rx_act,ry_act,nrchosen,rnchosen):

			global selec_data, actdrp, actnrp
			
			# nrchosen -> no. of receivers chosen
			# rnchosen -> receiver (station) numbers of chosen receivers (stations)

			self.act_dist_rp=np.zeros((nrchosen,nrchosen))
			# self.act_dist_rp -> actual_distance_receiver_pairs. The word "actual" is used in the name so as to distinguish
			# these distances from the "effective" ones which are computed using the appoximate reciever locations
			# on the uniform grid of h13.
			act_rnum_rp=np.zeros((nrchosen,nrchosen), dtype=object)
			# act_rnum_rp -> similar to self.act_dist_rp but for receiver-pair nos. rather than distances

			selec_data=np.zeros((orig_data.shape[0], nrchosen, nrchosen))
	
			# NB: in h13, the synthetic ccs computed directly are those corresponding to the LOWER TRIANGULAR part 
			# of the cross-correlation matrix, i.e. Cjk is computed, where j>k. In case of the stored data, what is
			# stored is the UPPER TRIANGULAR part, i.e. Ckj is stored. These two are of course flipped versions of each
			# other, but for consistency we actually pass the lower-triangular part to h13. The upper triangular part is
			# computed within h13 by flipping.

			nrtotal=allstored.size

			for b,brec in enumerate(rnchosen[:-1]):
				rbefore=list(allstored).index(brec)
				pbefore=rbefore*(2*(nrtotal-1)-(rbefore-1))/2	# sum of terms of an AP
				urecs=rnchosen[b+1:]
				crbr = map(lambda x: pbefore - 1 + (x-brec), urecs)
				# crbr is Columns_Relevant_to_Base_Receiver
				
				#print "Receiver ", brec
				#print "rbefore and pbefore: ", rbefore, pbefore
				#print "Selecting columns: ", crbr
			
				# for consistency with h13, convert UPPER TRIANGULAR cc-s to LOWER TRIANGULAR
				selec_data[:,b+1:,b]=np.flipud(orig_data[:,crbr])

				# compute pairwise distances
				x1=rx_act[np.searchsorted(rn_all,brec)]
				y1=ry_act[np.searchsorted(rn_all,brec)]
				x2=rx_act[np.searchsorted(rn_all,urecs)]
				y2=ry_act[np.searchsorted(rn_all,urecs)]
				self.act_dist_rp[b+1:,b]=np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
				self.act_dist_rp[b,b+1:]=self.act_dist_rp[b+1:,b]
				act_rnum_rp[b:,b]=map(lambda x: '%d-%d' %(x,brec), rnchosen[b:])
				act_rnum_rp[b,b:]=map(lambda x: '%d-%d' %(brec,x), rnchosen[b:])

			actdrp = self.act_dist_rp
			actnrp = act_rnum_rp
	
	#********************************************************************************************************

	class Process:

		def __init__(self,wsp,num_chosen):

			nyq = 1/(2*si)
			nmul = 5
			new_nyq = min(nyq,nmul*fpband[1]) # lower bound for the Nyquist frequency we now seek
			if new_nyq < nyq:
				# determine the extent of possible downsampling
				om=int(np.round(np.log10(si)))
				dt_rough = np.round(1./(2*new_nyq), abs(om))
				# we want to ensure that record length is unchanged by downsamlping. so the new
				# sampling interval must be a factor of "reclen" as well as a multiple of "si"
				trial = dt_rough
				# NB: avoid the perils of floats and the modulo operator!!!
				msi = int(si*(10**(abs(om))))
				mtrial = int(trial*(10**(abs(om))))
				mreclen = int(reclen*(10**(abs(om))))
				while (mreclen % mtrial != 0) or (mtrial % msi != 0):
					print trial
					trial -= 10**(om)
					mtrial = int(trial*(10**(abs(om))))
				
				self.dfac=mtrial/msi
			else:
				self.dfac=1

			self.dt = si*self.dfac
			self.downsample()
			self.taper(wsp,num_chosen)

		def downsample(self):
			dsd=selec_data[::self.dfac,:,:]	# dsd -> down_sampled_data
			self.ds_ccl=ccl[::self.dfac]
			print "\n\nData downsampled by a factor of ", self.dfac

			# ensure compatibility of number of samples
			self.nsam=(orig_data.shape[0] - 1)/self.dfac #+1
			lsi = -(dsd.shape[0] - self.nsam) if (dsd.shape[0] - self.nsam) > 0 else None
			self.use_data = dsd[:lsi,:,:]
			
			if abs(self.ds_ccl.size - self.nsam)>0:
				self.ds_ccl=self.ds_ccl[:-1]
			
			if (self.ds_ccl.size != self.nsam) or (self.use_data.shape[0] != self.nsam):
				raise SystemExit("Discrepancy in number of samples after downsampling")
			else:
				print "\n\nSignal parameters after downsampling: "
				print "Number of samples: ", (orig_data.shape[0] - 1)/self.dfac
				print "Sample spacing (time domain): ", self.dt
				print "Peak frequency (Hz): ", (fpband[0]+fpband[1])/2

		def taper(self,wspeed,nrchosen):

			#cslow=1.5
			#cfast=6.0
			#tstart=actdrp/cfast
			#tend=actdrp/cslow

			lefw = -4.0
			rigw = +4.0
			tstart=actdrp/wspeed + lefw
			tstart[tstart<0] = 0.
			tend=actdrp/wspeed + rigw
			
			tap_mid=False
			# If True, the complete cc is tapered not only at the ends but also in the middle. This is
			# achieved by individually tapering the positive and negative branches on the "start" side.
			# In DHG_2019, we kept this parameter set to False, meaning that the entire cc is 
			# treated as one waveform, not separated into positive and negative branches
	
			pb_iws=np.searchsorted(self.ds_ccl,tstart)
			pb_iwe=np.searchsorted(self.ds_ccl,tend)
			nb_iws=np.searchsorted(self.ds_ccl,-tend)-1
			nb_iwe=np.searchsorted(self.ds_ccl,-tstart)-1
			# p/nb_iws/e stands for positive/negative_branch_index_of_window_start/end

			#self.snr=np.nan*np.zeros((nrchosen,nrchosen))
			self.snr=1e3*np.ones((nrchosen,nrchosen))
	
			hnsam = (self.nsam)/2 if (self.nsam)%2==0 else ((self.nsam)-1)/2
			# hnsam -> half_the_number_of_samples
			cchlen = reclen/2
			# cchlen -> cross-correlation_half_length
			if tap_mid:
				tap_portion=np.round(tstart/cchlen,2)
			else:
				tap_portion=np.round((cchlen - tend)/reclen,2)

			for i in range(nrchosen-1):
				for j in range(i+1,nrchosen):
					#****************** first compute the S/N ratio ******************
					ud=self.use_data[:,j,i]
					out_win=np.copy(ud)
					ns_ow = hnsam - (pb_iwe[j,i] + 1 - pb_iws[j,i])
					# ns_ow -> number-of_samples_outside_window

					# positive branch window
					pbwin=ud[pb_iws[j,i]:pb_iwe[j,i]+1]
					out_win[pb_iws[j,i]:pb_iwe[j,i]+1]=0.0
					egy_win_pb=np.mean(np.square(pbwin))
					egy_owin_pb=np.sum(np.square(out_win[hnsam+1:]))/(ns_ow)
					self.snr[j,i] = egy_win_pb/egy_owin_pb

					# negative branch window
					nbwin=ud[nb_iws[j,i]:nb_iwe[j,i]+1]
					out_win[nb_iws[j,i]:nb_iwe[j,i]+1]=0.0
					egy_win_nb=np.mean(np.square(nbwin))
					egy_owin_nb=np.sum(np.square(out_win[:hnsam]))/(ns_ow)
					self.snr[i,j] = egy_win_nb/egy_owin_nb
					# negative branch of [p,q] is positive branch of [q,p] cross-correlation

					#**************** then do the tapering ***************************
					# NB: obspy operations are performed in-place. So when you taper a
					# trace, even the NumPy array which forms the tr.data gets tapered!!!
					if tap_mid:
					# tapering the positive and negative branches individually
						branches = np.array_split(ud,2)
						tr_nb=oc.trace.Trace(data=branches[0], header={'delta': self.dt})
						tr_pb=oc.trace.Trace(data=branches[1], header={'delta': self.dt})
						tr_nb.taper(max_percentage=tap_portion[j,i],side='right')
						tr_pb.taper(max_percentage=tap_portion[j,i],side='left')

					tr=oc.trace.Trace(data=ud, header={'delta': self.dt})
					tr.taper(max_percentage=tap_portion[j,i])

#***************************************************************************************************************

	class Errors:

		def __init__(self,num_chosen,stepaz):
		
			self.DelE = np.zeros((num_chosen,num_chosen))
			# in h13 we only need one (triangular) half of this matrix to cover ALL receiver pairs
			# However, we fill both halves here to account for the positive and negative branches
			# lower triangular half of DelE -> \Delta E for positive branch of h13-relevant waveform
			# upper triangular half of DelE -> \Delta E for negative branch of h13-relevant waveform

			swpmax=180

			if len(orig_data.shape)==2:
				# same-component cross-correlations
				ctype='sc'
			elif len(orig_data.shape)==3:
				# inter-component cross-correlations
				ctype='ic'

			for az in np.arange(0,swpmax,stepaz):
				saefo = azan.do_single_azimuth(orig_data,allstored,si,ccl,ctype,fpband,az,stepaz,(False,False,False,True))
				# saefo -> single_azimuth_energy_fitting_object
				for k,actrp in enumerate(saefo.azrp):
					h13_ind = np.where(actnrp==actrp)
					if len(h13_ind[0])>0:
						#print k, actrp, h13_ind
						self.DelE[h13_ind] = saefo.res_ep_pb[k]
						self.DelE[h13_ind[::-1]] = saefo.res_ep_nb[k]
					else:
						pass

#######################################################################################################################################

class post_run():
	
	def __init__(self,ws,osmd,oica,schar,ring_rad,ring_w,dowhat):
		
		self.wspeed = ws
		self.osmd = osmd
		self.oica = oica
		self.schar = schar
		self.ring_rad = ring_rad
		self.ring_gw = ring_w

		if dowhat==0:
			# save the results
			self.save_pickle()
		elif dowhat==1:
			# produce plots showing inversion setup
			self.make_plots()

	def save_pickle(self):

		jarname="result_anseicca.pckl" # center frequency and nrecs can be in the name
		archname="waveforms_anseicca"
		jarfile=os.path.join(os.getcwd(),jarname)
		archfile=os.path.join(os.getcwd(),archname)
		win_lr_np = np.stack((self.oica.negl,self.oica.negr,self.oica.posl,self.oica.posr),axis=0)
		try:
			np.savez_compressed(archfile, t=self.oica.t, wsp=self.wspeed, wobs=self.oica.obscross, wsyn_i=self.oica.allit_syncross[0], wsyn_f=self.oica.allit_syncross[-1], win_ind=win_lr_np)
		except AttributeError:
			np.savez_compressed(archfile, t=self.oica.t, wsp=self.wspeed, wobs=self.oica.obscross, wsyn_i=self.oica.flit_syncross[0], wsyn_f=self.oica.flit_syncross[-1], win_ind=win_lr_np)

		jar=gzip.open(jarfile,'w')
		pickle.dump(reald,jar)
		pickle.dump(self.osmd.rnchosen,jar)
		pickle.dump(self.osmd.rchosenx_igp,jar)
		pickle.dump(self.osmd.rchoseny_igp,jar)
		pickle.dump(self.schar,jar)
		pickle.dump(self.oica.fhz,jar)
		pickle.dump(self.oica.pss,jar)
		pickle.dump(self.oica.dx,jar)
		pickle.dump(self.oica.gx,jar)
		pickle.dump(self.oica.gy,jar)
		pickle.dump(self.ring_rad,jar)
		pickle.dump(self.ring_gw,jar)
		pickle.dump(self.oica.mfit_kern_pos,jar)
		pickle.dump(self.oica.mfit_kern_neg,jar)
		pickle.dump(self.oica.gamma,jar)
		pickle.dump(self.oica.allit_mc,jar)
		pickle.dump(self.oica.allit_misfit,jar)
		pickle.dump(self.oica.flit_indmis_p,jar)
		pickle.dump(self.oica.flit_indmis_n,jar)
		if not reald:
			#pickle.dump(self.oica.mc_true,jar)
			pickle.dump(self.oica.distribs_true,jar)
		jar.close()
			
		#plt.tight_layout() # use this when axis labels are off the plot

	def make_plots(self):
		
		if reald:
			plt.scatter(self.oica.alldist,self.oica.egy_obs)
			plt.plot(self.oica.alldist,self.oica.oef)	
		else:
			def plot_mod(inmod,ptitle):
				fig=plt.figure()
				axm=fig.add_subplot(111)
				axm.set_title(ptitle)
				cax=axm.pcolor(self.oica.gx,self.oica.gy,inmod,cmap=plt.cm.jet,vmin=mod_min,vmax=mod_max)
				#cax=axm.pcolor(self.oica.gx,self.oica.gy,inmod,cmap=plt.cm.jet)
				axm.plot(self.oica.dx*self.osmd.rchosenx_igp, self.oica.dx*self.osmd.rchoseny_igp, 'wd', markerfacecolor="None")
				axm.set_xlabel("km", fontsize=14)
				axm.set_ylabel("km", fontsize=14)
				axm.tick_params(axis='both', labelsize=14)
				for j in range(len(self.osmd.rchosenx_igp)):
					axm.annotate(j, xy=(self.oica.dx*self.osmd.rchosenx_igp[j],self.oica.dx*self.osmd.rchoseny_igp[j]), color='white')
				plt.colorbar(cax,ax=axm)

			mod_min=min(np.amin(self.oica.distribs_true),np.amin(self.oica.distribs_start))
			mod_max=max(np.amax(self.oica.distribs_true),np.amax(self.oica.distribs_start))
			plot_mod(self.oica.distribs_true,"True Model")
			plot_mod(self.oica.distribs_start,"Starting model")
