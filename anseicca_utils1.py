import numpy as np

###############################################################################################################
def read_station_file(st_file):

	""" Format of input file MUST be:
		COLUMNS(4): <Sl no.> <ID> <Easting(x)> <Northing(y)>
		ROWS(n + 1): one header line followed by n lines; n is no. of stations/receivers
	"""

	cfh=open(st_file,'r')
	
	cfh.readline()
	entire=cfh.readlines()
	try:
		st_no=map(lambda p: int(p.split()[0]), entire)
		st_id=map(lambda p: p.split()[1], entire)
		xr=np.array(map(lambda p: float(p.split()[2])/1e3, entire))
		yr=np.array(map(lambda p: float(p.split()[3])/1e3, entire))
	except IndexError:
		raise SystemExit("Problem reading %s. Check file format." %(st_file))

	cfh.close()
	del cfh
	return st_no, st_id, xr, yr

###############################################################################################################

class SignalParameters():
		""" defines the signal characteristics used for modelling sources (and cross-correlations) as
		    well as, in case of real data, the processing parameters applied to the primary data before
		    cross-correlation.
		    Attributes: 
				dt: temporal sampling interval (seconds),
				nsam: number of samples,
				cf: center (peak) frequency, (Hz)
				lf: low corner frequency (real data frequency passband),
				hf: high corner frequency (real data frequency passband),
				altukey: Tukey window alpha parameter (real data post-whitening filter).
		"""

###############################################################################################################

class somod:

	#ringg -> ring_of_gaussians (used by Datta_et_al, 2019)
	#rgring -> radially_gaussian_ring (used by Hanasoge, 2013)

	@staticmethod
	def ringg(ngp,dx,xall,yall,theta,rad,sigma_fac):
		ans=np.zeros((ngp,ngp))
		#rad=r*dx
			
		for j in range(ngp):
			x0 = rad*np.cos(theta)
			y0 = rad*np.sin(theta)
			ans[:,j] = np.exp( -((xall[j] - x0)**2 + (yall - y0)**2)/(sigma_fac*(dx**2)) )
			
		return ans

	#******************************************************************************************

	@staticmethod
	def rgring(dx,xall,yall,rad,mag1,mag2=None):
		#rad=100*dx
		r_ib = np.sqrt(xall[j]**2 + yall**2)
		if mag2 is None:
			ampl = mag1
		else:
			#if abs(xall[j])<10:
			#if xall[j]>-35 and xall[j]<-25:
			if xall[j]>-22 and xall[j]<-15:
				ampl = mag2
			else:
				ampl = mag1
		ans = ampl * ( np.exp( -(r_ib-rad)**2/(10*(dx**2))) )
		return ans

##########################################################################################
