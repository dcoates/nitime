#-----------------------------------------------------------------------------
# Nitime Function Fitter
#-----------------------------------------------------------------------------

"""This class provides general-purpose function fitting capability.
"""

#Imports:
import numpy as np
import scipy
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot
from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts

# XXX: What should horizontal axis be: linspace(0,1), linspace(0, 2pi) (-pi, pi)
# XXX: ...or something else... arange(len(data))

from inspect import getargspec

# XXX-Helper: is there something already that provides this?
def maxi(seq): 
    """Return the value and position of the first maximum item of a sequence."""
    return max((x,i) for i,x in enumerate(seq))


def gaussian_func( (mu, sigma, A, C), x):
	return A*np.exp( - ((x-mu) **2) / (2*sigma**2) ) + C
  
def gaussian_init( data ):
	(maxv,maxvi) = maxi( data )

	datalen = len(data)
	mu_guess = maxi(data)[1]/float(datalen)
	A_guess = maxi(data)[0]
	# Count how many data items exceed 1/2 amplitude for HWHH
	sigma_guess = len( mlab.find(data > (A_guess/2.0) ))/float(datalen)/ (2.0 * np.sqrt(2.0*np.log(2.0)))
		
	params = (mu_guess, sigma_guess, A_guess, 0)
	return params

# XXX 
def gamma_guess( data ):
    """This function guesses the starting parameters for a gamma_hrf

    Parameters
    ----------

    data: a nitime UniformTimeSeries object

    Returns
    -------

    params: a tuple with (A_guess, tau_guess, n_guess,
    delta_guess,y_offset_guess) which are the guesses for a starting point for
    optimization based on the data.
    
    """
    defaults ={}
    spec = getargspec(tsa.gamma_hrf)
    for i in range(len(spec.defaults)):
        defaults.update({spec.args[-i]:spec.defaults[-i]})

    (maxv,maxvi) = maxi(data)
    A_guess = maxi(data)[0]
    delta_guess = -1*data.t0 + defaults['delta']
    datalen = len(data)
    n_guess = defaults['n']
    tau_guess = (maxi(data)[1]*data.sampling_interval)/float(2)
    y_offset_guess = np.mean(data.data[...,: np.abs(data.t0)]

    params = (A_guess, tau_guess,  n_guess, delta_guess, y_offset_guess)

    return params


class fit_objective: pass

class shifted_scaled_gaussian(fit_objective):
	def __init__( self ):
		return None

	def func( self, (mu, sigma, A, C), x):
		return A*exp( - ((x-mu) **2) / (2*sigma**2) ) + C
  
	def initparams(self, data):
		(maxv,maxvi) = maxi( data )

		datalen = len(data)
		mu_guess = maxi(data)[1]/float(datalen)
		A_guess = maxi(data)[0]
		# Count how many data items exceed 1/2 amplitude for HWHH
		sigma_guess = len( find(data > (A_guess/2.0) ))/float(datalen)/ (2.0 * sqrt(2.0*log(2.0)))
		
		params = (mu_guess, sigma_guess, A_guess)
		return params

def vonMises( (mu,sigma,A,C), x):

#function r=vonMises(T,Tpref,sigma,b)
#Evaluates the Von Mises distribution, given: 
#<T> From [0,2pi] - the point in x to be evaluated
#<Tpref> The mean of the distribution
#<sigma> The bandwidth of the distribution
#
#100907 ASR made it
#120909 DRC converted to Python

    kappa=1.0/sigma;

    # normalize entire curve so that y==C(+0) at x=mu
    normA = np.exp(kappa*1.0)/(scipy.special.i0(kappa))

    #return lambda x: C+A/normA*exp(kappa*cos(x-mu))/(2*numpy.pi*scipy.special.i0(kappa))
    return C+A/normA*np.exp(kappa*np.cos(x-mu))/(scipy.special.i0(kappa))
  
def vonMises_initial(data):
    (maxv,maxvi) = maxi( data )

    datalen = len(data)
    mu_guess = maxi(data)[1]/float(datalen)
    A_guess = maxi(data)[0]
    C_guess = min(data)
    sigma_guess = len( mlab.find(data > ( (A_guess-C_guess)/2.0+C_guess )) )/2.0

    params = (mu_guess, sigma_guess/float(datalen)/2.0, A_guess-C_guess, C_guess)
    return params

# This has to be global to be callable from leastsq, since can't make leastsq use callback
# with self as first parameter.
# So, use a global function, with self as an extra argument
# TODO: don't really need data as a parameter anymore
def sse( params, self,data ):
	self.params = params
	return self.sse() 

class Fitter:
	#def __init__( self, funcclass ):
		#self.fitfunc = funcclass.func
		#self.initialfunc = funcclass.init

	def __init__( self, fitfunc, initfunc ):
		self.fitfunc = fitfunc
		self.initfunc = initfunc
		self.params = None
		self.data = None

	def datalen( self ):
		return len(self.data)

	def sse( self ):
		val = sum(np.core.fromnumeric.ravel(
			self.fitfunc(self.params,self.datarange)-self.data)**2)
		return val

	def error( self ):
		val = sum(np.core.fromnumeric.ravel(
			self.fitfunc(self.params,self.datarange)-self.data)**1)
		return val

	def resids( self ):
		val = self.fitfunc(self.params,self.datarange)-self.data
		return val

	def evalfunc( self ):
		return self.fitfunc( self.params, self.datarange )

	def dofit( self, data, datarange ):
		#(fit_params, warnflag, allvecs) = 
		#(fit_params, ier ) = \
		#	scipy.optimize.leastsq(error, self.params,args=(self,data), \
		#	full_output=False )
		#fit_params = scipy.optimize.fmin(sse, self.params, args=[data])
		#return ( fopt, iter, funcalls, warnflag )
		#return fit_params, warnflag, allvecs

		self.data = data
		self.datarange = datarange
		self.params = self.initfunc(data)

		fit_params, fopt, iter, funcalls, warnflag  = \
			scipy.optimize.fmin(sse, self.params, args=(self, data), \
			full_output=True, disp=False, retall=False )
		self.params = fit_params
		
		print "fopt, iter, funcalls, warnflag:", fopt, iter, funcalls, warnflag

		return fit_params

	def plotfit( self, plot_residuals=0 ):
		pyplot.figure()
		pyplot.plot(self.datarange, self.data, "bo-", label="data")
		pyplot.plot(self.datarange, self.evalfunc(), "k--", label="fit")
		if plot_residuals:
			pyplot.plot(self.datarange, (self.data - self.evalfunc()), "g.-", label="residuals")
		pyplot.legend()
		pyplot.grid()
