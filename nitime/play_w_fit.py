#-----------------------------------------------------------------------------
# Test of Nitime Function Fitter
#-----------------------------------------------------------------------------

#Imports:
import numpy as np
import fit

class ErpSeries:
	def __init__(self, data, erp_length):
		self.data = data
		self.erp_length = erp_length
	
	def epoch( self, which ):
		return self.data[ which*self.erp_length:(which+1)*self.erp_length]

	def __getitem__( self, key ):
		return self.epoch(key) 

datarange= np.linspace(0,1)
params = [ 0.55, 0.15, 10, 0 ]
#data = gaussian_func( (0.55, 0.30, 10, 0), datarange )  + cos(datarange*12*pi)
#data = gaussian_func( params, datarange )  + np.cos(datarange*10.*np.pi)
data = fit.gaussian_func( params, datarange ) + np.random.normal(size= len(datarange) )

theFitter = fit.Fitter( fit.gaussian_func, fit.gaussian_init )
theFitter.dofit( data, datarange  )
theFitter.plotfit(plot_residuals=1)
title("Gaussian")

print theFitter.params

datarange_vm = linspace(0, 2.*np.pi, len(data) )
datarange_vm = linspace(0, 1.0, len(data) )
theFitter = fit.Fitter( fit.vonMises, fit.vonMises_initial )
theFitter.dofit( data, datarange_vm  )
theFitter.plotfit(plot_residuals=1)
title("vonMises")

print theFitter.params

params = [ 0.25, 0.20, 10, 0 ]
#data = gaussian_func( (0.55, 0.30, 10, 0), datarange )  + cos(datarange*12*pi)
#data = gaussian_func( params, datarange )  + np.cos(datarange*10.*np.pi)
data = fit.gaussian_func( params, datarange ) + np.random.normal(size= len(datarange) )

datarange_vm = linspace(0, 2.*np.pi, len(data) )
datarange_vm = linspace(0, 1.0, len(data) )
theFitter = fit.Fitter( fit.vonMises, fit.vonMises_initial )
theFitter.dofit( data, datarange_vm  )
theFitter.plotfit(plot_residuals=1)
title("vonMises")

print theFitter.params
