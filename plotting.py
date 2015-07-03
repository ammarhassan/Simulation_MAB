from matplotlib.pylab import *


class PlottingStruct:
	def __init__(self, title="", xlabel="", ylabel="", v_x_locs=None):
		self.xy = []
		self.title = title
		self.ylabel = ylabel
		self.xlabel = xlabel
		self.legends = []
		self.verticleLines = v_x_locs
		self.yErr = []
	
	def addxy(self, x, y, legend):
		dic = {"x":x, "y":y}
		self.legends.append(legend)
		self.xy.append(dic)

	def addDetails(self, xlabel="", ylabel="", title=""):
		self.title=title;self.ylabel=ylabel;self.xlabel=xlabel
	
	def addYErrors(self, x, y, yerr):
		self.yErr.append({"x":x,"y":y,"yerr":yerr})

def subPlottingFunction(plottingStructs, saveFileAs=None):

	numsubplots = len(plottingStructs)
	f, axarr = plt.subplots(numsubplots, sharex=True)
	if numsubplots==1:axarr=[axarr]
	for i, p in enumerate(plottingStructs):
		for d in p.xy:
			axarr[i].plot(d["x"], d["y"])
		axarr[i].set_title(p.title,fontsize=9)
		axarr[i].legend(p.legends, loc=1, prop={'size':6}, fancybox=True, framealpha=0.3)
		axarr[i].set_ylabel(p.ylabel, fontsize=9)
		if p.verticleLines:
			plotLines(axarr[i], p.verticleLines)
	
	for i, p in enumerate(plottingStructs):
		if len(p.yErr):
			for d in p.yErr:
				axarr[i].errorbar(d["x"], d["y"], yerr=d["yerr"], fmt='+')

	if saveFileAs:
		savefig(saveFileAs, format="pdf")

"Plot vertical lines at specific events"
def plotLines(axes_, xlocs):
	for xloc, color in xlocs:
		for x in xloc:
			xSet = [x for _ in range(31)]
			ymin, ymax = axes_.get_ylim()
			ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
			axes_.plot(xSet, ySet, color)


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	# example data
	x = np.arange(0.1, 4, 0.5)
	y = x

	# example variable error bar values
	yerr = 1*x
	xerr = 0.1 + yerr

	# First illustrate basic pyplot interface, using defaults where possible.
	# plt.figure()
	# plt.errorbar(x, y, xerr=0.2, yerr=0.4)
	# plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

	# Now switch to a more OO interface to exercise more features.
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
	ax = axs[0,0]
	ax.errorbar(x, y, yerr=yerr,)
	ax.set_title('Vert. symmetric')

	# With 4 subplots, reduce the number of axis ticks to avoid crowding.
	ax.locator_params(nbins=4)

	ax = axs[0,1]
	ax.errorbar(x, y, xerr=xerr, fmt='o')
	ax.set_title('Hor. symmetric')

	ax = axs[1,0]
	ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
	ax.set_title('H, V asymmetric')

	ax = axs[1,1]
	ax.set_yscale('log')
	# Here we have to be careful to keep all y values positive:
	ylower = np.maximum(1e-2, y - yerr)
	yerr_lower = y - ylower

	ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
	            fmt='o', ecolor='g', capthick=2)
	ax.set_title('Mixed sym., log y')

	fig.suptitle('Variable errorbars')

	plt.show()