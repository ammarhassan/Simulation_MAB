import math
import numpy as np
from random import sample, random
from scipy.stats import lognorm
import os
from scipy.optimize import newton, brentq, minimize
from util_functions import featureUniform
from matplotlib.pylab import *
from conf import *


def Obj_constNoiseFit(gamma, R, X, time_):
	if type(gamma)==np.ndarray:
		gamma = gamma[0]
	R_hat = R*(gamma**(-time_))
	theta_hat = linearRegression(R_hat, X)
	C = np.dot(theta_hat, np.transpose(X))
	error = R_hat  - C
	error = sum(error**2)
	return log(error)

def Obj_constNoiseFitBaised(gamma, R, X, time_):
	if type(gamma)==np.ndarray:
		gamma = gamma[0]
	R_hat = R*(gamma**(-time_))
	theta_hat = linearRegression(R_hat, X)
	C = np.dot(theta_hat, np.transpose(X))
	error = R_hat  - C
	error = sum(error**2)
	return log(error) + 100 * gamma

def Obj_constNoiseFitBaised_Gradiant(gamma, R, X, time_):
	if type(gamma)==np.ndarray:
		gamma = gamma[0]
	R_hat = R*(gamma**(-time_))

	D = np.dot(np.transpose(X), X)
	A_inv = np.linalg.inv(D + np.identity(X.shape[1]))
	theta_hat = np.dot(np.dot(A_inv, np.transpose(X)), R_hat)
	C = np.dot(theta_hat, np.transpose(X))

	error = R_hat - C
	
	temp1 = np.dot(np.dot(np.dot(X, A_inv), np.transpose(X)), time_ * R_hat)

	gradiant = 2.0/gamma* sum(error * (-time_ * R_hat + temp1))/(sum(error*error))
	# print gradiant, len(time_)/2.0, sum(time_)/gamma
	gradiant = gradiant + 100

	return gradiant

def MultiObjective(gamma, R, X, time_):
	if type(gamma)==np.ndarray:
		gamma = gamma[0]
	# time_ = np.arange(R.shape[0])
	try:
		R_hat = R*(gamma**(-time_))
		theta_hat = linearRegression(R_hat, X)
		# print theta_hat.shape, X.shape
		C = np.dot(theta_hat, np.transpose(X))
		
		# print gamma
		error = R_hat  - C
		error = sum(error**2)
		# print error
		error = (len(time_)/2.0)*log(error) + sum(time_)*log(gamma)
	except RuntimeWarning:
		error = float('Inf')
	if math.isnan(error):error=float('Inf')
	return error

def MultiObjectiveGradiant(gamma, R, X, time_):
	if type(gamma)==np.ndarray:
		gamma = gamma[0]
	try:
		R_hat = R*(gamma**(-time_))

		D = np.dot(np.transpose(X), X)
		A_inv = np.linalg.inv(D + np.identity(X.shape[1]))
		theta_hat = np.dot(np.dot(A_inv, np.transpose(X)), R_hat)
		C = np.dot(theta_hat, np.transpose(X))

		error = R_hat - C
		
		temp1 = np.dot(np.dot(np.dot(X, A_inv), np.transpose(X)), time_ * R_hat)

		gradiant = 2.0/gamma* sum(error * (-time_ * R_hat + temp1))/(sum(error*error))
		# print gradiant, len(time_)/2.0, sum(time_)/gamma
		gradiant = (len(time_)/2.0) * gradiant + sum(time_) / gamma
	except RuntimeWarning:
		gradiant =0
	if math.isnan(gradiant):gradiant=0

	return gradiant


def linearRegression(R, X):
	D = np.dot(np.transpose(X), X)
	D_inv = np.linalg.inv(D + np.identity(X.shape[1]))
	theta_hat = np.dot(np.dot(D_inv, np.transpose(X)), R)
	return theta_hat

def evalFunc(start, end, points, func, X, R):
	step_size = abs(end - start) / points
	gammas = np.array([start + x*step_size for x in range(points)])
	loss = [func(g, R ,X, np.arange(R.shape[0])) for g in gammas]
	return gammas, loss

def plotObjectiveFunctions(start, end, points, trueGammas, noises, func, title_):
	theta = np.array([.1,.2])
	labels = []
	figure()
	for gamma in trueGammas:
		for noise in noises:
			print "beta",gamma, "noise", noise
			time_ = np.arange(data_lenght)
			X = np.array([featureUniform(dimension, argv={"l2_limit":1, "l2_type":"upper"}) for x in range(data_lenght)])

			R = (gamma**time_) * (np.dot(theta, np.transpose(X)) + np.random.normal(0,noise, (data_lenght)))

			gammas, loss = evalFunc(start, end, points, func, X, R)
			plot(gammas, loss)
			labels.append("Beta:"+str(gamma)+" Noise:"+str(noise))
	legend(labels)
	title(title_)
	xlabel("Beta estimation")
	ylabel("ObjectiveFunction")


def plot_gamma_estimates(X, noiseArray, time_ ,gammas, noises, modelType, theta, func, funcGrad):

	f, axarr = plt.subplots(len(gammas), sharex=True)
	f2, ax2 = plt.subplots(len(gammas), sharex=True)
	f3, ax3 = plt.subplots(len(gammas), sharex=True)

	iterations = [x*10 for x in range(1, (data_lenght-10)//10)]
	for ind, gamma in enumerate(gammas):
		true_gamma = [gamma for x in iterations]
		axarr[ind].plot(iterations, true_gamma)

		for noise in noises:
			# print "noise", noise, "gamma", gamma
			R = gamma**np.arange(data_lenght) * (np.dot(theta, np.transpose(X)) + noiseArray*math.sqrt(noise))
			est_gamma = []
			est_error = []
			theta_mse = []
			est_reward_error = []
			start = .8
			for index in iterations:
				res = minimize(fun=func, x0=start, args=(R[:index],X[:index,:], time_[:index]), method="Newton-CG", jac=funcGrad)
				estimated_gamma = res['x']
				if estimated_gamma > 1:
					estimated_gamma = [1]
				elif estimated_gamma < .1:
					estimated_gamma = [.1]
				est_gamma.append(estimated_gamma)
				start = est_gamma[-1]
				theta_hat = linearRegression(R[:index]*(start[0]**(-np.arange(index))) , X[:index,:])
				theta_mse.append(sum(abs(theta_hat - theta)))

				est_reward_error.append(estimatedRewardDiff(start, theta_hat,time_[index], X[index,:], R[index]))
			# est_reward_error = np.cumsum(est_reward_error)
			axarr[ind].plot(iterations, est_gamma)
			ax2[ind].plot(iterations, theta_mse)
			ax3[ind].plot(iterations, est_reward_error)


		axarr[ind].legend(["Truth"]+["Noise sigma:"+str(noise) for noise in noises], prop={'size':6})
		axarr[ind].set_ylabel("Beta")
		axarr[ind].set_title("True Beta:"+str(gamma))

		ax2[ind].legend(["Noise sigma:"+str(noise) for noise in noises], prop={'size':6})
		ax2[ind].set_ylabel("|theta-theta^|")
		ax2[ind].set_title("True Beta:"+str(gamma))

		ax3[ind].legend(["Noise sigma:"+str(noise) for noise in noises], prop={'size':6})
		ax3[ind].set_ylabel("Reward Est\n Error")
		ax3[ind].set_title("True Beta:"+str(gamma))
	
	axarr[-1].set_xlabel("Examples used for estimation")
	ax2[-1].set_xlabel("Examples used for estimation")
	ax3[-1].set_xlabel("Examples used for estimation")

	# f.savefig(os.path.join(result_folder, modelType+"estimatedError.pdf"), format="pdf")
	# f2.savefig(os.path.join(result_folder, modelType+"theta_convergence.pdf"), format="pdf")
	# f3.savefig(os.path.join(result_folder, modelType+"reward_estimation_error.pdf"), format="pdf")

	return sum(est_reward_error)


def calcVariance(gamma, R, X):
	time_ = np.arange(R.shape[0])
	R_hat = R*(gamma**(-time_))
	theta = linearRegression(R_hat, X)

	R_est = gamma**time_ * np.dot(X , theta)
	error = sqrt(sum((R_est - R)**2) / R.shape[0])
	return error

def estimatedRewardDiff(gamma_hat, theta_hat, time_, X, R):
	estimate = (gamma_hat**time_) * np.dot(X, theta_hat)
	est = estimate
	try:
		estimate[estimate>1] = np.array([1])
		estimate[estimate<0] = np.array([0])
	except IndexError:
		print "estimate", est, "beta", gamma_hat, "theta", theta_hat, "dot", np.dot(X, theta_hat)
	return sum(abs(estimate - R))

if __name__ == '__main__':
	gamma = 1
	dimension =2
	data_lenght = 500
	noise = 1


	noises = [.01]
	gammas = [.99,]
	repeat = 10

	# for noise in noises:
	# 	for gamma in gammas:
	# 		est_reward_error_var = []
	# 		est_reward_error_const = []
	# 		print noise, gamma
	# 		for i in range(repeat):

	# 			theta = featureUniform(dimension, argv={"l2_limit":1, "l2_type":"upper"})
				
	# 			# theta = np.array([.1,.2])
	# 			time_ = np.arange(data_lenght)
	# 			noiseArray = np.random.normal(0,1, (data_lenght))
	# 			X = np.array([featureUniform(dimension, argv={"l2_limit":1, "l2_type":"upper"}) for x in range(data_lenght)])
	# 			# R = (gamma**time_) * (np.dot(theta, np.transpose(X)) + noiseArray)

	# 			func = MultiObjective
	# 			funcGrad = MultiObjectiveGradiant
	# 			modelType = "decayingNoise_variableNoiseFit"
	# 			estimate_error = plot_gamma_estimates(X, noiseArray, time_,
	# 				gammas=[gamma], noises=[noise], 
	# 				modelType=modelType, theta=theta, func=func, funcGrad = funcGrad)
	# 			est_reward_error_var.append(estimate_error)
	# 			# func = functionGamma
	# 			# funcGrad = functionGammaGradiant
	# 			# modelType = "decayingNoise_HetroObj"
				
	# 			func = Obj_constNoiseFitBaised
	# 			funcGrad = Obj_constNoiseFitBaised_Gradiant
	# 			modelType = "decayingNoise_constantNoiseFit_Biased"
	# 			estimate_error = plot_gamma_estimates(X, noiseArray, time_,
	# 				gammas=[gamma], noises=[noise], 
	# 				modelType=modelType, theta=theta, func=func, funcGrad = funcGrad)
	# 			est_reward_error_const.append(estimate_error)

	# 		with open("estimate_reward_error.csv", 'a') as g:
	# 			g.write(','.join([str(x) for x in (["variable",gamma,noise]+est_reward_error_var)]) + '\n')
	# 			g.write(','.join([str(x) for x in (["constant",gamma,noise]+est_reward_error_const)]) + '\n')

	
	func = Obj_constNoiseFitBaised
	funcGrad = Obj_constNoiseFitBaised_Gradiant
	modelType = "constantNoise_HetroObj"


	theta = featureUniform(dimension, argv={"l2_limit":1, "l2_type":"upper"})
	time_ = np.arange(data_lenght)
	noiseArray = np.random.normal(0,1, (data_lenght))
	X = np.array([featureUniform(dimension, argv={"l2_limit":1, "l2_type":"upper"}) for x in range(data_lenght)])
	R = (gamma**time_) * (np.dot(theta, np.transpose(X)) + noiseArray)

	estimate_error = plot_gamma_estimates(X, noiseArray, time_,
		gammas=[.9, .99, 1], noises=[.1, .01], 
		modelType=modelType, theta=theta, func=func, funcGrad = funcGrad)

	start = .9
	end = 1.1
	points = 2000

	f, axarr = plt.subplots(1, sharex=True)
	gammas, loss = evalFunc(start, end, points, func, X, R)
	axarr.plot(gammas, loss, "b")

	# axarrT = axarr.twinx()
	# gammas, loss = evalFunc(start, end, points, funcGrad)
	# axarrT.plot(gammas, loss, "r")
	# axarrT.set_ylim([-2000, 2000])

	# print "appr gradiant", (func(.76, R, X, time_)- func(.75, R,X, time_))/.01
	# print "true gradiant", funcGrad(.755, R, X, time_)
	# func(.97, R, X, time_)
	# funcGrad(.97, R, X, time_)

	# print minimize(fun=func, x0=.7, args=(R,X,time_), method="Newton-CG", jac=funcGrad)

	

	""" Plot objective functions to gauge their robustness and behviour to noise"""
	# plotObjectiveFunctions(start, end, points,[.9,.99], [1,.01,.0001], func, "ConstantNoiseFit")
	# savefig("decayingNoise_VariableNoiseFit.pdf")


