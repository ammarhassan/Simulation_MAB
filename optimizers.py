import math
import numpy as np
from random import sample, random
from scipy.stats import lognorm
import os
from scipy.optimize import newton

def gradiant_descent_oneD(R, C, gradiant_func, parameter_upper, parameter_lower, step_size, stop_threshold):
	parameter = parameter_lower + random() * (parameter_upper - parameter_lower)
	parameter_l = parameter + stop_threshold
	while abs(parameter_l - parameter) > stop_threshold:
		parameter_l = parameter
		gradiant = gradiant_func(parameter, R, C)
		parameter = parameter - step_size * gradiant
		print gradiant

	return parameter

def decayFuncGradiant(parameter, R, C):
	time_ = np.arange(1, R.shape[0]+1)
	G = np.diag(time_)
	parameter_mat = np.diag((1/parameter)**(time_))
	parameter_mat2 = np.diag((1/parameter)**(2*time_))
	t1 = np.dot(np.dot(np.dot(np.transpose(R),G), parameter_mat2), R)
	# t2 = np.dot(t1, parameter_mat2)
	# t3 = np.dot(t2, R)
	gradiant = 2 * (1.0/parameter) * (t1 - np.dot(np.dot(np.dot(np.transpose(C),G),parameter_mat),R))
	return gradiant

def objectiveFunction(x, R, C):
	time_ = np.arange(1, R.shape[0]+1)
	parameter_mat = np.diag(x**time_)
	return np.linalg.norm(R  - parameter_mat*C, 2)**2


def applyNewtons(a):pass

if __name__ == '__main__':
	gamma = .999

	C = np.transpose(np.array([random() for _ in range(1000)]))
	R = np.transpose(np.array([gamma**i * x for i, x in enumerate(C)]))

	print newton(objectiveFunction, .9, args=(R, C), fprime=decayFuncGradiant)

	# print gradiant_descent_oneD(R, C, decayFuncGradiant, 1, 0, .001, .001)

	# parameter = .9
	# time_ = np.arange(1, R.shape[0]+1)
	# G = np.diag(time_)
	# parameter_mat = np.diag((1/parameter)**(time_))
	# parameter_mat2 = np.diag((1/parameter)**(2*time_))
	# t1 = np.dot(np.dot(np.dot(np.transpose(R),G), parameter_mat2), R)
	# # t2 = np.dot(t1, parameter_mat2)
	# # t3 = np.dot(t2, R)
	# gradiant = 2 * (1.0/parameter) * (t1 - np.dot(np.dot(np.dot(np.transpose(C),G),parameter_mat),R))
	# print "gradiant", gradiant



