import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency
from util_functions import Stats
from optimizers import *
from math import log

def addToListIndex(list_, index, item):
	if len(list_)-1 < index:
		list_ = list_ + [item*0 for _ in range( index - (len(list_)-1) )]
	list_[index] = item
	return list_

def newInterval(time_):
	tj = floor(log(time_,2))
	tjminus1 = floor(log(time_-1,2))
	if tj == tjminus1:
		return 0
	return 1

class pf(float):
    def __repr__(self):
        return "%0.5f" % self

class LinUCBAlgorithm(object):
	def __init__(self, dimension, alpha, decay=None):
		self.articles = {}
		self.dimension = dimension
		self.alpha = alpha
		self.decay = decay
		self.last_iteration_time = 0

	def decide(self, pool_articles, user, time_, print_=False):
		minPTA = float("-inf")
		articlePicked = choice(pool_articles)
		for x in pool_articles:
			if x.id not in self.articles:
				self.articles[x.id] = LinUCBStruct(self.dimension, x.id, time_)
				return x
		for x in pool_articles:
			x_pta = self.articles[x.id].getProbability(user.featureVector, self.alpha)
			self.articles[x.id].storeLrntParams(time_)

			if minPTA < x_pta:
				articlePicked = x
				minPTA = x_pta
		if print_:
			for x in pool_articles:
				if x.id in self.articles:
					print x.id, "LinUCB theta",map(pf,self.articles[x.id].theta)#, "AInv",map(pf,self.articles[x.id].A_inv[0,]), map(pf,self.articles[x.id].A_inv[1,]), 
					# print "chosen", self.articles[x.id].learn_stats.accesses, "reward", pf(self.articles[x.id].learn_stats.clicks), "CTR", pf(self.articles[x.id].learn_stats.CTR)

		return articlePicked

	def updateParameters(self, pickedArticle, userArrived, click, time_):
		self.articles[pickedArticle.id].updateParameters(userArrived.featureVector, click, time_)
		self.articles[pickedArticle.id].learn_stats.addrecord(click)
		self.articles[pickedArticle.id].learn_stats.updateCTR()

		if self.decay:
			# each iteration is 1 second.
			self.applyDecayToAll(duration=time_ - self.last_iteration_time)
			self.last_iteration_time = time_

	def applyDecayToAll(self, duration):
		for key in self.articles:
			self.articles[key].applyDecay(self.decay, duration)
		return True

	def getLearntParams(self, article_id):
		# print "acc", self.articles[article_id].learn_stats.accesses
		if article_id in self.articles:
			return self.articles[article_id].theta
		else: return 0

	def getPredictedReward(self, article_id):
		return self.articles[article_id].getRewardEstimate()

	def getConfidenceBound(self, article_id):
		return self.articles[article_id].getConfidenceBound()

	def getarticleCTR(self, article_id):
		return 0
		# return self.articles[article_id].learn_stats.CTR

	def getLearnParamsList(self, article_id):
		return self.articles[article_id].getParamsList()

class DoublingRestartLinUCBAlgorithm(LinUCBAlgorithm):

	def updateParameters(self, pickedArticle, userArrived, click, time_):
		self.articles[pickedArticle.id].updateParameters(userArrived.featureVector, click, time_)
		self.articles[pickedArticle.id].learn_stats.addrecord(click)
		self.articles[pickedArticle.id].learn_stats.updateCTR()

		if time_ > 1000 and newInterval(time_):
			print time_, "reInitilize"
			for x in self.articles:
				self.articles[x].reInitilize()
			

class TemporalLinUCBAlgorithm(LinUCBAlgorithm):
	def __init__(self, dimension, alpha, beta=None):
		super(TemporalLinUCBAlgorithm, self).__init__(dimension, alpha, decay=None)
		self.beta = beta

	def decide(self, pool_articles, user, time_, print_=False):
		minPTA = float("-inf")
		articlePicked = choice(pool_articles)
		for x in pool_articles:
			if x.id not in self.articles:
				if self.beta:
					self.articles[x.id] = TemporalLinUCBStruct(self.dimension, x.id, time_, self.beta)
				else:
					self.articles[x.id] = TemporalLinUCBStruct(self.dimension, x.id, time_, x.gamma)
				return x
		for x in pool_articles:
			x_pta = self.articles[x.id].getProbability(user.featureVector, self.alpha, time_)
			self.articles[x.id].storeLrntParams(time_)
			if minPTA < x_pta:
				articlePicked = x
				minPTA = x_pta
		if print_:
			for x in pool_articles:
				if x.id in self.articles:
					print x.id,"TLinUCB theta",map(pf,self.articles[x.id].theta), "gamma", self.articles[x.id].gamma
					# print "chosen", self.articles[x.id].learn_stats.accesses, "reward", pf(self.articles[x.id].learn_stats.clicks), "CTR", pf(self.articles[x.id].learn_stats.CTR)

		return articlePicked
	def getLearntParams(self, article_id):
		# return 0
		# print "Lrnt gamma", self.articles[article_id].gamma, "acc", self.articles[article_id].learn_stats.accesses
		if article_id in self.articles:
			return self.articles[article_id].theta
		else: return 0


class LinUCBStruct:
	def __init__(self, d, id, tim = None):
		self.id = id
		self.A = np.identity(n=d) 			# as given in the pseudo-code in the paper
		self.b = np.zeros(d) 				# again the b vector from the paper 
		self.A_inv = np.identity(n=d)		# the inverse
		self.learn_stats = Stats()	# in paper the evaluation is done on two buckets; so the stats are saved for both of them separately; In this code I am not doing deployment, so the code learns on all examples
		self.deploy_stats = Stats()
		self.theta = np.zeros(d)			# the famous theta
		self.pta = 0 						# the probability of this article being chosen
		self.var = 0
		self.mean = 0
		self.DD = np.identity(n=d)*0
		self.identityMatrix = np.identity(n=d)
		self.startTime = tim
		self.thetas = []

	def reInitilize(self):
		d = np.shape(self.A)[0]				# as theta is re-initialized some part of the structures are set to zero
		self.A = np.identity(n=d)
		self.b = np.zeros(d)
		self.A_inv = np.identity(n=d)
		self.theta = np.zeros(d)
		self.pta = 0
		self.DD = np.identity(n=d)

	def updateParameters(self, featureVector, click, time_=None):
		self.DD +=np.outer(featureVector, featureVector)
		self.updateA()
		self.b += featureVector*click
		self.updateInv()
		self.updateTheta()

	def applyDecay(self, decay, duration):
		self.DD *= (decay**duration)
		self.b *= (decay**duration)
		
		self.updateA()
		self.updateInv()
		self.updateTheta()

	def updateInv(self):
		self.A_inv = np.linalg.inv(self.A)		# update the inverse

	def updateA(self):
		self.A = self.DD + self.identityMatrix

	def updateTheta(self):
		self.theta = np.dot(self.A_inv, self.b) # as good software code a function to update internal variables

	def getProbability(self, featureVector, alpha):
		self.mean = np.dot(self.theta, featureVector)
		self.var = alpha * np.sqrt(np.dot(np.dot(featureVector,self.A_inv), featureVector))
		self.pta = self.mean + self.var
		return self.pta

	def getRewardEstimate(self):
		return self.mean

	def getConfidenceBound(self):
		return self.var

	def storeLrntParams(self, time_):
		self.thetas = addToListIndex(self.thetas, time_, self.theta)

	def getParamsList(self):
		return self.thetas

class TemporalLinUCBStruct(LinUCBStruct):
	def __init__(self, d, id, tim, trueGamma=1):
		self.id = id
		self.learn_stats = Stats()	# in paper the evaluation is done on two buckets; so the stats are saved for both of them separately; In this code I am not doing deployment, so the code learns on all examples
		self.deploy_stats = Stats()
		self.theta = np.zeros(d)			# the famous theta
		self.shrinkedTheta = np.zeros(d)
		self.pta = 0 						# the probability of this article being chosen
		self.var = 0
		self.mean = 0
		self.startTime = tim
		self.timeArray = None

		self.gamma = 1
		self.trueGamma = trueGamma
		self.X = None
		self.R = None

		self.thetas=[]
		self.gammas=[]

	def updateParameters(self, featureVector, click, tim):
		if self.R == None:
			self.R = np.array([click])
			self.X = np.zeros(shape=(1, featureVector.shape[0]))
			self.X[0,:] = featureVector
			self.timeArray = np.array([0])
		else:
			self.X = np.vstack((self.X,featureVector))
			self.R = np.hstack((self.R,click))
			self.timeArray = np.hstack((self.timeArray, tim - self.startTime))

		# print "X, R, T", self.X.shape, self.R.shape, self.timeArray.shape
		
		# self.gamma = minimize(fun=Obj_constNoiseFitBaised, x0=self.gamma, args=(self.R,self.X, self.timeArray), method="Newton-CG", jac=Obj_constNoiseFitBaised_Gradiant)['x']
		self.gamma = minimize(bounds=(0.1,1), fun=MultiObjective, x0=self.gamma, args=(self.R,self.X, self.timeArray), method="Newton-CG", jac=MultiObjectiveGradiant)['x']
		if self.gamma>1:
			self.gamma=1
		elif self.gamma < .1:
			self.gamma = .1

		# print self.gamma, self.timeArray
		self.theta = linearRegression(self.R*(self.gamma**(-self.timeArray)), self.X)
		self.shrinkedTheta = self.theta*self.gamma**(tim - self.startTime)


	def getProbability(self, featureVector, alpha, tim):
		D = np.dot(np.transpose(self.X), self.X)
		A_inv = np.linalg.inv(D + np.identity(featureVector.shape[0]))

		self.mean = self.gamma**(tim - self.startTime) * np.dot(self.theta, featureVector)
		# if self.mean < 0: self.mean=0
		self.var = self.trueGamma**tim * alpha * np.sqrt(np.dot(np.dot(featureVector,A_inv), featureVector))
		self.pta = self.mean +  self.var
		return self.pta

	def getLearntParams(self, article_id):
		# return 0
		# print "Lrnt gamma", self.articles[article_id].gamma, "acc", self.articles[article_id].learn_stats.accesses
		if article_id in self.articles:
			return self.articles[article_id].theta,self.articles[article_id].gamma
		else: return (0,0)

	def storeLrntParams(self, time_):
		self.thetas = addToListIndex(self.thetas, time_, self.theta)
		self.gammas = addToListIndex(self.gammas, time_, self.gamma)

	def getParamsList(self):
		return self.thetas, self.gammas

if __name__ == '__main__':
	for t in range(2,100):
		print t, newInterval(t)
