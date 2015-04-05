import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter  	# for easiness in sorting and finding max and stuff
import datetime
import numpy as np 	# many operations are done in numpy as matrix inverse; for efficiency
from util_functions import Stats

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

			x_pta = self.articles[x.id].getProbability(user.featureVector, self.alpha)
			
			if minPTA < x_pta:
				articlePicked = x
				minPTA = x_pta
		if print_:
			for x in pool_articles:
				if x.id in self.articles:
					print x.startTime, "theta",map(pf,self.articles[x.id].theta), "AInv",map(pf,self.articles[x.id].A_inv[0,]), map(pf,self.articles[x.id].A_inv[1,]), 
					print "chosen", self.articles[x.id].learn_stats.accesses, "reward", pf(self.articles[x.id].learn_stats.clicks), "CTR", pf(self.articles[x.id].learn_stats.CTR)

		return articlePicked

	def updateParameters(self, pickedArticle, userArrived, click, time_):
		self.articles[pickedArticle.id].updateParameters(userArrived.featureVector, click)
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
		return 0
		# return self.articles[article_id].theta

	def getarticleCTR(self, article_id):
		return 0
		# return self.articles[article_id].learn_stats.CTR




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

	def reInitilize(self):
		d = np.shape(self.A)[0]				# as theta is re-initialized some part of the structures are set to zero
		self.A = np.identity(n=d)
		self.b = np.zeros(d)
		self.A_inv = np.identity(n=d)
		self.theta = np.zeros(d)
		self.pta = 0

	def updateParameters(self, featureVector, click):
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
		self.var = np.sqrt(np.dot(np.dot(featureVector,self.A_inv), featureVector))
		self.pta = self.mean + alpha * self.var
		return self.pta

