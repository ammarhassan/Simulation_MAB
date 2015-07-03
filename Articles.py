import cPickle
import numpy as np
from util_functions import calculateEntropy, featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample

class Article():
	def __init__(self, id, startTime, endTime, FV=None):
		self.id = id
		self.startTime = startTime
		self.endTime = endTime
		self.initialTheta = None
		self.theta = None
		self.featureVector = FV
		self.time_ = {}
		self.testVars = {}
		self.lnorm = 0
		self.gamma = np.random.beta(100, 1)
		self.thetasDiff={}
		self.gammasDiff={}
		self.rewardTrue={}
		self.estimateReward={}
		self.var = {}

	def setTheta(self, theta):
		self.initialTheta = theta
		self.theta = theta

	def setDeltaTheta(self, finalTheta, total_iterations):
		self.deltaTheta = (finalTheta - self.initialTheta) / total_iterations

	def evolveThetaWTime(self):
		self.theta += self.deltaTheta

	def inPool(self, curr_time):
		return curr_time <= self.endTime and curr_time >= self.startTime

	def addRecord(self, time_, alg_name, thetaDiff=0, gammaDiff=0, rewardTrue=0, estimateReward=0, var=0):
		self.addToDictLists(alg_name, time_, self.time_)
		self.addToDictLists(alg_name, thetaDiff, self.thetasDiff)
		self.addToDictLists(alg_name, gammaDiff, self.gammasDiff)

		self.addToDictLists(alg_name, rewardTrue, self.rewardTrue)
		self.addToDictLists(alg_name, estimateReward, self.estimateReward)
		self.addToDictLists(alg_name, var+estimateReward, self.var)


	def addToDictLists(self, key, item, dic):
		if key in dic:
			dic[key].append(item)
		else:
			dic[key] = [item]

	def plotAbsDiff(self):
		figure()
		for k in self.time_.keys():
			plot(self.time_[k], self.absDiff[k])
		legend(self.time_.keys(), loc=2)
		xlabel("iterations")
		ylabel("Abs Difference between Learnt and True parameters")
		title("Observing Learnt Parameter Difference")

class ArticleManager():
	def __init__(self, iterations, dimension, thetaFunc, argv, n_articles, poolArticles, influx=5):
		self.iterations = iterations
		self.dimension = dimension
		self.signature = ""
		self.n_articles = n_articles
		self.poolArticles = poolArticles
		self.thetaFunc = thetaFunc
		self.argv = argv
		self.signature = "A-"+str(self.n_articles)+"+PA"+ str(self.poolArticles)+"+TF-"+self.thetaFunc.__name__
		self.influx = influx
		self.articles = []

	def saveArticles(self, Articles, filename, force=False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			cPickle.dump(Articles, f)

	def loadArticles(self, filename):
		with open(filename, 'r') as f:
			return cPickle.load(f)

	def simulateArticlePool(self, bad=False, given_thetas=None):
		def getEndTimes():
			pool = range(self.poolArticles)
			endTimes = [0 for i in startTimes]
			last = self.poolArticles
			for i in range(1,intervals):
				chosen = sample(pool, self.influx)
				for c in chosen:
					endTimes[c] = intervalLength * i
				pool = [x for x in pool if x not in chosen]
				pool += [x for x in range(last,last+self.influx)]
				last+=self.influx
			for p in pool:
				endTimes[p] = self.iterations
			return endTimes

		
		# articles = []
		articles_id = range(self.n_articles)
		
		if self.poolArticles and self.poolArticles < self.n_articles:
			remainingArticles = self.n_articles - self.poolArticles
			intervals = remainingArticles / self.influx + 1
			intervalLength = self.iterations / intervals
			startTimes = [0 for x in range(self.poolArticles)] + [
				(1+ int(i/self.influx))*intervalLength for i in range(remainingArticles)]
			endTimes = getEndTimes()
		else:
			startTimes = [0 for x in range(self.n_articles)]
			endTimes = [self.iterations for x in range(self.n_articles)]


		# min_lnorm = self.argv["l2_limit"] if "l2_limit" in self.argv else 1
		# lastest_article = 0
		"the code below assumes that article with same start dates are adjacent"
		for key, st, ed in zip(articles_id, startTimes, endTimes):
			self.articles.append(Article(key, st, ed, featureUniform(self.dimension, {})))
			
			self.articles[-1].theta = self.thetaFunc(self.dimension, argv=self.argv)
			self.articles[-1].lnorm = np.linalg.norm(self.articles[-1].theta, ord=2)

		if given_thetas:
			for x, theta in zip(self.articles, given_thetas):
				x.theta = theta

		# "if bad is selected and when the article has a different start time"
		# 	if bad and lastest_article is not st:
		# 		self.argv["l2_limit"] = min_lnorm
			
		# 	if min_lnorm > self.articles[-1].lnorm:
		# 		min_lnorm = self.articles[-1].lnorm

		return self.articles
