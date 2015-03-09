import cPickle
import numpy as np
from util_functions import calculateEntropy, featureUniform, gaussianFeature
from random import sample

class Article():
	def __init__(self, id, startTime, endTime, FV=None):
		self.id = id
		self.startTime = startTime
		self.endTime = endTime
		self.initialTheta = None
		self.theta = None
		self.featureVector = FV
		self.absDiff = {}
		self.time_ = {}
		self.testVars = {}

	def setTheta(self, theta):
		self.initialTheta = theta
		self.theta = theta

	def setDeltaTheta(self, finalTheta, total_iterations):
		self.deltaTheta = (finalTheta - self.initialTheta) / total_iterations

	def evolveThetaWTime(self):
		self.theta += self.deltaTheta

	def inPool(self, curr_time):
		return curr_time <= self.endTime and curr_time >= self.startTime

	def addRecord(self, time_, absDiff, alg_name):
		if alg_name in self.time_:
			self.time_[alg_name].append(time_)
		else:
			self.time_[alg_name] = [time_]

		if alg_name in self.absDiff:
			self.absDiff[alg_name].append(absDiff)
		else:
			self.absDiff[alg_name] = [absDiff]

	def plotAbsDiff(self):
		figure()
		for k in self.time_.keys():
			plot(self.time_[k], self.absDiff[k])
		legend(self.time_.keys(), loc=2)
		xlabel("iterations")
		ylabel("Abs Difference between Learnt and True parameters")
		title("Observing Learnt Parameter Difference")

class ArticleManager():
	def __init__(self, iterations, dimension):
		self.iterations = iterations
		self.dimension = dimension
		self.signature = ""

	def saveArticles(self, Articles, filename):
		with open(filename, 'w') as f:
			cPickle.dump(Articles, f)

	def loadArticles(self, filename):
		with open(filename, 'r') as f:
			return cPickle.load(f)

	def simulateArticlePool(self, n_articles, thetaFunc, argv, poolArticles=None):
		def getEndTimes():
			pool = range(poolArticles)
			endTimes = [0 for i in startTimes]
			last = poolArticles
			for i in range(1,intervals):
				chosen = sample(pool, 5)
				for c in chosen:
					endTimes[c] = intervalLength * i
				pool = [x for x in pool if x not in chosen]
				pool += [x for x in range(last,last+5)]
				last+=5
			for p in pool:
				endTimes[p] = self.iterations
			return endTimes

		self.signature = "A-"+str(n_articles)+"+TF-"+thetaFunc.__name__
		articles = []
		articles_id = range(n_articles)
		
		if poolArticles and poolArticles < n_articles:
			remainingArticles = n_articles - poolArticles
			intervals = remainingArticles / 5 + 1
			intervalLength = self.iterations / intervals
			startTimes = [0 for x in range(poolArticles)] + [
				(1+ int(i/5))*intervalLength for i in range(remainingArticles)]
			endTimes = getEndTimes()
		else:
			startTimes = [0 for x in range(n_articles)]
			endTimes = [self.iterations for x in range(n_articles)]

		for key, st, ed in zip(articles_id, startTimes, endTimes):
			articles.append(Article(key, st, ed, featureUniform(self.dimension, {})))
			articles[-1].theta = thetaFunc(self.dimension, argv=argv)
		return articles
