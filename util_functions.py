from collections import Counter
from math import log
import numpy as np
from random import *


class Stats():
	def __init__(self):
		self.accesses = 0.0 # times the article was chosen to be presented as the best articles
		self.clicks = 0.0 	# of times the article was actually clicked by the user
		self.CTR = 0.0 		# ctr as calculated by the updateCTR function

	def updateCTR(self):
		try:
			self.CTR = self.clicks / self.accesses
		except ZeroDivisionError: # if it has not been accessed
			self.CTR = 0
		return self.CTR

	def addrecord(self, click):
		self.clicks += click
		self.accesses += 1
		self.updateCTR()


def calculateEntropy(array):
	counts = 1.0* np.array(map(lambda x: x[1], Counter(array).items()))
	counts = counts / sum(counts)
	entropy = sum([x*log(x) for x in counts])
	return entropy

def gaussianFeature(dimension, argv ):
	mean= argv['mean'] if 'mean' in argv else 0
	std= argv['std'] if 'std' in argv else 1

	mean_vector = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(np.zeros(dimension), stdev)
	# print "vector", vector,

	l2_norm = np.linalg.norm(vector, ord=2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		"This makes it uniform over the circular range"
		vector = (vector / l2_norm)
		vector = vector * (random())
		vector = vector * argv['l2_limit']

		# print "l2_limit", vector,

	if mean is not 0:
		vector = vector + mean_vector
		# print "meanShifted",vector

	return vector

def featureUniform(dimension, argv):

	vector = np.array([random() for _ in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord=2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		while np.linalg.norm(vector, ord=2) > argv['l2_limit']:
			vector = np.array([random() for _ in range(dimension)])

	return vector


		
def simulateArticlePool(self, n_articles):
	def getEndTimes():
		pool = range(20)
		endTimes = [0 for i in startTimes]
		last = 20
		for i in range(1, self.iterations / article_life+1):
			chosen = sample(pool, 5)
			for c in chosen:
				endTimes[c] = article_life * i
			pool = [x for x in pool if x not in chosen]
			pool += [x for x in range(last,last+5) if x < len(startTimes)]
			last += 5
			if last > len(startTimes):
				break
		for p in pool:
			endTimes[p] = self.iterations
		return endTimes

	articles_id = self.createIds(n_articles)
	if n_articles >= 20:
		article_life = int(self.iterations / ((n_articles-20)*1.0/5))
		print article_life
		startTimes = [0 for x in range(20)] + [
			(1+ int(i/5))*article_life for i in range(n_articles - 20)]
		endTimes = getEndTimes()
	else:
		startTimes = [0 for x in range(n_articles)]
		endTimes = [self.iterations for x in range(n_articles)]

	print startTimes, endTimes
	for key, st, ed in zip(articles_id, startTimes, endTimes):
		self.articles.append(Article(key, st, ed, self.featureUniform()))
		self.articles[-1].theta = self.featureUniform()
