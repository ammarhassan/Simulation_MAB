from collections import Counter
from math import log
import numpy as np
from random import *

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


def calculateEntropy(array):
	counts = 1.0* np.array(map(lambda x: x[1], Counter(array).items()))
	counts = counts / sum(counts)
	entropy = sum([x*log(x) for x in counts])
	return entropy

def gaussianFeature(dimension, scaled=False, mean=.2, std=.1):
	mean = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(mean, stdev)
	if scaled:
		return vector / np.linalg.norm(vector, ord=2)
	else: return vector

def featureUniform(dimension):
	feature = np.array([random() for _ in range(dimension)])
	return feature / sum(feature)