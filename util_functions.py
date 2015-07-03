from collections import Counter
from math import log
import numpy as np
from random import *
from custom_errors import FileExists


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


class batchAlgorithmStats():
	def __init__(self):
		self.stats = Stats()
		self.clickArray = []
		self.accessArray = []
		self.CTRArray = []
		self.time_ = []
		self.poolMSE = []
		self.articlesCTR = {}
		self.articlesPicked_temp = []
		self.entropy = []
		self.regret = []
		self.countNewArticles = 0
		self.countNewArticlesArray = []
		self.regret = []
		self.iterationRegret = 0
		self.bestRewardError = []
		self.iterationbestRewardError=0
		self.chosenRewardError = []
		self.iterationChosenRewardError=0
		self.chosenReward = []
		self.iterationChosenReward = 0
		self.chosenActionPredReward = []
		self.iterationChosenPredReward = 0
		self.bestReward = []
		self.iterationbestReward = 0
		self.bestActionPredReward = []
		self.iterationbestPredReward = 0

	def addRecord(self,iter_,poolMSE,poolArticles):
		self.clickArray.append(self.stats.clicks)
		self.accessArray.append(self.stats.accesses)
		self.CTRArray.append(self.stats.CTR)
		self.time_.append(iter_)
		self.poolMSE.append(poolMSE)
		for x in poolArticles:
			if x in self.articlesCTR:
				self.articlesCTR[x].append(poolArticles[x])
			else:
				self.articlesCTR[x] = [poolArticles[x]]
		self.entropy.append(calculateEntropy(self.articlesPicked_temp))
		self.articlesPicked_temp = []
		self.countNewArticlesArray.append(self.countNewArticles)
		self.countNewArticles = 0	

		self.regret.append(self.iterationRegret)
		self.iterationRegret = 0

		self.bestRewardError.append(self.iterationbestRewardError)
		self.iterationbestRewardError = 0

		self.chosenRewardError.append(self.iterationChosenRewardError)
		self.iterationChosenRewardError = 0

		self.chosenReward.append(self.iterationChosenReward)
		self.iterationChosenReward = 0

		self.chosenActionPredReward.append(self.iterationChosenPredReward)
		self.iterationChosenPredReward = 0

		self.bestReward.append(self.iterationbestReward)
		self.iterationbestReward = 0

		self.bestActionPredReward.append(self.iterationbestPredReward)
		self.iterationbestPredReward = 0

	def iterationRecord(self, click, articlePicked, new=False, regret=0,bestReward=0,bestActionPredReward=0,chosenReward=0,chosenActionPredReward=0):
		self.stats.addrecord(click)
		self.articlesPicked_temp.append(articlePicked)
		self.countNewArticles += new
		self.iterationRegret += regret
		self.iterationbestRewardError += bestReward-bestActionPredReward
		self.iterationbestReward += bestReward
		self.iterationbestPredReward += bestActionPredReward

		self.iterationChosenRewardError += chosenReward- chosenActionPredReward
		self.iterationChosenReward += chosenReward
		self.iterationChosenPredReward += chosenActionPredReward




	def plotArticle(self, article_id):
		plot(self.time_, self.articlesCTR[article_id])
		xlabel("Iterations")
		ylabel("CTR")
		title("")

class SpecialStats(Stats):
	def __init__(self):
		super(SpecialStats, self).__init__()
		self.pickedArticle = []

	def addRecord(self, ):
		pass

def calculateEntropy(array):
	counts = 1.0* np.array(map(lambda x: x[1], Counter(array).items()))
	counts = counts / sum(counts)
	entropy = sum([-x*log(x) for x in counts])
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


def condition_bound(x, xlim, type_):
	return (x <= xlim or type_!="upper") and (x >= xlim or type_!="lower")

def featureUniform(dimension, argv=None):
	l2_limit = argv["l2_limit"] if "l2_limit" in argv else 1
	l2_type = argv["l2_type"] if "l2_type" in argv else "upper"
	
	max_limit = argv["max_limit"] if "max_limit" in argv else 1
	max_type = argv["max_type"] if "max_type" in argv else "upper"

	condition = lambda x, xlim, x_type, y, ylim, y_type: condition_bound(x, xlim, x_type) and condition_bound(y, ylim, y_type)

	vector = np.array([random() for _ in range(dimension)])
	l2 = np.linalg.norm(vector, ord=2)
	m = max(vector)
	
	while not condition(l2, l2_limit, l2_type, m, max_limit, max_type):
		vector = np.array([random() for _ in range(dimension)])
		l2 = np.linalg.norm(vector, ord=2)
		m = max(vector)

	# print l2, m, condition_bound(l2, l2_limit, l2_type), condition_bound(m, max_limit, max_type)
	return vector

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def checkFileExists(filename):
	try:
		with open(filename, 'r'):
			return 1
	except IOError:
		return 0

def fileOverWriteWarning(filename, force):
	if checkFileExists(filename):
		if force == True:
			print "Warning: Overwriting %s"%(filename)
		else:
			raise FileExists(filename)

def everyX(li, X, offSet=0):
	return [li[i*X+offSet] for i in range((len(li)-offSet)/X)]

if __name__ == '__main__':
	vector = featureUniform(5, argv={"l2_limit":.1, "l2_type":"upper"})
	assert np.linalg.norm(vector, 2) <= .5

	vector = featureUniform(5, argv={"l2_limit":.9, "l2_type":"lower"})
	assert np.linalg.norm(vector, 2) >= .5

	vector = featureUniform(5, argv={"max_limit":.3, "max_type":"upper"})
	assert max(vector) <= .5

	vector = featureUniform(5, argv={"max_limit":.7, "max_type":"lower"})
	assert max(vector) >= .5

	vector = featureUniform(5, argv={"max_limit":.3, "max_type":"upper", "l2_limit":1, "l2_type":"upper"})
	assert max(vector) <= .5

	vector = featureUniform(5, argv={"max_limit":.7, "max_type":"lower", "l2_limit":.1, "l2_type":"upper"})
	assert max(vector) >= .5


