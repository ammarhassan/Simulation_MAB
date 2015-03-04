import math
import numpy as np
from MAB_algorithms import *
import datetime
from matplotlib.pylab import *
from random import sample
from util_functions import calculateEntropy, featureUniform, gaussianFeature
from Articles import *
import json


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

	def addRecord(self, iter_, poolMSE, poolArticles):
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

	def iterationRecord(self, click, articlePicked):
		self.stats.addrecord(click)
		self.articlesPicked_temp.append(articlePicked)

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

class User():
	def __init__(self, id, featureVector=None):
		self.id = id
		self.featureVector = featureVector

class UserManager():
	def __init__(self, dimension, iterations, filename):
		self.userContexts = []
		self.dimension = dimension
		self.iterations = iterations
		self.filename = filename

	def simulateContextfromUsers(self, numUsers, featureFunction, **argv):
		"""users of all context arriving uniformly"""
		usersids = range(numUsers)
		users = []
		for key in usersids:
			users.append(User(key, featureFunction(self.dimension, argv=argv)))
		
		with open(self.filename, 'w') as f:
			for it in range(self.iterations):
				chosen = choice(users)
				f.write(json.dumps((chosen.id, chosen.featureVector.tolist()))+'\n')

	def randomContexts(self, featureFunction, **argv):
		with open(self.filename, 'w') as f:
			for it in range(self.iterations):
				f.write(json.dumps((0, featureFunction(self.dimension, argv=argv).tolist() ))+'\n')

	def contextsIterator(self):
		with open(self.filename, 'r') as f:
			for line in f:
				id, FV = json.loads(line)
				yield User(id, np.array(FV))
		

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles,
		noise=lambda x: np.random.normal(scale=x),
		userGenerator=None,
		type="ConstantTheta", environmentVars={}):
		self.dimension = dimension
		self.type = type
		self.iterations = iterations
		self.noise = noise

		self.alg_perf = {}
		self.iter_ = 0
		self.batchSize = 1000
		self.startTime = None

		self.articles = articles
		self.articlePool = {}
		"""the temp memory for storing the articles click from expectations
		for each iteration"""
		self.articlesPicked = [] 
		self.userGenerator = userGenerator

		self.reward_vector = {}

		self.environmentVars = environmentVars
		self.initiateEnvironment()

	def initiateEnvironment(self):
		if self.type=="evolveTheta":
			for x in self.articles:
				# "Find a random direction"
				x.testVars["deltaTheta"] = (featureUniform(self.dimension) - x.theta)
				# "Make the change vector of with stepSize norm"
				x.testVars["deltaTheta"] = x.testVars["deltaTheta"] / np.linalg.norm(x.testVars["deltaTheta"])*self.environmentVars["stepSize"]

	def regulateEnvironment(self):
		self.reward_vector = {}
		self.articlePool = self.articles
		if self.type=="abruptThetaChange":
			if self.iter_%self.environmentVars["reInitiate"]==0 and self.iter_>1:
				for x in self.articlePool:
					x.theta = gaussianFeature(self.dimension, scaled=True)
				print "Re-initiating parameters"
		elif self.type=="evolveTheta":
			for x in self.articlePool:
				x.theta += x.testVars["deltaTheta"]
				x.theta /= sum(x.theta)
				if any(x.theta < 0):
					print "negative detected; re-initiating theta"
					x.theta = gaussianFeature(self.dimension, scaled=True)
		elif self.type=="shrinkTheta":
			for x in self.articlePool:
				x.theta = np.dot(np.identity(self.dimension)*self.environmentVars['shrinker'], x.theta)

	def getUser(self):
		return self.userGenerator.next()

	def getClick(self, pickedArticle, userArrived):
		if pickedArticle.id not in self.reward_vector:
			clickExpectation = np.dot(pickedArticle.theta, userArrived.featureVector)
			if clickExpectation <1 and clickExpectation > 0:
				click = np.random.binomial(1, clickExpectation)
				self.reward_vector[pickedArticle.id] = click
			else:
				self.reward_vector[pickedArticle.id] = 1*(clickExpectation>0)
		return self.reward_vector[pickedArticle.id]

	def getReward(self, pickedArticle, userArrived):
		if pickedArticle.id not in self.reward_vector:
			reward = np.dot(pickedArticle.theta, userArrived.featureVector)
			self.reward_vector[pickedArticle.id] = reward + self.noise()
		return self.reward_vector[pickedArticle.id]		

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		for self.iter_ in xrange(self.iterations):
			"regulateEnvironment is essential; if removed, copy its code here"
			self.regulateEnvironment()
			userArrived = self.getUser()
			for alg_name, alg in algorithms.items():
				pickedArticle = alg.decide(self.articlePool, userArrived, self.iter_)
				click = self.getClick(pickedArticle, userArrived)
				alg.updateParameters(pickedArticle, userArrived, click, self.iter_)

				self.iterationRecord(alg_name, userArrived.id, click, pickedArticle.id)
			if self.iter_%self.batchSize==0 and self.iter_>1:
				self.batchRecord(algorithms)


	def iterationRecord(self, alg_name, user_id, click, article_id):
		if alg_name not in self.alg_perf:
			self.alg_perf[alg_name] = batchAlgorithmStats()
		self.alg_perf[alg_name].iterationRecord(click, article_id)

	def batchRecord(self, algorithms):
		for alg_name, alg in algorithms.items():
			poolArticlesCTR = dict([(x.id, alg.getarticleCTR(x.id)) for x in self.articlePool])
			if self.iter_%self.batchSize == 0:
				self.alg_perf[alg_name].addRecord(self.iter_, self.getPoolMSE(alg), poolArticlesCTR)

			for article in self.articlePool:
				article.addRecord(self.iter_, self.getArticleAbsDiff(alg, article), alg_name)

		print "Iteration %d"%self.iter_, "Pool ", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def analyzeExperiment(self, alpha, decay):
		"Plot vertical lines at specific events"
		def plotLines(axes_, xlocs):
			for xloc, color in xlocs:
				# axes = plt.gca()
				for x in xloc:
					xSet = [x for _ in range(31)]
					ymin, ymax = axes_.get_ylim()
					ySet = ymin + (np.array(range(0, 31))*1.0/30) * (ymax - ymin)
					axes_.plot(xSet, ySet, color)

		xlocs = [(list(set(map(lambda x: x.startTime, self.articles))), "black")]

		if self.type=="abruptThetaChange":
			ch_theta_loc = [i*self.environmentVars["reInitiate"] for i in range(self.iterations//self.environmentVars["reInitiate"])]
			env_name = "Abrupt-" + str(self.environmentVars["reInitiate"]//1000)+"k"
			xlocs.append((ch_theta_loc, "red"))
		elif self.type=="evolveTheta":
			env_name = "evolveTheta-"+str(self.environmentVars["stepSize"])
		else:
			env_name = "Const"

		sig_name = env_name+"_A"+str(len(self.articles))+"_It"+str(self.iterations//1000)+"k_alp-"+str(alpha)+"dec-"+str(decay)+".png"

		f, axarr = plt.subplots(3, sharex=True)
		for alg_name in self.alg_perf:
			axarr[0].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].CTRArray)
		
		# axarr[0].set_xlabel("Iteration")
		axarr[0].set_ylabel("Cumulative CTR")
		axarr[0].set_title("CTR Performance")
		plotLines(axarr[0], xlocs)
		axarr[0].legend(self.alg_perf.keys(), loc=2, prop={'size':8})

		for alg_name, record in self.alg_perf.items():
			poolBatchMSE = np.concatenate((np.array([record.poolMSE[0]]), np.diff(record.poolMSE)))
			axarr[1].plot(self.alg_perf[alg_name].time_, poolBatchMSE)
		axarr[1].set_ylabel("MSE")
		plotLines(axarr[1], xlocs)
		axarr[1].legend(self.alg_perf.keys(), loc=2, prop={'size':8})

		for alg_name, record in self.alg_perf.items():
			axarr[2].plot(self.alg_perf[alg_name].time_, record.entropy)
		axarr[2].set_xlabel("Iteration")
		axarr[2].set_ylabel("Entropy")
		plotLines(axarr[2], xlocs)
		axarr[2].legend(self.alg_perf.keys(), loc=2, prop={'size':8})

		savefig("SimulationResults/"+sig_name)

	def writeResults(self, alpha, decay):
		# write performance in csv file
		try:
			with open("SimulationResults/recordPerformance.csv", 'r') as f:
				pass
		except:
			with open("SimulationResults/recordPerformance.csv", 'w') as f:
				f.write("Algorithm, SimulationType, environmentVars, #Articles,#users,iterations,CTR\n")

		with open("SimulationResults/recordPerformance.csv", 'a') as f:
			for alg_name in self.alg_perf:
				res = [alg_name,
						self.type,
						';'.join([str(x)+'-'+str(y) for x,y in self.environmentVars.items()]),
						str(len(self.articles)),
						str(self.iterations),
						str(self.alg_perf[alg_name].CTRArray[-1]),
						str(alpha),
						str(decay),
						str(mean(self.alg_perf[alg_name].entropy)),
				]
				f.write('\n'+','.join(res))

	def getPoolMSE(self, alg):
		diff = 0
		for article in self.articlePool:
			diff += self.getArticleAbsDiff(alg, article)**2
		diff = math.sqrt(diff)
		return diff

	def getArticleAbsDiff(self, alg, article):
		return sum(map(abs, article.theta - alg.getLearntParams(article.id)))


if __name__ == '__main__':

	iterations = 10000
	dimension = 5
	alpha = .3
	decay = .99
	n_articles = 10
	n_users = 1000
	userFilename = "users.p"

	articleManager = ArticleManager(iterations, dimension)
	articles = articleManager.simulateArticlePool(n_articles=n_articles)

	UM = UserManager(dimension, iterations, userFilename)
	# UM.simulateContextfromUsers(1000,gaussianFeature, argv={"scaled":True, 'mean':0, 'std':.5})

	" articles can be saved by following instruction"
	#articleManager.saveArticles(articles, 'articles.p')
	" saved articles can be loaded by the following instruction"
	# articles = articleManager.loadArticles('articles.p')

	simExperiment = simulateOnlineData(articles=articles,
						userGenerator = UM.contextsIterator(),
						dimension=dimension,
						iterations=iterations,
						# type="abruptThetaChange",environmentVars={"reInitiate":100000},
						type="ConstantTheta",environmentVars={},
						# type="evolveTheta", environmentVars={"stepSize":.0000001},
					)

	LinUCB = LinUCBAlgorithm(dimension=dimension, alpha=alpha)
	decLinUCB = LinUCBAlgorithm(dimension=dimension, alpha=alpha, decay=decay)

	simExperiment.runAlgorithms({"LinUCB":LinUCB, "decLinUCB":decLinUCB})
	simExperiment.analyzeExperiment(alpha, decay)
	simExperiment.writeResults(alpha, decay)

