import math
import numpy as np
from MAB_algorithms import *
from exp3_MAB import Exp3Algorithm, UCB1Algorithm, Exp3QueueAlgorithm
import datetime
from matplotlib.pylab import *
from random import sample
from scipy.stats import lognorm
from util_functions import *
from Articles import *
from Users import *
import json
from conf import sim_files_folder, result_folder
import os

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, userGenerator,
		batchSize=1000,
		noise=lambda : np.random.normal(scale=.001),
		type_="ConstantTheta", environmentVars=None,
		signature=""):

		self.simulation_signature = signature
		self.dimension = dimension
		self.type = type_
		self.environmentVars = environmentVars
		self.iterations = iterations
		self.noise = noise
		self.batchSize = batchSize
		self.iter_ = 0
		
		self.startTime = None

		self.articles = articles
		self.articlePool = {}
		"""the temp memory for storing the articles click from expectations for each iteration"""
		self.articlesPicked = [] 
		self.alg_perf = {}
		self.reward_vector = {}

		self.userGenerator = userGenerator
		self.initiateEnvironment()

	def initiateEnvironment(self):
		env_sign = self.type
		if self.type=="evolveTheta":
			for x in self.articles:
				"# Find a random direction"
				x.testVars["deltaTheta"] = (featureUniform(self.dimension) - x.theta)
				"# Make the change vector of with stepSize norm"
				x.testVars["deltaTheta"] = x.testVars["deltaTheta"] / np.linalg.norm(
					x.testVars["deltaTheta"])*self.environmentVars["stepSize"]
		elif self.type=="shrinkTheta":
			if "type" in self.environmentVars and self.environmentVars["type"]=="contextDependent":
				for x in self.articles:
					x.testVars["shrinker"] = np.diag(1 - featureUniform(self.dimension) * self.environmentVars["shrink"])
				env_sign += "+CD"
			else:
				for x in self.articles:
					x.testVars["shrinker"] = np.diag(1 - featureUniform(self.dimension) * random() *self.environmentVars["shrink"])
			env_sign += "+rate-" + str(self.environmentVars["shrink"])

		elif self.type=="shrinkOrd2":
			for x in self.articles:
				# x.initialTheta = x.theta
				# x.testVars["shrinker"] = random()
				x.testVars["shrinker"] = np.diag(np.ones(self.dimension) * random() *self.environmentVars["shrink"])
			env_sign += "+rate-" + str(self.environmentVars["shrink"])

		elif self.type=="abruptThetaChange":
			env_sign += 'reInit-' + str(self.environmentVars["reInitiate"]//1000)+'k'
		elif self.type=="popularityShift":
			env_sign += "+SL-"+str(self.environmentVars["sigmaL"])+"+SU-"+str(self.environmentVars["sigmaU"])
			for x in self.articles:
				x.initialTheta = x.theta
				sigma = self.environmentVars["sigmaL"] + random()*(self.environmentVars["sigmaU"] - self.environmentVars["sigmaL"])

				m = lognorm(s=[sigma], loc=0).pdf(
						[(i+1.0)/self.iterations for i in range(self.iterations)])
				x.testVars["popularity"] = m/max(m)


		sig_name = [("It",str(self.iterations//1000)+'k')]
		sig_name = '_'.join(['-'.join(tup) for tup in sig_name])
		self.simulation_signature += '_' + env_sign + '_' + sig_name


	def regulateEnvironment(self):
		self.reward_vector = {}
		self.articlePool = [x for x in self.articles if self.iter_ <= x.endTime and self.iter_ >= x.startTime]
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
				x.theta = np.dot(x.testVars['shrinker'], x.theta)

		elif self.type=="shrinkOrd2":
			for x in self.articlePool:
				temp = np.identity(self.dimension) - (x.testVars['shrinker']*(self.iter_ - x.startTime)*1.0/self.iterations)
				x.theta = np.dot(temp, x.theta)
				# x.theta = np.dot(1-x.testVars['shrinker']*(self.iter_*1.0 - x.startTime)/self.iterations, x.theta)
		elif self.type=="popularityShift":
			for x in self.articlePool:
				x.theta = x.initialTheta * (x.testVars["popularity"][self.iter_ - x.startTime])

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

	def getPositiveReward(self, pickedArticle, userArrived): 
		reward = self.getReward(pickedArticle, userArrived)
		if reward < 0 : reward = 0
		return reward

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		for self.iter_ in xrange(self.iterations):
			"regulateEnvironment is essential; if removed, copy its code here"
			self.regulateEnvironment()
			userArrived = self.getUser()
			for alg_name, alg in algorithms.items():
				pickedArticle = alg.decide(self.articlePool, userArrived, self.iter_)
				click = self.getReward(pickedArticle, userArrived)
				if click < 0 :
					print click
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

	def analyzeExperiment(self, result_folder, alpha, decay):
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
		sig_name = self.simulation_signature+"_alp-"+str(alpha)+"dec-"+str(decay)+".pdf"

		if self.type=="abruptThetaChange":
			ch_theta_loc = [i*self.environmentVars["reInitiate"] for i in range(self.iterations//self.environmentVars["reInitiate"])]
			xlocs.append((ch_theta_loc, "red"))


		f, axarr = plt.subplots(3, sharex=True)
		for alg_name in self.alg_perf:
			axarr[0].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].CTRArray)
			
			batchCTR = getBatchStats(self.alg_perf[alg_name].clickArray)/getBatchStats(self.alg_perf[alg_name].accessArray)
			axarr[1].plot(self.alg_perf[alg_name].time_, batchCTR)

			axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].entropy)

		# axarr[0].set_xlabel("Iteration")
		axarr[0].legend(self.alg_perf.keys(), loc=1, prop={'size':6}, fancybox=True, framealpha=0.3)
		axarr[0].set_ylabel("Cumulative reward")
		axarr[0].set_title("Performance of MABs")

		axarr[1].set_ylabel("Batch Reward")
		axarr[1].legend(self.alg_perf.keys(), loc=1, prop={'size':6}, fancybox=True, framealpha=0.3)

		axarr[2].set_ylabel("Entropy of picked \n actions")

		plotLines(axarr[0], xlocs)
		plotLines(axarr[1], xlocs)
		plotLines(axarr[2], xlocs)		

		

		# axarr[1].legend(self.alg_perf.keys(), loc=2, prop={'size':8})

		
		savefig(os.path.join(result_folder, sig_name), format="pdf")

	def writeResults(self, filename, alpha, decay, numPool=None):
		# write performance in csv file
		try:
			with open(filename, 'r') as f:
				pass
		except:
			with open(filename, 'w') as f:
				f.write("Algorithm, SimulationType, environmentVars, #Articles,#users,iterations,CTR\n")

		with open(filename, 'a') as f:
			for alg_name in self.alg_perf:
				res = [alg_name,
						self.type,
						';'.join([str(x)+'-'+str(y) for x,y in self.environmentVars.items()]),
						str(len(self.articles)),
						str(self.iterations),
						str(self.alg_perf[alg_name].CTRArray[-1]),
						str(alpha),
						str(decay),
						str(numPool),
						str(mean(self.alg_perf[alg_name].entropy)),
						str(datetime.datetime.now()),
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
	# def constructSignature():
	# 	signature = AM.signature + 

	iterations = 30000
	dimension = 2
	alpha = .3

	n_articles = 50
	shrinks = [.001]
	poolArticles = [20]
	n_users = 100
	decay = .99
	batchSize = 100

	userFilename = os.path.join(sim_files_folder, "users"+str(iterations)+".p")
	resultsFile = os.path.join(result_folder, "Results.csv")
	# sim_type = "ConstantTheta"
	UM = UserManager(dimension, iterations, userFilename)
	# UM.randomContexts(featureUniform, argv={"l2_limit":1})
	

	for p_art in poolArticles:

		articlesFilename = os.path.join(sim_files_folder, "articles"+str(n_articles)+"+AP-"+str(p_art)+"+IT-"+str(iterations)+".p")
		AM = ArticleManager(iterations, dimension, n_articles=n_articles, 
				poolArticles=p_art, thetaFunc=featureUniform,  argv={'l2_limit':1})
		# articles = AM.simulateArticlePool()	
		# AM.saveArticles(articles, articlesFilename)	
		
		# print map(lambda x:x.startTime, articles), map(lambda x:x.endTime, articles)

		for shrink in shrinks:
			UM = UserManager(dimension, iterations, userFilename)
			articles = AM.loadArticles(articlesFilename)

			simExperiment = simulateOnlineData(articles=articles,
								userGenerator = UM.contextsIterator(),
								dimension  = dimension,
								iterations = iterations,
								noise = lambda : 0,
								batchSize = batchSize,
								# type_ = "abruptThetaChange",environmentVars={"reInitiate":100000},
								# type_ = "ConstantTheta",environmentVars={},
								# type_ = "evolveTheta", environmentVars={"stepSize":.0000001},
								# type_ = "shrinkTheta", environmentVars={"shrink":shrink},
								type_ = "shrinkOrd2", environmentVars={"shrink":shrink},
								# type_ = "popularityShift", environmentVars={"sigmaL":1, "sigmaU":2},
								signature = AM.signature,
							)
			print "Starting for ", simExperiment.simulation_signature
			algorithms = {}
			for decay in [.999]:
				algorithms["decLinUCB=" + str(decay)] = LinUCBAlgorithm(dimension=dimension, alpha=alpha, decay=decay)
			algorithms["LinUCB"] = LinUCBAlgorithm(dimension=dimension, alpha=alpha)
			algorithms["UCB1"] = UCB1Algorithm(dimension=dimension)
			algorithms["EXP3"] = Exp3Algorithm(dimension=dimension,gamma=.5)
			# algorithms["decEXP3=.9"] = Exp3Algorithm(dimension=dimension, gamma=.5, decay = .9)
			# algorithms["EXP3Queue"] = Exp3QueueAlgorithm(dimension=dimension, gamma=.5)

			simExperiment.runAlgorithms(algorithms)
			simExperiment.analyzeExperiment(result_folder, alpha, decay)
			simExperiment.writeResults(resultsFile, alpha, decay, p_art)

