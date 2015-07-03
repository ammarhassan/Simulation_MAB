import math
import numpy as np
from MAB_algorithms import *
from exp3_MAB import Exp3Algorithm, UCB1Algorithm, Exp3QueueAlgorithm, Exp3Algorithm_Baised, EpsilonGreedyAlgorithm
import datetime
from random import sample
from scipy.stats import lognorm
from util_functions import *
from Articles import *
from Users import *
import json
from conf import sim_files_folder, result_folder
from operator import itemgetter
import os
from plotting import *



class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, userGenerator,
		batchSize=1000,
		noise=lambda : np.random.normal(scale=.001),
		type_="ConstantTheta", environmentVars=None,
		signature="",
		thetaFunc = None):

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
		self.thetaFunc = thetaFunc
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
		elif self.type=="ConstantTheta":
			for x in self.articles:
				x.initialTheta = x.theta
				print x.id, "true Theta", x.theta
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

		elif self.type=="shrinkOrd2_deterministic":
			for x in self.articles:
				# x.initialTheta = x.theta
				# x.testVars["shrinker"] = random()
				x.testVars["shrinker"] = np.diag(np.ones(self.dimension) *self.environmentVars["shrink"])
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
		elif self.type=="expShrink":
			if "gammas" in self.environmentVars:
				for x,gamma in zip(self.articles, self.environmentVars["gammas"]):
					x.gamma = gamma
					x.initialTheta = x.theta
			else:
				for x in self.articles:
					print x.gamma
					print x.theta
					x.initialTheta = x.theta


		sig_name = [("It",str(self.iterations//1000)+'k'), ("U",str(n_users))]
		sig_name = '_'.join(['-'.join(tup) for tup in sig_name])
		self.simulation_signature += '_' + env_sign + '_' + sig_name


	def regulateEnvironment(self):
		self.reward_vector = {}
		if self.type=="abruptThetaChange":
			if self.iter_%self.environmentVars["reInitiate"]==0 and self.iter_>1:
				for x in self.articlePool:
					x.theta = gaussianFeature(self.dimension, scaled=True)
				print "Re-initiating parameters"
		elif self.type=="ConstantTheta":
			for x in self.articlePool:
				x.theta = x.theta
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

		elif self.type=="shrinkOrd2" or self.type=="shrinkOrd2_deterministic":
			for x in self.articlePool:
				temp = np.identity(self.dimension) - (x.testVars['shrinker']*(self.iter_ - x.startTime)*1.0/self.iterations)
				x.theta = np.dot(temp, x.theta)
				# x.theta = np.dot(1-x.testVars['shrinker']*(self.iter_*1.0 - x.startTime)/self.iterations, x.theta)
		elif self.type=="popularityShift":
			for x in self.articlePool:
				x.theta = x.initialTheta * (x.testVars["popularity"][self.iter_ - x.startTime])
		elif self.type=="expShrink":
			for x in self.articlePool:
				x.theta = x.theta * x.gamma


	def getUser(self):
		return self.userGenerator.next()

	def updateArticlePool(self):
		min_limit = 0
		max_limit = 1
		if "bad" in self.environmentVars and len(self.articlePool):
			max_limit = min([min(x.theta) for x in self.articlePool])

		elif "good" in self.environmentVars and len(self.articlePool):
			min_limit = max([max(x.theta) for x in self.articlePool])

		self.articlePool = [x for x in self.articles if self.iter_ <= x.endTime and self.iter_ >= x.startTime]
		
		if self.iter_ > 0:
			for x in self.articlePool:
				if x.startTime==self.iter_:
					if "bad" in self.environmentVars:
						x.theta = np.array([random()*max_limit for _ in range(self.dimension)])
					elif "good" in self.environmentVars:
						x.theta = np.array([min_limit+random()*(1-min_limit) for _ in range(self.dimension)])

	def getClick(self, pickedArticle, userArrived):
		if pickedArticle.id not in self.reward_vector:
			clickExpectation = np.dot(pickedArticle.theta, userArrived.featureVector) + self.noise()
			if clickExpectation <1 and clickExpectation > 0:
				click = np.random.binomial(1, clickExpectation)
				self.reward_vector[pickedArticle.id] = click
			else:
				self.reward_vector[pickedArticle.id] = 1*(clickExpectation>0)
		return self.reward_vector[pickedArticle.id]

	def getReward(self, pickedArticle, userArrived):
		if pickedArticle.id not in self.reward_vector:
			reward = np.dot(pickedArticle.theta, userArrived.featureVector)
			self.reward_vector[pickedArticle.id] = reward + pickedArticle.gamma**(self.iter_ - pickedArticle.startTime) * self.noise()
		return self.reward_vector[pickedArticle.id]	

	def getPositiveReward(self, pickedArticle, userArrived): 
		reward = self.getReward(pickedArticle, userArrived)
		if reward < 0 : reward = 0
		return reward

	def getTrueBestArticle(self, user):
		reward = [(art, np.dot(art.theta,user.featureVector)) for art in self.articlePool]
		return max(reward, key=itemgetter(1))[0]


	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		for self.iter_ in xrange(self.iterations):
			"regulateEnvironment is essential; if removed, copy its code here"
			self.updateArticlePool()
			self.regulateEnvironment()
			userArrived = self.getUser()
			for alg_name, alg in algorithms.items():
				# print_ = self.iter_>0  and not(self.iter_%50)
				# if print_: print '\n', alg_name
				print_=False
				pickedArticle = alg.decide(self.articlePool, userArrived, self.iter_, print_=print_)
				click = self.getReward(pickedArticle, userArrived)
				# if click < 0 :
				# 	print "negative reward", click
				alg.updateParameters(pickedArticle, userArrived, click, self.iter_)

				if self.iter_>self.batchSize and self.iter_>len(self.articles):
					self.iterationRecord(alg, alg_name, userArrived, click, pickedArticle, self.getTrueBestArticle(userArrived))
			
			if self.iter_%self.batchSize==0 and self.iter_>self.batchSize and self.iter_>len(self.articles):
				self.batchRecord(algorithms)


	def iterationRecord(self, alg, alg_name, user, click, article, bestArticle):
		if alg_name not in self.alg_perf:
			self.alg_perf[alg_name] = batchAlgorithmStats()
		new = article.startTime == max([x.startTime for x in self.articlePool])
		if self.iter_ > len(self.articlePool):
			bestActionreward = np.dot(bestArticle.theta, user.featureVector)
			bestActionPredReward = alg.getPredictedReward(bestArticle.id)
			# bestActionrewardError =  bestActionPredReward - bestActionreward
			chosenReward = np.dot(article.theta, user.featureVector)
			chosenActionPredReward = alg.getPredictedReward(article.id)
			# chosenRewardError = chosenActionPredReward - chosenReward

		else: bestActionreward=0;bestActionPredReward=0;chosenReward=0;chosenActionPredReward=0;
		self.alg_perf[alg_name].iterationRecord(click, article.id, new, regret=bestActionreward - chosenReward, bestReward=bestActionreward,bestActionPredReward=bestActionPredReward,chosenReward=chosenReward,chosenActionPredReward=chosenActionPredReward)
		
		if alg_name in ["LinUCB", "tLinUCB", "tLinUCB_BC", "decLinUCB", "DoublingLinUCB"]:
			for x in self.articlePool:
				rewardTrue = np.dot(x.theta, user.featureVector)
				estimateReward = alg.getPredictedReward(x.id)
				var = alg.getConfidenceBound(x.id)
				# print alg_name, "estimateReward :", estimateReward
				x.addRecord(self.iter_, alg_name, rewardTrue=rewardTrue, estimateReward=float(estimateReward), var=float(var))

	def batchRecord(self, algorithms):
		for alg_name, alg in algorithms.items():
			poolArticlesCTR = dict([(x.id, alg.getarticleCTR(x.id)) for x in self.articlePool])
			if self.iter_%self.batchSize == 0:
				self.alg_perf[alg_name].addRecord(self.iter_, self.getPoolMSE(alg, alg_name), poolArticlesCTR)
		print "Iteration %d"%self.iter_, "Pool ", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime


	def analyzeExperiment(self, result_folder, alpha, decay, algorithms, ctrPlots, estimatePlots, regretDecisionPlots):
		xlocs = [(list(set(map(lambda x: x.startTime, self.articles))), "black")]
		sig_name = self.simulation_signature+"_alp-"+str(alpha)+"dec-"+str(decay)+"sig-"+str(noise)+".pdf"

		# if self.type=="abruptThetaChange":
		# 	ch_theta_loc = [i*self.environmentVars["reInitiate"] for i in range(self.iterations//self.environmentVars["reInitiate"])]
		# 	xlocs.append((ch_theta_loc, "red"))

		OraclePerfPlots = [PlottingStruct(), PlottingStruct(), PlottingStruct(), PlottingStruct()]
		MABPerfPlots = [PlottingStruct(), PlottingStruct(), PlottingStruct(), PlottingStruct()]
		MainRegretPlot = [PlottingStruct()]
		alg_names = ["UCB1","EXP3","egreedy","LinUCB","tLinUCB","tLinUCB_BC", "decLinUCB", "DoublingLinUCB"]
		# alg_names = ["LinUCB", "decLinUCB"]
		alg_names = [x for x in self.alg_perf if x in set(alg_names)]
		for alg_name in alg_names:
			"plot CTR"
			# axarr[0].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].CTRArray)

			"Plot batch level CTR"
			# batchCTR = getBatchStats(self.alg_perf[alg_name].clickArray)/getBatchStats(self.alg_perf[alg_name].accessArray)
			# axarr[1].plot(self.alg_perf[alg_name].time_, batchCTR)
			"Plot reward estimate error of best action"
			# axarr[0].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].bestRewardError)
			# legends[0].append(alg_name)
			"Plot regret"
			if alg_name in ctrPlots:
				MABPerfPlots[0].addxy(x=self.alg_perf[alg_name].time_,y=np.cumsum(self.alg_perf[alg_name].regret),legend=alg_name)
				OraclePerfPlots[0].addxy(x=self.alg_perf[alg_name].time_,y=np.cumsum(self.alg_perf[alg_name].regret),legend=alg_name)
			MainRegretPlot[0].addxy(x=self.alg_perf[alg_name].time_,y=np.cumsum(self.alg_perf[alg_name].regret),legend=alg_name)				
		MABPerfPlots[0].addDetails(ylabel="Regret", title="Regret")
		OraclePerfPlots[0].addDetails(ylabel="Regret", title="Regret")
		MainRegretPlot[0].addDetails(ylabel="Regret", title="Regret curves for different MABs")
			# axarr[0].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].chosenRewardError)
			# axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].countNewArticlesArray)
		"Plot beta and theta estimates"
		for alg_name in estimatePlots:
		
			thetaErrors = np.zeros(len(self.alg_perf[alg_name].time_))
			gammaErrors = np.zeros(len(self.alg_perf[alg_name].time_))
			estimationError = np.zeros(len(self.articles[0].time_[alg_name]))
			for x in self.articles:
				t1, t2 = getEstimateErrors(algorithms, alg_name, x, self.alg_perf[alg_name].time_)
				thetaErrors+=t1;gammaErrors+=t2;estimationError+=abs(np.array(x.estimateReward[alg_name])-np.array(x.rewardTrue[alg_name]))

			thetaErrors/=len(self.articles);gammaErrors/=len(self.articles);estimationError/=len(self.articles)

			OraclePerfPlots[1].addxy(self.alg_perf[alg_name].time_, thetaErrors, legend=alg_name)
			if alg_name=="tLinUCB":OraclePerfPlots[2].addxy(self.alg_perf[alg_name].time_, gammaErrors, legend=alg_name)
			OraclePerfPlots[3].addxy(self.articles[0].time_[alg_name], estimationError, legend=alg_name)
		
		OraclePerfPlots[1].addDetails(ylabel="Avg. Theta\n MSE", title="Theta MSE")
		OraclePerfPlots[2].addDetails(ylabel="Avg. Beta\nMSE", title="Beta MSE")
		OraclePerfPlots[3].addDetails(ylabel="True-Pred\nReward", title="Estimation Error")

			# axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].entropy)
		"plot best reward, best reward estimate, chosen reward and chosen reward estimate"
			# if alg_name=="tLinUCB":
			# 	axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].bestReward)
			# 	axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].bestActionPredReward)
			# 	axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].chosenReward)
			# 	axarr[2].plot(self.alg_perf[alg_name].time_, self.alg_perf[alg_name].chosenActionPredReward)
			# 	legends[2].extend(["best reward", "best reward estimate", "chosen reward", "chosen reward estimate"])
			# 	legends[2].extend(["best reward", "chosen reward"])

		"plot article Regret and decision criterion"
		ticksRatio = iterations / 50
		for i, alg_name in enumerate(regretDecisionPlots):
			for j, x in enumerate(getTopArticles(self.articles, alg_name)):
				time_ = range(len(x.estimateReward[alg_name]))
				MABPerfPlots[1+i].addxy(x=time_,y=x.estimateReward[alg_name],
					legend="article "+str(x.id) +" "+ alg_name+" est reward")
				yerr = [(tot - me)/2.0 for tot, me in zip(x.var[alg_name], x.estimateReward[alg_name])]
				y = [me + err for err, me in zip(yerr, x.estimateReward[alg_name])]
				MABPerfPlots[1+i].addYErrors(x=everyX(time_,ticksRatio,2*j+5),y=everyX(y,ticksRatio,2*j+5), yerr=everyX(yerr,ticksRatio,2*j+5))

				MABPerfPlots[1+i].addxy(x=time_, y=x.rewardTrue["LinUCB"],legend="article "+str(x.id)+" True Reward")

			MABPerfPlots[1+i].addDetails(ylabel="reward\n=beta*theta'*x", title=alg_name+" Plot batch reward")

		"plot Variance+Mean"
		for alg_name in regretDecisionPlots:
			for x in getTopArticles(self.articles, alg_name):
				MABPerfPlots[3].addxy(x=range(len(x.var[alg_name])), y=x.var[alg_name], legend="article "+str(x.id) +" "+ alg_name+" r + CI")

		MABPerfPlots[3].addDetails(ylabel="reward +\n alpha beta^t*\nsqrt{x' A^-1 x}", title="Plot of estimate_reward + CI")

		subPlottingFunction(plottingStructs=MABPerfPlots, saveFileAs=os.path.join(result_folder, "MABPerf"+sig_name))
		subPlottingFunction(plottingStructs=OraclePerfPlots, saveFileAs=os.path.join(result_folder, "Oracle"+sig_name))
		subPlottingFunction(plottingStructs=MainRegretPlot, saveFileAs=os.path.join(result_folder, "Regret"+sig_name))


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

	def getPoolMSE(self, alg, alg_name):
		diff = 0
		return diff

def getEstimateErrors(algorithms, alg_name, article, times):
	# print alg_name, article.initialTheta
	lrntParameters = algorithms[alg_name].getLearnParamsList(article.id)
	if alg_name in ["tLinUCB", "tLinUCB_BC", "decLinUCB"]:
		lrntThetas, lrntGammas = lrntParameters
	else:
		lrntThetas = lrntParameters
		lrntThetas = [np.ones(article.initialTheta.shape[0]) if x is None else x for x in lrntThetas]
		lrntGammas = [1 for i in range(times[-1]+1)]
	thetaErrors = [np.linalg.norm(th-article.initialTheta) for th in lrntThetas]
	gammaErrors = [np.linalg.norm(gm-article.gamma) for gm in lrntGammas]

	thetaErrors = np.array([thetaErrors[t] for t in times])
	gammaErrors = np.array([gammaErrors[t] for t in times])

	return thetaErrors, gammaErrors

def getTopArticles(articles, alg_name):
	if len(articles)>2:
		top2Articles = [max(articles, key=lambda x:x.rewardTrue[alg_name][-1]), max(articles, key=lambda x:x.rewardTrue[alg_name][0])]
	else:top2Articles = articles
	return top2Articles

if __name__ == '__main__':

	iterations = 5000
	dimension = 2
	alpha = 1
	talpha = 1

	n_articles = 20
	shrinks = [.005]
	poolArticles = [n_articles]
	articleInflux = 20
	n_users =1
	decay =.9
	batchSize = 10
	noises = [0.01]
	beta = None

	userFilename = os.path.join(sim_files_folder, "users+it-"+str(iterations)+"+dim-"+str(dimension)+".p")
	resultsFile = os.path.join(result_folder, "Results.csv")
	# sim_type = "ConstantTheta"

	"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	UM = UserManager(dimension, iterations, userFilename)
	UM.simulateContextfromUsers(n_users, featureUniform, argv={'l2_limit':1}, force=True)
	# UM.randomContexts(featureUniform, argv={"l2_limit":1})


	for p_art in poolArticles:

		articlesFilename = os.path.join(sim_files_folder, "articles"+str(n_articles)+"+AP-"+str(p_art)+"+influx"+str(articleInflux)+"+IT-"+str(iterations)+"+dim"+str(dimension)+".p")
		AM = ArticleManager(iterations, dimension, n_articles=n_articles, 
				poolArticles=p_art, thetaFunc=featureUniform,  argv={'l2_limit':1},
				influx=articleInflux)
		# articles = AM.simulateArticlePool(given_thetas=[np.array([0.6, 0.7]), np.array([ 0.5 ,0.5])])
		articles = AM.simulateArticlePool()
		AM.saveArticles(articles, articlesFilename, force=True)

		# print map(lambda x:x.startTime, articles), map(lambda x:x.endTime, articles)

		for noise in noises:
			UM = UserManager(dimension, iterations, userFilename)
			articles = AM.loadArticles(articlesFilename)

			simExperiment = simulateOnlineData(articles=articles,
								userGenerator = UM.contextsIterator(),
								dimension  = dimension,
								iterations = iterations,
								# noise = lambda : 0,
								noise = lambda : np.random.normal(0, noise),
								batchSize = batchSize,
								# type_ = "abruptThetaChange",environmentVars={"reInitiate":100000},
								# type_ = "ConstantTheta",environmentVars={"bad":False},
								# type_ = "evolveTheta", environmentVars={"stepSize":.0000001},
								# type_ = "shrinkTheta", environmentVars={"shrink":shrink},
								# type_ = "shrinkOrd2_deterministic", environmentVars={"shrink":shrink, "good":True},
								# type_ = "popularityShift", environmentVars={"sigmaL":1, "sigmaU":2},
								type_ = "expShrink", environmentVars={"gamms":[.9995, .99995]},
								signature = AM.signature,
								thetaFunc = featureUniform,
							)
			print "Starting for ", simExperiment.simulation_signature
			algorithms = {}
			# for decay in [.9, .99]:
			algorithms["decLinUCB"] = LinUCBAlgorithm(dimension=dimension, alpha=alpha, decay=.99)
			algorithms["LinUCB"] = LinUCBAlgorithm(dimension=dimension, alpha=alpha)
			algorithms["DoublingLinUCB"] = DoublingRestartLinUCBAlgorithm(dimension=dimension, alpha=alpha)
			ctrPlots = ["LinUCB", "decLinUCB", "DoublingLinUCB"]
			estimatePlots = ["LinUCB", "DoublingLinUCB"]
			regretDecisionPlots = ["LinUCB", "DoublingLinUCB"]
			# algorithms["UCB1"] = UCB1Algorithm(dimension=dimension)
			# algorithms["EXP3"] = Exp3Algorithm(dimension=dimension,gamma=.5)
			# algorithms["egreedy"] = EpsilonGreedyAlgorithm(dimension=dimension,epsilon=.1)
			# algorithms["decEXP3=.9"] = Exp3Algorithm(dimension=dimension, gamma=.5, decay = .9)
			# algorithms["EXP3Queue"] = Exp3QueueAlgorithm(dimension=dimension, gamma=.5)
			# algorithms["EXP3_Baised"] = Exp3Algorithm_Baised(dimension=dimension, gamma=.5)
			# algorithms["tLinUCB"] = TemporalLinUCBAlgorithm(dimension=dimension, alpha=talpha, beta=1)
			# algorithms["tLinUCB_BC"] = TemporalLinUCBAlgorithm(dimension=dimension, alpha=talpha, beta=None)


			simExperiment.runAlgorithms(algorithms)
			simExperiment.analyzeExperiment(result_folder, alpha, decay, algorithms, ctrPlots, estimatePlots, regretDecisionPlots)
			simExperiment.writeResults(resultsFile, alpha, decay, p_art)
