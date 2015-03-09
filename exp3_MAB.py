#from conf import *
import time
import re
from random import random, choice
from operator import itemgetter
import datetime
import numpy as np
import math
import random

class Stats():
    def __init__(self):
        self.accesses = 0.0
        self.clicks = 0.0
        self.CTR = 0.0
    def updateCTR(self):
        try:
            self.CTR = self.clicks / self.accesses
        except ZeroDivisionError:
            self.CTR = 0
        return self.CTR
        
    def addrecord(self, click):
        self.clicks += click
        self.accesses +=1
        self.updateCTR()


class myQueue:  # Basic structure for popularity Queue
    def __init__(self):
        self.dic = {}
        self.QueueLength = len(self.dic)
    def push(self, articleID):
        if articleID in self.dic:
            self.dic[articleID] += 1
        else:
            self.dic[articleID] = 0
    def pop(self):
        removed = min(self.dic, key=self.dic.get)
        a = copy.copy(removed)
        del self.dic[removed]
        return a
    def decreaseAll(self):
        for article in self.dic:
            self.dic[article] = self.dic[article] - 0.05
    def initialize(self):
        self.dic = {}

class RandomStruct:
    def __init__(self, id):
        self.stats = Stats()
        self.id = id
        
class Exp3Struct:
    def __init__(self, gamma, id):
        self.id = id
        self.gamma = gamma
        self.weights = 1.0
        self.pta = 0.0
        self.stats = Stats()
    def reInitilize(self):
        self.weights = 1.0
        self.pta = 0.0        
    def updateWeight(self, n_arms, reward):
        X = reward / self.pta
        growth_factor = math.exp((self.gamma / n_arms)*X)
        self.weights = self.weights * growth_factor
    def updatePta(self, n_arms, total_weight):
        self.pta = (1-self.gamma) * (self.weights / total_weight)
        self.pta +=(self.gamma)*(1.0/float(n_arms))
    def applyDecay(self, decay, duration):
        self.weights *= (decay**duration)
        
class UCB1Struct:
    def __init__(self, id):
        self.id = id
        self.totalReward = 0
        self.numPlayed = 0
        self.pta = 0.0
        self.stats = Stats()
    def reInitilize(self):
        self.totalReward = 0.0
        self.numPlayed = 0.0
        self.pta = 0.0  
    def updateParameter(self, click):
        self.totalReward += click
        self.numPlayed +=1
        
    def updatePta(self, allNumPlayed):
        try:
            self.pta = self.totalReward / self.numPlayed + np.sqrt(2*np.log(allNumPlayed) / self.numPlayed)
        except ZeroDivisionError:
            self.pta = 0.0
        return self.pta
    def applyDecay(self, decay, duration):  # where to add decay
        self.totalReward *=(decay**duration)

class RandomAlgorithm:
    def __init__(self):
        self.articles = {}
    def decide(self, pool_articles): #praameter: article pool
        for x in pool_articles:
            if x.id not in pool_articles:
                self.articles[x.id] = RandomStruct(x.id)
        return choice(pool_articles)
    def updateParameters(self, pickedArticle, ArticleNum, click): # meaningless, just add this part to make it consistent 
        a = 1        
    def getarticleCTR(self, article_id):
        return self.articles[article_id].stats.CTR
        
class Exp3Algorithm:
    def __init__(self, gamma, decay = None, dimension = 5):
        self.articles = {}
        self.gamma = gamma
        self.decay = decay
        self.dimension = dimension
        self.ArticleNum = 0
    
    def decide(self, pool_articles, user, time_): #(paramters: article pool)
        "Should self.ArticleNum be total articles or total pool_articles?? Please correct the following line if its wrong."
        self.ArticleNum = len(pool_articles)
        r = random.random()
        cum_pta = 0.0        
        total_Weights = 0.0
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = Exp3Struct(self.gamma, x.id)
            total_Weights += self.articles[x.id].weights
        for x in pool_articles:
            self.articles[x.id].updatePta(len(pool_articles), total_Weights)
            cum_pta += self.articles[x.id].pta
            if cum_pta >r:
                return x
        return choice(pool_articles)
    # parameters : (pickedArticle, Nun of articles in article pool, click)
    def updateParameters(self, pickedArticle, userArrived, click, time_): 
        self.articles[pickedArticle.id].updateWeight(self.ArticleNum, click)
        if self.decay:
            self.applyDecayToAll(1)
    
    def applyDecayToAll(self, duration):
        for key in self.articles:
            self.articles[key].applyDecay(self.decay, duration)
    
    def getarticleCTR(self, article_id):
        return self.articles[article_id].stats.CTR

    def getLearntParams(self, article_id):
        return np.zeros(self.dimension)

class Exp3QueueAlgorithm:
    def __init__(self, gamma, decay = None, dimension=5):
        self.articles = {}
        self.gamma = gamma
        self.decay = decay
        self.dimension = dimension
        self.ArticleNum = 0
    
    def decide(self, pool_articles, user, time_):  #(paramters: article pool)
        self.ArticleNum = len(pool_articles)
        MyQ = myQueue()
        QueueSize = 15
        MyQ.decreaseAll()
        
        r = random.random()
        cum_pta = 0.0        
        total_Weights = 0.0
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = Exp3Struct(self.gamma, x.id)
            
            if MyQ.QueueLength < QueueSize:
                MyQ.push(x)
            elif x.id in MyQ.dic:
                MyQ.dic[x.id] += 1
            else:
                a=MyQ.pop()
                self.articles[a].reInitilize()
                MyQ.push(x.id)
                
            total_Weights += self.articles[x.id].weights
        for x in pool_articles:
            self.articles[x.id].updatePta(len(pool_articles), total_Weights)
            cum_pta += self.articles[x.id].pta
            if cum_pta >r:
                return x
        return choice(pool_articles)

    def updateParameters(self, pickedArticle, userArrived, click, time_):   # parameters : (pickedArticle, Nun of articles in article pool, click)
        self.articles[pickedArticle.id].updateWeight(self.ArticleNum, click)
        if self.decay:
            self.applyDecayToAll(1)
    
    def applyDecayToAll(self, duration):
        for key in self.articles:
            self.articles[key].applyDecay(self.decay, duration)
    
    def getarticleCTR(self, article_id):
        return self.articles[article_id].stats.CTR

    def getLearntParams(self, article_id):
        return np.zeros(self.dimension)


class UCB1Algorithm:
    def __init__(self, decay = None, dimension = 5):
        self.articles = {}
        self.decay = decay
        self.dimension = dimension
    def decide(self, pool_articles, user, time_): #parameters:(article pool, number of times that has been played)
        articlePicked = None
        for x in pool_articles:
            if x.id not in self.articles:
                self.articles[x.id] = UCB1Struct(x.id)
        
        allNumPlayed = sum([self.articles[x.id].numPlayed for x in pool_articles])

        for x in pool_articles:
            x_pta = self.articles[x.id].updatePta(allNumPlayed)
            
            if self.articles[x.id].numPlayed == 0:
                articlePicked = x
                return articlePicked
        return max(np.random.permutation([(x, self.articles[x.id].pta) for x in pool_articles]), key = itemgetter(1))[0]

            
    def updateParameters(self, pickedArticle, userArrived, click, time_):  #parameters: (pickedArticle, click)
        self.articles[pickedArticle.id].updateParameter( click)
        if self.decay:
            self.applyDecayToAll(1)
    def applyDecayToAll(self, duration):
        for key in self.articles:
            self.articles[key].applyDecay(self.decay, duration)
    def getarticleCTR(self, article_id):
        return self.articles[article_id].stats.CTR

    def getLearntParams(self, article_id):
        return np.zeros(self.dimension)
        
