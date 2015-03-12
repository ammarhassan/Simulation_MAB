import numpy as np
from util_functions import calculateEntropy, featureUniform, gaussianFeature, Stats, fileOverWriteWarning
import json


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
		self.signature = ""

	def simulateContextfromUsers(self, numUsers, featureFunction, argv):
		"""users of all context arriving uniformly"""
		usersids = range(numUsers)
		users = []
		for key in usersids:
			users.append(User(key, featureFunction(self.dimension, argv=argv)))
		
		with open(self.filename, 'w') as f:
			for it in range(self.iterations):
				chosen = choice(users)
				f.write(json.dumps((chosen.id, chosen.featureVector.tolist()))+'\n')

	def randomContexts(self, featureFunction, argv, force=False):

		fileOverWriteWarning(self.filename, force)

		self.signature = "U+FF-"+featureFunction.__name__
		with open(self.filename, 'w') as f:
			for it in range(self.iterations):
				f.write(json.dumps((0, featureFunction(self.dimension, argv=argv).tolist() ))+'\n')

	def contextsIterator(self):
		with open(self.filename, 'r') as f:
			self.userContexts = f.read().split('\n')
		for line in self.userContexts:
			id, FV = json.loads(line)
			yield User(id, np.array(FV))
		