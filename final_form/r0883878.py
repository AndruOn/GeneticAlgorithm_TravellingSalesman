
from numpy.random import randint
from numpy.random import random
from numpy.random.mtrand import beta
import Reporter
import numpy as np
import csv
from copy import deepcopy
from random import sample
from scipy.spatial import distance
import time



class r0883878:
	class Individual:

		def __init__(self, numberOfCities=None, perturbation_prob=None, 
			base_mutation_proba=None, base_crossover_proba=None, 
			init_k_selection=None, init_k_elimination=None):
			if (numberOfCities != None):
				self.path = np.random.permutation(np.arange(1,numberOfCities))
			if (base_mutation_proba != None):
				self.mutation_proba = self.floatToPercentage( base_mutation_proba + np.random.uniform(-perturbation_prob,perturbation_prob) )
			if (base_crossover_proba != None):
				self.crossover_proba = self.floatToPercentage( base_crossover_proba + np.random.uniform(-perturbation_prob,perturbation_prob) )
			if (init_k_selection != None):
				self.k_selection = self.kinInterval( init_k_selection + np.random.randint(0,4) - 2 )
			if (init_k_elimination != None):
				self.k_elimination = self.kinInterval( init_k_elimination + np.random.randint(0,4) - 2 )


		def cost(self, distanceMatrix) -> np.double:
			totalCost = distanceMatrix[0][self.path[0]]
			for i in range(len(self.path)-1):
				cost = distanceMatrix[self.path[i]][self.path[i+1]]
				if cost == np.inf:
					totalCost += 10**20
				else:
					totalCost += cost
			totalCost+= distanceMatrix[self.path[len(self.path)-1]][0]
			return totalCost

		def floatToPercentage(self,val) -> float:
			return min(1.0, max(0.0 , val))

		def kinInterval(selk, k):
			return min(10,max(k,1))

		def __str__(self) -> str:
			return "Individual: mutation:{mutation} crossover:{crossover} k_selection:{k_selection} k_elimination:{k_elimination} path:{path}".format(
				mutation = self.mutation_proba, crossover=self.crossover_proba, k_selection=self.k_selection, k_elimination=self.k_elimination, path=self.path)

		def __repr__(self) -> str:
			return "r0883878.Individual: mutation:{mutation} crossover:{crossover} k_selection:{k_selection} k_elimination:{k_elimination} path:{path}".format(
				mutation = self.mutation_proba, crossover=self.crossover_proba, k_selection=self.k_selection, k_elimination=self.k_elimination, path=self.path)


#-----------------SETUP--------------------------------------
	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.population = None
		self.distanceMatrix = None
		self.numberOfCities = None

	#Initializes the population
	def initialisation_random(self, perturbation_prob : float):
		self.population = np.ndarray(dtype=r0883878.r0883878.Individual ,shape=(self.populationsize))
		for i in range(self.populationsize):
			self.population[i] =  r0883878.Individual(self.numberOfCities, perturbation_prob, self.init_mutation_proba, self.init_crossover_proba,
									self.init_k_selection, self.init_k_elimination)
	def initialisation(self, perturbation_prob : float):
		if self.heuristicInit:
			return self.nearest_neighbour_heuristic_init(perturbation_prob)
		return self.initialisation_random(perturbation_prob)



#-----------------PRINTS--------------------------------------
	def str_param(self):
		return """
		populationsize = {_1}
		#selection & elimination
		init_k_selection = {_2}
		k_selection = {_21}
		init_k_elimination = {_4}
		k_selection = {_41}
		#mutation
		init_mutation_proba = {_5}
		percentageOfSwitches = {_3}
		numberOfSwitchies = {nbSwitchies}
		#crossover
		init_crossover_proba = {_6}

		iterations = {_7}
		genForConvergence = {_8}
		stoppingConvergenceSlope = {_9}
		numberOfCities = {_10}

		#diversity
		sigma = {sigma}
		alpha = {alpha}
		sharedCost_percentageOfSearchSpace = {percentSearchSpace}
		""".format(_1=self.populationsize,_2=self.init_k_selection,_21=self.k_selection,_3=self.percentageOfSwitches,_4=self.init_k_elimination,
		_41=self.k_elimination,_5=self. init_mutation_proba,_6=self.init_crossover_proba, _7=self.iterations,_8=self.genForConvergence,
		_9=self.stoppingConvergenceSlope,_10=self.numberOfCities, nbSwitchies= self.numberOfSwitches,
		sigma = self.sigma, alpha = self.sharedCost_alpha, percentSearchSpace = self.sharedCost_percentageOfSearchSpace)

	def printPopulation(self, population : np.array(Individual)):
		print("Population:[")
		for ind in population:
			print(ind,end=" ")
			print("cost:", ind.cost(self.distanceMatrix))
		print("]")

#-----------------SELECTION--------------------------------------
	def k_tournament(self, k : int, population):
		random_index_Sample = sample(range(population.size), k)
		costSample = np.ndarray(k)
		#print("random_index_Sample:",random_index_Sample)

		for i in range(k):
			sampledInd = population[ random_index_Sample[i] ]
			costSample[i] = sampledInd.cost(self.distanceMatrix)

		best_index = np.argmin(costSample)
		return population[random_index_Sample[best_index]]
	
	def get_k_selection(self, population) -> int:
		total_k = 0
		for ind in population:
			total_k += ind.k_selection
		#print("k_selection total_k:",total_k,"population.size:",population.size,"return:",int(total_k//population.size))
		return round(total_k/population.size)

#-----------------MUTATION--------------------------------------#https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
	def pmx(self, a, b, start, stop):#https://www.uio.no/studier/emner/matnat/ifi/INF3490/h16/exercises/inf3490-sol2.pdf
		child = [None]*len(a)
		# Copy a slice from first parent:
		child[start:stop] = a[start:stop]
		# Map the same slice in parent b to child using indices from parent a:
		for ind,x in enumerate(b[start:stop]):
			ind += start
			if x not in child:
				while child[ind] != None:
					#ind = b.index(a[ind])
					ind = np.where(b == a[ind])[0][0]
				child[ind] = x
			# Copy over the rest from parent b
		for ind,x in enumerate(child):
			if x == None:
				child[ind] = b[ind]
		return np.array(child)

	def pmx_pair(self, p1 : Individual, p2 : Individual):
		a = p1.path
		b = p2.path
		half = len(a) // 2
		start = randint(0, len(a)-half)
		stop = start + half
		off1 = r0883878.Individual()
		off2 = r0883878.Individual()
		off1.path , off2.path = ( self.pmx(a,b,start,stop) , self.pmx(b,a,start,stop) )
		(off1,off2) = self.combineSelfAdaptivity(p1,p2,off1,off2)
		return (off1,off2)

	def combineSelfAdaptivity(self, p1 : Individual, p2 : Individual, off1 : Individual, off2 : Individual):
		off1.mutation_proba, off2.mutation_proba = (self.combineProbMutation(p1.mutation_proba, p2.mutation_proba) , self.combineProbMutation(p1.mutation_proba, p2.mutation_proba))
		off1.crossover_proba, off2.crossover_proba = (self.combineProbCrossover(p1.crossover_proba, p2.crossover_proba) , self.combineProbCrossover(p1.crossover_proba, p2.crossover_proba))
		off1.k_selection, off2.k_selection = (self.combineK(p1.k_selection, p2.k_selection) , self.combineK(p1.k_selection, p2.k_selection))
		off1.k_elimination, off2.k_elimination = (self.combineK(p1.k_elimination, p2.k_elimination) , self.combineK(p1.k_elimination, p2.k_elimination))
		#print("new crossovered k_selection k1:",off1.k_selection,"k2:",off2.k_selection)
		#print("new crossovered k_elimination k1:",off1.k_elimination,"k2:",off2.k_elimination)
		return (off1,off2)

	def combineProbMutation(self, per1, per2):
		beta = 2*random() - 0.5
		newPercent = per1 + beta * (per2-per1)
		return min(1.0, max(self.min_mutation , newPercent))
	
	def combineProbCrossover(self, per1, per2):
		beta = 2*random() - 0.5
		newPercent = per1 + beta * (per2-per1)
		return min(1.0, max(self.min_crossover , newPercent))
	
	def combineK(self, k1, k2):
		#return round((k1+k2) /2)
		beta = 2*random() - 0.5
		newK = int( k1 + beta * min((k2-k1),1.0) )
		#print("combineK beta",beta,"return:",max(1, min(int(k1 + beta * (k2-k1)) , 10 )),"oldk1:",k1,"oldk2:",k2 )
		return max(self.min_k_value, min( newK, self.max_k_value )) 

	def mutation_randomSwaps(self, ind : Individual ,numberofswitches : int):
		path = ind.path
		numberOfCities = len(path)
		for i in range(numberofswitches):
			index1 = randint(numberOfCities-1)
			index2 = randint(numberOfCities-1)
			temp = path[index1]
			path[index1] = path[index2]
			path[index2] = temp

		self.mutationSelfAdaptivity(ind)

		return ind


	def mutationSelfAdaptivity(self, ind : Individual):
		#TODO check that passage per reference and dont need to return Individual
		ind.mutation_proba = self.mutate_probaMutation(ind.mutation_proba)
		ind.crossover_proba = self.mutate_probaCrossover(ind.crossover_proba)
		ind.k_selection = self.mutate_k(ind.k_selection)
		#print("mutate_k k_selection return", ind.k_selection)
		ind.k_elimination = self.mutate_k(ind.k_elimination)
		#print("mutate_k k_elimination return", ind.k_elimination)
	
	def mutate_probaMutation(self, percent : float) -> float:
		newPercent = percent + 0.1 * (random() - 0.5)
		return min(1.0, max(self.min_mutation , newPercent))

	def mutate_probaCrossover(self, percent : float) -> float:
		newPercent = percent + 0.1 * (random() - 0.5)
		return min(1.0, max(self.min_crossover , newPercent))
	
	def mutate_k(self, k : int):
		#return k
		beta = int((5*random()) - 2)
		#print("mutate_k beta:",beta,"oldK:",k)
		newK = max(self.min_k_value, min(self.max_k_value , k+beta))
		return newK

	#Hard mutation
	def mutation_scramble(self,individual : Individual):
		#scramble a subset of the path
		start = randint(self.numberOfCities)
		stop = randint(self.numberOfCities)
		individual.path[start:stop] = np.random.permutation(individual.path[start:stop])

		self.mutationSelfAdaptivity(individual)
		return individual

#-----------------ELIMINATION--------------------------------------
	def get_k_elimination(self, population : np.array(Individual)) -> int:
		total_k = 0
		for ind in population:
			total_k += ind.k_elimination
		return int(total_k//population.size)

	def elimination_kTournament(self, oldPopulation : np.array(Individual), NbSurvivors : int) -> np.array(Individual):
		newPopulation = np.ndarray(NbSurvivors,dtype=r0883878.Individual)
		for i in range(NbSurvivors):
			newPopulation[i] = self.k_tournament(self.k_elimination, oldPopulation)
		return newPopulation

#-----------------QUALITY--------------------------------------
	def getFitnessPopulation(self, population : np.array(Individual)) -> np.array(float):
		fitness = np.ndarray((population.size))
		for i in range(population.size):
			fitness[i] = population[i].cost(self.distanceMatrix)
		return fitness

	def accessQualityOfGeneration(self, population):
		fitness = self.getFitnessPopulation(population)

		meanObjective = np.mean(fitness)
		bestObjective = np.min(fitness)
		bestSolution = population[np.argmin(fitness)].path.tolist()
		bestSolution.append(0)
		return (meanObjective, bestObjective, np.array(bestSolution))

	def getWorstOnesIndex(self, population : np.array(Individual), NbWorsts : int ) -> np.array(Individual):
		fitness = self.getFitnessPopulation(population)
		#print("sorted fitness",np.sort(fitness[np.argsort(fitness)]))
		return np.argsort(fitness)[population.size - NbWorsts: ]

	def getRandomSubset(self, population : np.array(Individual), NbRandom : int ) -> np.array(Individual):
		return np.random.choice(population, size=NbRandom, replace=False)

#-----------------STOPPING CRITERIA--------------------------------------
	def stoppingCriteria(self, means, index):
		flag = True
		if index>self.genForConvergence:
			indexes = np.arange(float(self.genForConvergence))
			slope = np.polyfit(indexes, means, 1)
			if self.printEveryIter :
				print("slope:",slope[0]/np.mean(means))
			if abs(slope[0]/np.mean(means)) < self.stoppingConvergenceSlope:
				print("slope:",slope[0]/np.mean(means),"lastmeans:",means)
				return False
			return flag
		return flag

	def addNewMean(self, means, newMean):
		means = np.roll(means, 1)
		means[0] = newMean
		return means

	def getMeanMutation(self) -> float:
		mutationTotal = 0
		for ind in self.population:
			mutationTotal += ind.mutation_proba
		return mutationTotal/self.populationsize

	def getMeanCrossover(self) -> float:
		crossoverTotal = 0
		for ind in self.population:
			crossoverTotal += ind.crossover_proba
		return crossoverTotal/self.populationsize

#-----------------LOCAL SEARCH OPERATORS--------------------------------------	
	"""Interchange: two paths are neighbour is you can interchange any 2 nodes
	#and get the other path
	#STATS:
	#	finding the solution if 5steps from it: first:0.0-0.03 best:0.97-1
	# 	computation cost: really long computation O(n**2)"""
	def lsoInterchange(self, individual : Individual, firstOrBest : str) -> Individual:
		bestCost = individual.cost(self.distanceMatrix)
		bestPath = individual.path
		copyInd = deepcopy(individual)
		copyPath = copyInd.path
		individualPath = individual.path
		lengthOfPath = self.numberOfCities-1
		for i in range(lengthOfPath):
			for j in range(i+1,lengthOfPath):
                #Interchange
				copyPath[j] = individualPath[i]
				copyPath[i] = individualPath[j]
                #Check if better
				newCost =  copyInd.cost(self.distanceMatrix)
				if newCost < bestCost : 
					if firstOrBest == "first":
						return copyInd
					bestPath = np.copy(copyPath)
					bestCost = newCost
					
				#Unswap
				copyPath[i] = individualPath[i]
				copyPath[j] = individualPath[j]

		#choose the bast path
		copyInd.path = bestPath

		return copyInd
		
	"""Insertion: two paths are neighbour is you can insert one node at beginning
	#and get the other path
	#STATS:
	#	finding the solution if 5steps from it: first:0.3-0.5 best:0.97-1
	# 	computation cost: fast computation O(n) or a bit more"""
	#TODO check speed diff with swap
	def lsoInsert(self, individual : Individual, firstOrBest : str):
		bestCost = individual.cost(self.distanceMatrix)
		bestPath = individual.path
		copyInd = deepcopy(individual)
		copyPath = copyInd.path
		individualPath = individual.path
		lengthOfPath = self.numberOfCities-1
		for i in range(lengthOfPath):
			#Insert
			copyPath[0] = individualPath[i]
			copyPath[1:i+1] = individualPath[0:i]
			copyPath[i+1:] = individualPath[i+1:]
			#Check if better
			newCost =  copyInd.cost(self.distanceMatrix)
			if newCost < bestCost : 
				if firstOrBest == "first":
						return copyInd
				bestPath = np.copy(copyPath)
				bestCost = newCost

		#choose the bast path
		copyInd.path = bestPath

		return copyInd

	"""Swap: two paths are neighbour is you can swap 2 adjacent nodes
	#and get the other path
	#STATS:
	#	finding the solution if 5steps from it: first:0.5-0.7 best:0.92-1
	# 	computation cost: fast computation O(n)"""

	#TODO check speed diff with insert
	def lsoSwap(self, individual : Individual, firstOrBest : str) -> Individual:
		bestCost = individual.cost(self.distanceMatrix)
		bestPath = individual.path
		copyInd = deepcopy(individual)
		copyPath = copyInd.path
		individualPath = individual.path
		lengthOfPath = self.numberOfCities-1
		for i in np.random.permutation(range(lengthOfPath)):
			j = (i + 1) % lengthOfPath
			#Swap adjacent position
			copyPath[j] = individualPath[i]
			copyPath[i] = individualPath[j]
			#Check if better
			newCost =  copyInd.cost(self.distanceMatrix)
			if newCost < bestCost : 
				if firstOrBest == "first":
					return copyInd
				bestPath = np.copy(copyPath)
				bestCost = newCost
			#Unswap
			copyPath[i] = individualPath[i]
			copyPath[j] = individualPath[j]

		#choose the bast path
		copyInd.path = bestPath

		return copyInd

	#TODO
	def nearest_neighbour_heuristic_init(self, perturbation_prob : float):
		heuristicPopulation = np.ndarray(self.populationsize, dtype = r0883878.Individual)
		nbHeuristic = int(self.heuristicInitPercent * self.populationsize)
		for i in range(nbHeuristic):
			heuristicPopulation[i] = r0883878.Individual(self.numberOfCities, perturbation_prob, self.init_mutation_proba, self.init_crossover_proba,
									self.init_k_selection, self.init_k_elimination)
			completePath = self.nearest_neighbour_heuristic(self.numberOfCities, randint(0, self.numberOfCities),self.distanceMatrix)
			#print("completepath:",completePath)
			heuristicPopulation[i].path = self.reducePath(completePath)
			#print("reducePath:",heuristicPopulation[i].path )
		for i in range(nbHeuristic,self.populationsize):
			heuristicPopulation[i] = r0883878.Individual(self.numberOfCities, perturbation_prob, self.init_mutation_proba, self.init_crossover_proba,
									self.init_k_selection, self.init_k_elimination)

		self.population =  heuristicPopulation

	def nearest_neighbour_heuristicOld(self, NbCities : int, start :int, distMatrix):
		path = np.zeros(NbCities, dtype = int)
		path[0] = start
		path[NbCities-1] = 0
		mask = np.ones(NbCities, dtype=bool) 
		mask[start] = False
		mask[0] = False

		for i in range(1,NbCities-1):
			last = path[i]
			next_ind = np.argmin(distMatrix[last][mask]) # find minimum of remaining locations
			next_loc = np.arange(NbCities)[mask][next_ind] # convert to original location
			path[i] = next_loc
			mask[next_loc] = False

		return path

#https://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm
	def nearest_neighbour_heuristic(self, NbCities : int, start :int, distMatrix):
		path = np.zeros(NbCities, dtype = int)
		path[0] = start
		mask = np.ones(NbCities, dtype=bool) 
		mask[start] = False

		for i in range(1,NbCities):
			last = path[i-1]
			next_ind = np.argmin(distMatrix[last][mask]) # find minimum of remaining locations
			next_loc = np.arange(NbCities)[mask][next_ind] # convert to original location
			path[i] = next_loc
			mask[next_loc] = False

		return path

	def reducePath(self, oldPath):
		N = self.numberOfCities
		newPath = np.zeros(N-1, dtype = int)
		i_FirstCity = np.where(oldPath == 0)[0][0]
		newPath[:N-1-i_FirstCity] = oldPath[i_FirstCity+1:]
		newPath[N-1-i_FirstCity:] = oldPath[:i_FirstCity]

		return newPath


		
#-----------------DIVERSITY--------------------------------------
	def getDistances(self, ind : Individual, population : np.array(Individual)) -> np.array(int):
		distances = np.ndarray(population.size, dtype = int)
		for i, x in enumerate(population):
			distances[i] = self.getHammingDist(ind,x)
		return distances

	"""hammingDistance: number of differences"""
	def getHammingDist(self, ind1 : Individual, ind2 : Individual) -> int : 
		return int(distance.hamming(ind1.path, ind2.path) * (self.numberOfCities-1))

	"""Depenbds on the chosen distance operator"""
	def getSearchSpace(self, n : int) -> int:
		return n

	def sharedCostPopulation(self, individuals : np.array(Individual), population : np.array(Individual), betaInit = 0):
		sharedCosts = np.zeros( np.size(individuals), dtype=float )
		for i, x in enumerate(individuals):
			subPopulation = np.random.choice(population, round(self.percentageCostSharing * population.size))
			sharedCosts[i] = self.sharedCost(x, subPopulation, betaInit)
		return sharedCosts

	def sharedCost(self, individual : Individual, population : np.array(Individual), betaInit = 0):
		if population is None :
			return individual.cost(self.distanceMatrix)

		ds = self.getDistances(individual, population)
		onePlusBeta = betaInit
		for d in ds:
			if d <= self.sigma:
				onePlusBeta += 1 - (d/self.sigma)**self.sharedCost_alpha

		fval = individual.cost(self.distanceMatrix)
		#print("fval:",fval,"onePlusBeta:",onePlusBeta)
		returnVal = fval * onePlusBeta**np.sign(fval)
		return returnVal

	def getHammingMatrix(self, pop : np.array(Individual)):
		m = np.ndarray((pop.size,pop.size), dtype = int)
		for i in range(pop.size):
			m[i] = self.getDistances(pop[i],pop)
		return m

	def getDictProximity(self, distMatrix, pop: np.array(Individual)):
		dic = {}
		for i in range(pop.size):
			dic[i] = np.where(distMatrix[i] < self.sigma)[0]
		return dic

	#OPTI
	def sharedCostOpti(self, compareTo : np.array(int), index_indiv : int, proximityIndexes : np.array(int), distArray : np.array(int), basicCosts : np.array(float), betaInit = 0):
		onePlusBeta = betaInit

		for i in proximityIndexes:
			if i in compareTo:
				d = distArray[i]
				onePlusBeta += 1 - (d/self.sigma)**self.sharedCost_alpha
			
		fval = basicCosts[index_indiv]
		return fval * onePlusBeta

	def sharedCostPopulationOpti(self, compareTo : np.array(int), individuals : np.array(Individual), proximityDict : dict, 
		distaceMatrix, fvals : np.array(float), basicCosts : np.array(float), betaInit = 0):
		newSurvivor = compareTo[compareTo.size - 1]
		for index in proximityDict[newSurvivor]:
			fvals[index] = self.sharedCostOpti(compareTo, index, proximityDict[index], distaceMatrix[index], basicCosts, betaInit)
		return fvals


	def eliminationDiversityPromotionOpti(self, pop : np.array(Individual), NbSurvivors : int, percentagePopu : float) -> np.array(Individual):
		 #setup do most calculation just once
		survivors = np.ndarray(NbSurvivors, dtype = r0883878.Individual)
		survivors_indexes = np.ndarray(NbSurvivors, dtype = int)
		subPopulation = np.random.choice(pop, round(percentagePopu * pop.size))
		distMatrix = self.getHammingMatrix(subPopulation)
		proximityDic = self.getDictProximity(distMatrix, subPopulation)

		fvals = self.getFitnessPopulation(subPopulation)
		basicCosts = deepcopy(fvals)

		#fisrt survivor
		idx = np.argmin(fvals)
		survivors_indexes[0] = idx
		survivors[0] = subPopulation[idx]

		#loop to get all survivors
		for i in range(1,NbSurvivors):
			fvals = self.sharedCostPopulationOpti( survivors_indexes[0:i], subPopulation, proximityDic, distMatrix, fvals, basicCosts, betaInit=1.0)
			
			#k_tournament
			random_index_Sample = sample(range(subPopulation.size), self.k_elimDiversity) 
			idx = random_index_Sample[ np.argmin(fvals[random_index_Sample]) ]
			"""
			#best one
			idx = np.argmin(fvals)
			"""
			survivors_indexes[i] = idx
			survivors[i] = subPopulation[idx]
		
		#print("eliminationDiversityPromotionOpti survivors_indexes:",survivors_indexes)
		return survivors
	
	def selectionDiversityPromotion(self,  k : int, population : np.array(Individual), sharedCosts : np.array(float)) -> np.array(Individual):
		random_index_Sample = sample(range(population.size), k)
		costSample = np.ndarray(k, dtype= float)

		for i in range(k):
			costSample[i] = sharedCosts[random_index_Sample[i]]

		best_index = np.argmin(costSample)
		#print("selectionDiversityPromotionc random_index_Sample:",random_index_Sample,"costSample:",costSample,"best_index:",best_index,"bestInd index:",random_index_Sample[best_index])

		return population[random_index_Sample[best_index]]

	def getDivesityIndicator(self, population : np.array(Individual)) -> float:
		totalDistances = 0.0
		for i, ind in enumerate(population):
			totalDistances += np.mean(self.getDistances(ind,population)) 
		return (totalDistances / self.populationsize) / self.numberOfCities

	def getDivesityIndicatorNew(self, population : np.array(Individual)) -> float:
		distMatrix = self.getHammingMatrix(population)
		return distMatrix.mean() / self.numberOfCities

#-----------------MAIN LOOP--------------------------------------
	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		if self.timeSteps:
			initStart = time.time()

		#SETUP of The problem
		self.numberOfCities = len(self.distanceMatrix)

		
		self.k_elimDiversity = round(0.1* self.populationsize)

		self.sigma = round(self.sharedCost_percentageOfSearchSpace * self.getSearchSpace(self.numberOfCities))
		self.numberOfSwitches = int(self.percentageOfSwitches * self.numberOfCities)
		self.initialisation(self.perturbation_prob)
		(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
		if self.AssesQuality or self.printEveryIter: 
			self.meanMutation, self.meanCrossover = self.getMeanMutation(),self.getMeanCrossover()
		if self.AssesQuality:

			self.k_selection = self.get_k_selection(self.population)
			self.k_elimination = self.get_k_elimination(self.population)
			print("BEFORE LSO Init meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective,
				"k_selection:",self.k_selection,"k_elimination:",self.k_elimination ,
				"mean_mutation:",self.meanMutation,"mean_crossover:",self.meanCrossover)
		#LSO initital
		if self.LsoInit:
			r.lsoInit_(self.population)

		(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
		if self.AssesQuality : 
			print("AFTER LSO Init meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective,
				"k_selection:",self.k_selection,"k_elimination:",self.k_elimination ,
				"mean_mutation:",self.meanMutation,"mean_crossover:",self.meanCrossover)
		
		lastMeans = np.zeros(self.genForConvergence)

		f = open('plot.csv', 'w',newline='')
		writer = csv.writer(f)
		writer.writerow(['Iteration','MeanValue','BestValue','DiversityIndicator','MeanMutation','MeanCrossover','k_selection','k_elimination'])

		if self.timeSteps:
			initTime = time.time() - initStart
			print("TIME: initTime:",initTime)

			selectTotal = 0.0
			LsoTotal = 0.0
			reportTotal = 0.0
			elimTotal = 0.0
			AssesQualityTotal = 0.0
			reDiversityTotal = 0.0
			itertotal = 0.0

		i=0
		lastMeans = self.addNewMean(lastMeans,meanObjective)
		#TODO put the stop criteria back: and self.stoppingCriteria(lastMeans,i)

		if self.AssesQuality:
			print(self.str_param() and self.stoppingCriteria(lastMeans,i))

		while( i<self.iterations and self.stoppingCriteria(lastMeans,i)):

			if self.timeSteps:
				selectStart = time.time()
				iterStart = time.time()

			#k_selection
			self.k_selection = self.get_k_selection(self.population)
			if self.AssesQuality or self.printEveryIter: 
				self.meanMutation, self.meanCrossover = self.getMeanMutation(),self.getMeanCrossover()
			#crossover + mutation
			offsprings = np.ndarray(self.populationsize, dtype=r0883878.Individual)
			nbr_offspring = 0

			for ind in self.population:
				self.mutation(ind,self.numberOfSwitches)

			sharedCosts=None
			if self.selectionDiversity:
				subPopulation_sharedCost = np.random.choice(self.population, round(self.population.size * self.percentageCostSharing))
				sharedCosts = self.sharedCostPopulation(self.population,subPopulation_sharedCost)

			


			for j in range(self.populationsize//2):
				#crossover
				p1 = self.selection(sharedCosts)
				p2 = self.selection(sharedCosts)
				if self.LsoToParents :
					self.lsoGeneration_(p1)
					self.lsoGeneration_(p2)

				new_individuals = self.crossover(p1,p2,self.numberOfSwitches)
				if not (new_individuals is None) : #mutate the offsprings
					offsprings[nbr_offspring] = self.mutation(new_individuals[0], self.numberOfSwitches)
					offsprings[nbr_offspring+1] = self.mutation(new_individuals[1], self.numberOfSwitches)
					nbr_offspring += 2

			offsprings.resize(nbr_offspring)
			
			newPopulation = np.concatenate((self.population,offsprings))

			if self.timeSteps:
				selectTime = time.time() - selectStart
				LsoStart = time.time()

			if self.RandomHardMutationThenLso:
				NbRandoms = round(self.population.size * self.percentHardMutation)
				for ind in self.getRandomSubset(newPopulation, NbRandoms):
					self.mutation_scramble(ind)
					self.lsoInterchange(ind,"best")

			if self.LsoToWorstOnes : 
				NbOfWorstOnes = round(self.population.size * self.percentOfPopuLso)
				#print("NbOfWorstOnes:",NbOfWorstOnes)
				#print("self.getWorstOnes(newPopulation, NbOfWorstOnes):",self.getWorstOnes(newPopulation, NbOfWorstOnes))
				for index in self.getWorstOnesIndex(newPopulation, NbOfWorstOnes):
					#self.mutation_scramble(newPopulation[index]) 
					self.lsoGeneration_(newPopulation[index])
			
			if self.LsoToRandomSubset : 
				NbRandoms = round(self.population.size * self.percentOfPopuLso)
				for ind in self.getRandomSubset(newPopulation, NbRandoms):
					#self.mutation_scramble(ind)
					self.lsoGeneration_(ind)

			if self.timeSteps:
				LsoTime = time.time() - LsoStart
				elimStart = time.time()

			#elimination
			self.k_elimination = self.get_k_elimination(newPopulation)
			self.population = self.elimination(newPopulation)

			if self.timeSteps:
				elimTime = time.time() - elimStart
				AssesQualityStart = time.time()


			(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
			if self.AssesQuality : 
				diversityIndicator = self.getDivesityIndicator(self.population)
				# write a row to the csv file
				writer.writerow([i,meanObjective,bestObjective,diversityIndicator,self.meanMutation, self.meanCrossover,self.k_selection,self.k_elimination])
			else:
				diversityIndicator = None
			if self.printEveryIter :
				print("I:", i,"meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective,
				"diversityIndicator:", diversityIndicator,
				"k_selection:",self.k_selection,"k_elimination:",self.k_elimination ,
				"mean_mutation:",self.meanMutation,"mean_crossover:", self.meanCrossover)
				#print("Population:", self.population)
			

			if self.timeSteps:
				AssesQualityTime = time.time() - AssesQualityStart
				reDiversityStart = time.time()


			#if diversity is dead relaunch 30% of population big mutation
			if self.reDiversitification and diversityIndicator < 0.1:
				NbOfIndivToMutate = round(self.population.size * 0.3)
				#print("NbOfWorstOnes:",NbOfWorstOnes)
				#print("self.getWorstOnes(newPopulation, NbOfWorstOnes):",self.getWorstOnes(newPopulation, NbOfWorstOnes))
				subPopu = self.getRandomSubset(newPopulation, NbOfIndivToMutate)
				for ind in subPopu:
					self.mutation_scramble(ind) 
					self.mutation_scramble(ind)
					self.lsoGeneration_(ind)

			if self.timeSteps:
				reDiversityTime = time.time() - reDiversityStart
				ReportStart = time.time()


			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < self.minSecLeft:
				break

			i+=1
			lastMeans = self.addNewMean(lastMeans,meanObjective)

			if self.timeSteps:
				ReportTime = time.time() - ReportStart
				iterTime = time.time() - iterStart

				selectTotal += selectTime 
				LsoTotal += LsoTime 
				elimTotal += elimTime 
				AssesQualityTotal += AssesQualityTime 
				reDiversityTotal += reDiversityTime
				reportTotal += ReportTime
				itertotal += iterTime

				print("TIME: selectTime:      ",selectTime,"=>", selectTime/iterTime * 100)
				print("TIME: LsoTime:         ",LsoTime,"=>", LsoTime/iterTime * 100)
				print("TIME: elimTime:        ",elimTime,"=>", elimTime/iterTime * 100)
				print("TIME: AssesQualityTime:",AssesQualityTime,"=>", AssesQualityTime/iterTime * 100)
				print("TIME: reDiversityTime: ",reDiversityTime,"=>", reDiversityTime/iterTime * 100)
				print("TIME: ReportTime:      ",ReportTime,"=>", ReportTime/iterTime * 100)
				print("TIME: Total iterTime:  ",iterTime,"=>", 100)
				print("TOTAL TIME:", time.time() - initStart)
				print("--------------------------------------------------------------------------------")

	
		# close the file
		f.close()

		#keep finalresults
		fFinal = open('finalResults.csv', 'a',newline='')
		writerF = csv.writer(fFinal)
		writerF.writerow([meanObjective,bestObjective,i,timeLeft])
		fFinal.close()

		if self.AssesQuality:
			#store last params
			with open('lastParams.txt', 'w') as fParams:
				fParams.write(self.str_param())
				fParams.write("""I: {i} meanObjective:{meanObj} bestObjective:{bestObjective} diff:{diff}
					diversityIndicator:{diversityIndicator}\n
					mean_mutation:{mean_mutation} mean_crossover{mean_crossover}
					min_mutation:{minMutation} min_crossover:{minCrossover}\n
					select_diversity:{select_diversity} elim_diveristy:{elim_diversity} percentageCostSharing:{percentageCostSharing} k_elimDiversity:{k_elimDiversity}\n
					LsoInit:{LsoInit}
					LsoToParents:{LsoToParents} LsoToWorstOnes:{LsoToWorstOnes} LsoToRandomSubset:{LsoToRandomSubset}
					percentOfPopuLso:{percentOfPopuLso}\n 
					reDiversificationScheme:{reDiversity}
					RandomHardMutationThenLso:{Hardmutation} percentHardMutation = {percentHardMutation}
					""".format(i=i, meanObj=meanObjective, bestObjective=bestObjective, diff=meanObjective-bestObjective
					, diversityIndicator=diversityIndicator,
					mean_mutation= self.meanMutation,mean_crossover = self.meanCrossover,minMutation=self.min_mutation, minCrossover=self.min_crossover,
					select_diversity = self.selectionDiversity, elim_diversity= self.eliminationDiversity,percentageCostSharing = self.percentageCostSharing,k_elimDiversity=self.k_elimDiversity,
					LsoInit=self.LsoInit,
					LsoToParents = self.LsoToParents, LsoToWorstOnes=self.LsoToWorstOnes, LsoToRandomSubset=self.LsoToRandomSubset,
					percentOfPopuLso=self.percentOfPopuLso, reDiversity= self.reDiversitification,
					Hardmutation=self.RandomHardMutationThenLso, percentHardMutation=self.percentHardMutation))
				
				fParams.write("\nTIME: initTime:      {initTime}=>{pInit}\n".format(initTime=initTime , pInit=initTime / itertotal * 100))
				fParams.write("TIME: selectTime:      {meanselect}=>{pSelect}\n".format(meanselect=selectTotal / i, pSelect=selectTotal / itertotal * 100))
				fParams.write("TIME: LsoTime:         {meanLso}=>{pLso}\n".format(meanLso=LsoTotal / i, pLso=LsoTotal / itertotal * 100))
				fParams.write("TIME: elimTime:        {meanElim}=>{pElim}\n".format(meanElim=elimTotal / i, pElim=elimTotal / itertotal * 100))
				fParams.write("TIME: AssesQualityTime:{meaAsses}=>{pAsses}\n".format(meaAsses=AssesQualityTotal / i, pAsses=AssesQualityTotal / itertotal * 100))
				fParams.write("TIME: ReDiversityTime:{meanReDIversity}=>{pReDIversity}\n".format(meanReDIversity=reDiversityTotal / i, pReDIversity=reDiversityTotal / itertotal * 100))
				fParams.write("TIME: ReportTime:      {meanReport}=>{pReport}\n".format(meanReport=reportTotal / i, pReport=reportTotal / itertotal * 100))
				fParams.write("TIME: Total iterTime:  {meanIter}=>{pIter}\n".format(meanIter=itertotal / i, pIter=100))
				fParams.write("TOTAL TIME:"+ str(time.time() - initStart))
				

			print(self.str_param())

		print("meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective)
		#print(bestSolution)
		print("I:", i,"timeleft:",timeLeft)
		#print("best Solution:", bestSolution)

		if self.AssesQuality:
			print("tour29: simple greedy heuristic : 30350")
			print("tour100: simple greedy heuristic 272865")
			print("tour250: simple greedy heuristic 49889")
			print("tour750: simple greedy heuristic 119156")
			"""
			tour29: simple greedy heuristic 30350
			tour100: simple greedy heuristic 272865
			tour250: simple greedy heuristic 49889
			tour500: simple greedy heuristic 122355
			tour750: simple greedy heuristic 119156
			tour1000: simple greedy heuristic 226541
			"""

		if self.timeSteps:
				selectTotal += selectTime 
				LsoTotal += LsoTime 
				elimTotal += elimTime 
				AssesQualityTotal += AssesQualityTime 
				reportTotal += ReportTime
				itertotal += iterTime

				print("TIME Means :")
				print("TIME: selectTime:      ",selectTotal / i,"=>", selectTotal / itertotal * 100)
				print("TIME: LsoTime:         ",LsoTotal / i,"=>", LsoTotal / itertotal * 100)
				print("TIME: elimTime:        ",elimTotal / i,"=>", elimTotal / itertotal * 100)
				print("TIME: AssesQualityTime:",AssesQualityTotal / i,"=>", AssesQualityTotal / itertotal * 100)
				print("TIME: ReDiversityTime:{meanReDIversity}=>{pReDIversity}".format(meanReDIversity=reDiversityTotal / i, pReDIversity=reDiversityTotal / itertotal * 100))
				print("TIME: ReportTime:      ",reportTotal / i,"=>", reportTotal / itertotal * 100)
				print("TIME: Total iterTime:  ",itertotal / i,"=>", 100)
				print("TOTAL TIME:", time.time() - initStart)
				print("--------------------------------------------------------------------------------")	
		return 0

#-----------------CHOOSE OPERATORS--------------------------------------

	"""selection & elimination"""
	def selection(self, sharedCosts = None) -> Individual:
		if self.selectionDiversity:
			return self.selectionDiversityPromotion(self.k_selection, self.population, sharedCosts)
		return self.k_tournament(self.k_selection, self.population)
	def elimination(self, oldPopulation):
		if self.eliminationDiversity:
			return self.eliminationDiversityPromotionOpti(oldPopulation, self.populationsize, self.percentageCostSharing)
		return self.elimination_kTournament(oldPopulation, self.populationsize)
	
	
	"""mutation"""
	def mutation(self, individual : Individual, numberofswitches):
		if random() <= individual.mutation_proba:
			return self.mutation_randomSwaps(individual,numberofswitches)
		else:
			return individual
	def crossover(self, p1 : Individual, p2 : Individual, numberOfSwitchies):
		if random() <= (p1.crossover_proba + p2.crossover_proba)/2:
			return self.pmx_pair(p1,p2)
		return None
	
	"""LSO"""
	def lsoInit_(self,population):
		for i in range(population.size):
			population[i] = self.lsoSwap(population[i],"best")
			population[i] = self.lsoSwap(population[i],"best")
	def lsoGeneration_(self, individual : Individual):
		self.lsoSwap(individual,"first")



#-----------------OPTIONS--------------------------------------
	printEveryIter = True
	AssesQuality = True
	timeSteps = True
	minSecLeft = 30

	"""Mutation Selection"""
	percentageOfSwitches = 0.1 #how many of the cities will be swapped
	min_k_value = 8 #min for k
	max_k_value = 15 #max for k
	min_crossover = 0.8 #min for crossover probability
	min_mutation = 0.1 #max for crossover probability
	RandomHardMutationThenLso = True #option talked about in
	percentHardMutation = 0.1
	"""Local Search Operator"""
	LsoInit = False #Apply a LSO with selector best solution and the sawp neighbour function
	LsoToParents = False #apply Lso generation to all prarents
	LsoToWorstOnes = True #for worst ones:  lso (take first better)
	LsoToRandomSubset = True  #for random subset:  lso (take first better)
	percentOfPopuLso = 0.1 # For random and worst what percentage of the population will be sampled
	"""Diversity"""
	percentageCostSharing = 0.75 #percentage of population taken into account when calculating the shared cost
	selectionDiversity = True
	eliminationDiversity = False
	k_elimDiversity = None #done in the main

	sharedCost_alpha = 1 #alpha in the shared cost calculation
	sharedCost_percentageOfSearchSpace = 0.05  #Tpercentage of the searchSpace for sigma

	reDiversitification = False #reDiversification scheme

	"""Initialisation"""
	populationsize = 200 #population size at each iteration
	init_k_selection = 5 #initial value of k_selection
	init_k_elimination = 5 #initial value of k_elimination
	init_mutation_proba = 0.1 #initial value of mutation_proba
	init_crossover_proba = 1 #initial value of crossover_proba
	perturbation_prob = 0.2 #

	heuristicInit = True  #ADDED
	heuristicInitPercent = 0.2   #ADDED

	"""Stopping Criteria"""
	iterations = 75 #max iterations
	genForConvergence = 5 #number or generation taken into account for conevrgence
	stoppingConvergenceSlope = 0.000001 #mùin slope before stopping the loop




	
	
if __name__== "__main__":

	r = r0883878()
	r.optimize("tour100.csv")
