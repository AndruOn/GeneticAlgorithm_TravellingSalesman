
from individual import Individual
import Reporter
import numpy as np
import random
import csv
#import Individual

#TODO:
#   -duplicate paths ?
#	- more mutation
#	- more recombination
#	- more crossover
#	- relaunch diversity if converges
#	- local search operators
#	- multithreading ?
#	- diversity promotion
#	- object? (i think array are faster but with objects can change more quicker)
#	- measure selective pressure(module2Q&A) and diversity
#	- self adaptivity (mutation prob,crosssover_prob,...?) 
# 		done for mutation nd crossover but not for k



# Modify the class name to match your student number.
class r0883878:

	printEveryIter = True

	def __init__(self, populationsize, init_k_selection, percentageOfSwitches, init_k_elimination,
	 init_mutation_proba, init_crossover_proba, iterations, genForConvergence, stoppingConvergenceSlope):

		self.populationsize = populationsize
		#selection
		self.init_k_selection = init_k_selection
		self.k_selection = init_k_selection
		#elimination
		self.init_k_elimination = init_k_elimination
		self.k_elimination = init_k_elimination
		#mutation
		self.perturbation_prob = 0.2
		self.percentageOfSwitches = percentageOfSwitches
		self.init_mutation_proba = init_mutation_proba
		self.init_crossover_proba = init_crossover_proba
		#stopping criteria
		self.iterations = iterations
		self.genForConvergence = genForConvergence
		self.stoppingConvergenceSlope = stoppingConvergenceSlope

		self.reporter = Reporter.Reporter(self.__class__.__name__)

		self.population = None
		self.distanceMatrix = None
		



	def print_param(self):
		print("""
		populationsize = {_1}
		#selection
		init_k_selection = {_2}
		k_selection = {_21}
		percentageOfSwitches = {_3}

		init_k_elimination = {_4}
		k_selection = {_41}
		init_mutation_proba = {_5}
		#crossover
		init_crossover_proba = {_6}

		iterations = {_7}
		genForConvergence = {_8}
		stoppingConvergenceSlope = {_9}

		numberOfCities = {_10}
		""".format(_1=self.populationsize,_2=self.init_k_selection,_21=self.k_selection,_3=self.percentageOfSwitches,_4=self.init_k_elimination,
		_41=self.k_elimination,_5=self. init_mutation_proba,_6=self.init_crossover_proba, _7=self.iterations,_8=self.genForConvergence,
		_9=self.stoppingConvergenceSlope,_10=self.numberOfCities))

	#Initializes the population
	def initialisation(self, perturbation_prob : float):
		l = np.arange(1, self.numberOfCities)
		self.population = np.ndarray(dtype=Individual ,shape=(self.populationsize))
		for i in range(self.populationsize):
			self.population[i] =  Individual(self.numberOfCities, perturbation_prob, self.init_mutation_proba, self.init_crossover_proba)

#-----------------CHOOSE OPERATORS--------------------------------------
	def selection(self) -> Individual:
		return self.k_tournament(self.init_k_selection, self.population)

	def mutation(self, individual : Individual, numberofswitches):
		if np.random.rand() <= individual.mutation_proba:
			return self.mutation_randomSwaps(individual,numberofswitches)
		else:
			return individual

	
	def crossover(self, p1 : Individual, p2 : Individual, numberOfSwitchies):
		if np.random.rand() <= (p1.crossover_proba + p2.crossover_proba)/2:
			off1,off2 = self.pmx_pair(p1,p2)
			return np.array([ self.mutation(off1,numberOfSwitchies),self.mutation(off2,numberOfSwitchies) ] )
		return None

	def elimination(self, oldPopulation):
		return self.elimination_kTournament(oldPopulation)


#-----------------SELECTION--------------------------------------
	def k_tournament(self, k : int, population):
		random_index_Sample = np.ndarray(k,dtype=int)
		costSample = np.ndarray(k)
		for i in range(k):
			random_index_Sample[i] = np.random.randint(0,len(population))

			costSample[i] = population[i].cost(self.distanceMatrix)

		best_index = np.argmin(costSample)

		return population[random_index_Sample[best_index]]
	
	def get_k_selection(self, population) -> int:
		total_k = 0
		for ind in population:
			total_k += ind.k_selection
		#print("k_selection total_k:",total_k,"population.size:",population.size,"return:",int(total_k//population.size))
		return int(total_k//population.size)

#-----------------MUTATION--------------------------------------
#https://www.uio.no/studier/emner/matnat/ifi/INF3490/h16/exercises/inf3490-sol2.pdf
	def pmx(self, a, b, start, stop):
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
		start = random.randint(0, len(a)-half)
		stop = start + half
		off1 = Individual()
		off2 = Individual()
		off1.path , off2.path = ( self.pmx(a,b,start,stop) , self.pmx(b,a,start,stop) )
		(off1,off2) = self.combineSelfAdaptivity(p1,p2,off1,off2)
		return (off1,off2)

	def combineSelfAdaptivity(self, p1 : Individual, p2 : Individual, off1 : Individual, off2 : Individual):
		off1.mutation_proba, off2.mutation_proba = (self.combineProb(p1.mutation_proba, p2.mutation_proba) , self.combineProb(p1.mutation_proba, p2.mutation_proba))
		off1.crossover_proba, off2.crossover_proba = (self.combineProb(p1.crossover_proba, p2.crossover_proba) , self.combineProb(p1.crossover_proba, p2.crossover_proba))
		off1.k_selection, off2.k_selection = (self.combineK(p1.k_selection, p2.k_selection) , self.combineK(p1.k_selection, p2.k_selection))
		off1.k_elimination, off2.k_elimination = (self.combineK(p1.k_elimination, p2.k_elimination) , self.combineK(p1.k_elimination, p2.k_elimination))
		#print("new crossovered k_selection k1:",off1.k_selection,"k2:",off2.k_selection)
		#print("new crossovered k_elimination k1:",off1.k_elimination,"k2:",off2.k_elimination)
		return (off1,off2)


	def combineProb(self, per1, per2):
		beta = 2*np.random.random() - 0.5
		newPercent = per1 + beta * (per2-per1)
		return min(1.0, max(0.0 , newPercent))
	
	def combineK(self, k1, k2):
		beta = 5*np.random.random() - 2
		return max(1, min(int(k1 + beta * (k2-k1)) , 10 )) 

	def mutation_randomSwaps(self, ind : Individual ,numberofswitches : int):
		path = ind.path
		numberOfCities = len(path)
		for i in range(numberofswitches):
			index1 = np.random.randint(numberOfCities-1)
			index2 = np.random.randint(numberOfCities-1)
			temp = path[index1]
			path[index1] = path[index2]
			path[index2] = temp

		self.mutationSelfAdaptivity(ind)

		return ind

#TODO check that passage per reference and dont need to return Individual
	def mutationSelfAdaptivity(self, ind : Individual):
		ind.mutation_proba = self.mutate_proba(ind.mutation_proba)
		ind.crossover_proba = self.mutate_proba(ind.crossover_proba)
		ind.k_selection = self.mutate_k(ind.k_selection)
		ind.k_elimination = self.mutate_k(ind.k_elimination)
	
	def mutate_proba(self, percent : float):
		newPercent = percent + 0.1 * random.random() - 0.5
		return min(1.0, max(0.0 , newPercent))
	
	def mutate_k(self, k : int):
		newK = k + random.randint(0,4) - 2
		print("new mutated k:",min(1, max(10 , newK)))
		return min(1, max(10 , newK))

		
#TODO mutation_scramble(self,individual):


#-----------------ELIMINATION--------------------------------------
	def get_k_elimination(self, population) -> int:
		total_k = 0
		for ind in population:
			total_k += ind.k_elimination
		#print("k_elimination total_k:",total_k,"population.size:",population.size,"return:",int(total_k//population.size))
		return int(total_k//population.size)

	def elimination_kTournament(self, oldPopulation):
		numberOfCities = len(self.distanceMatrix)
		newPopulation = np.ndarray(self.populationsize,dtype=Individual)
		for i in range(self.populationsize):
			newPopulation[i] = self.k_tournament(self.init_k_elimination, oldPopulation)
		return newPopulation


#-----------------QUALITY--------------------------------------
	def accessQualityOfGeneration(self, population):
		#Values for each iteration
		fitness = np.ndarray((self.populationsize))

		for i in range(self.populationsize):
			fitness[i] = population[i].cost(self.distanceMatrix)

		meanObjective = np.mean(fitness)
		bestObjective = np.min(fitness)
		bestSolution = population[np.argmin(fitness)].path.tolist()
		bestSolution.append(0)
		return (meanObjective, bestObjective, np.array(bestSolution))


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
				flag = False
		return flag

	def addNewMean(self, means, newMean):
		means = np.roll(means, 1)
		means[0] = newMean
		return means

#-----------------LOCAL SEARCH OPERATORS--------------------------------------		


		
#-----------------MAIN LOOP--------------------------------------
	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		self.numberOfCities = len(self.distanceMatrix)

		self.initialisation(self.perturbation_prob)
		print("Population:", self.population)

		(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
		lastMeans = np.zeros(self.genForConvergence)

		f = open('plot.csv', 'w',newline='')
		writer = csv.writer(f)
		writer.writerow(['Iteration','MeanValue','BestValue'])
		
		self.print_param()
		

		i=0
		lastMeans = self.addNewMean(lastMeans,meanObjective)
		
		while( i<self.iterations and self.stoppingCriteria(lastMeans,i)):
			#k_selection
			self.k_selection = self.get_k_selection(self.population)
			#crossover + mutation
			offsprings = np.ndarray(self.populationsize,dtype=Individual)
			nbr_offspring = 0
			for j in range(self.populationsize//2):
				#crossover
				p1 = self.selection()
				p2 = self.selection()
				numberOfSwitches = int(self.percentageOfSwitches * self.numberOfCities)
				new_individuals = self.crossover(p1,p2,numberOfSwitches)
				if isinstance(new_individuals,np.ndarray): #mutate the offsprings
					offsprings[nbr_offspring] = self.mutation(new_individuals[0],numberOfSwitches)
					offsprings[nbr_offspring+1] = self.mutation( new_individuals[1],numberOfSwitches)
					nbr_offspring += 2


			offsprings.resize(nbr_offspring)
			
	
			newPopulation = np.concatenate((self.population,offsprings))

			#elimination
			self.k_elimination = self.get_k_elimination(newPopulation)
			self.population = self.elimination(newPopulation)

	


			(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
			if self.printEveryIter :
				print("I:", i,"meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective,
				"k_selection:",self.k_selection,"k_elimination:",self.k_elimination ,end=" ")
				print("Population:", self.population)
			


			# write a row to the csv file
			
			
			writer.writerow([i,meanObjective,bestObjective])
			

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

			i+=1
			lastMeans = self.addNewMean(lastMeans,meanObjective)

		

		# close the file
		
		f.close()

		#keep finalresults
		fFinal = open('finalResults.csv', 'a',newline='')
		writerF = csv.writer(fFinal)
		writerF.writerow([meanObjective,bestObjective,i,timeLeft])
		fFinal.close()

		print("meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective)
		#print(bestSolution)
		print("I:", i,"timeleft:",timeLeft)
		print("best Solution:", bestSolution)

		print("tour29: simple greedy heuristic : 30350")
		return 0


if __name__== "__main__":

	r = r0883878(
		populationsize = 10, init_k_selection = 2, percentageOfSwitches = 0.1, init_k_elimination = 3,
	 	init_mutation_proba = 0.4, init_crossover_proba = 0.8, 
	 	iterations = 100, genForConvergence = 5, stoppingConvergenceSlope = 0.0001)
	r.optimize("tourData/tour5.csv")
