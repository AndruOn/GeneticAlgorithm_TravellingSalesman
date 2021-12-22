
import Reporter
import numpy as np
import random
import csv

#TODO:
#   -duplicate paths ?
#	- more mutation
#	- more recombination
#	- more crossover
#	- relaunch diversity if converges
#	- multithreading ?
#	- diversity promotion
#	- object? (i think array are faster but with objects can change more quicker)
#	- measure selective pressure(module2Q&A) and diversity

#FIRST
#	- local search operators
#	- self adaptivity (mutation prob,crosssover_prob,...?)


# Modify the class name to match your student number.
class r0883878:

	printEveryIter = False

	def __init__(self, populationsize, k_selection, percentageOfSwitches, k_elimination,
	 mutation_proba, crossover_proba, iterations, genForConvergence, stoppingConvergenceSlope):

		self.populationsize = populationsize
		#selection
		self.k_selection = k_selection
		#elimination
		self.k_elimination = k_elimination
		#mutation
		self.percentageOfSwitches = percentageOfSwitches
		self.mutation_proba = mutation_proba
		self.crossover_proba = crossover_proba
		#stopping criteria
		self.iterations = iterations
		self.genForConvergence = genForConvergence
		self.stoppingConvergenceSlope = stoppingConvergenceSlope

		self.reporter = Reporter.Reporter(self.__class__.__name__)


	def print_param(self):
		print("""
		populationsize = {_1}
		#selection
		k_selection = {_2}
		percentageOfSwitches = {_3}

		k_elimination = {_4}
		mutation_proba = {_5}
		#crossover
		crossover_proba = {_6}

		iterations = {_7}
		genForConvergence = {_8}
		stoppingConvergenceSlope = {_9}

		numberOfCities = {_10}
		""".format(_1=self.populationsize,_2=self.k_selection,_3=self.percentageOfSwitches,_4=self.k_elimination,_5=self. mutation_proba,_6=self.crossover_proba, 
		_7=self.iterations,_8=self.genForConvergence,_9=self.stoppingConvergenceSlope,_10=self.numberOfCities))

	#Initializes the population
	def initialisation(self,numberOfCities):
		l = np.arange(1,numberOfCities)
		self.population = np.ndarray(dtype=int, shape=(self.populationsize,numberOfCities-1))
		self.populationValues = np.ndarray(self.populationsize, dtype=dict)
		for i in range(self.populationsize):
			self.population[i] =  np.random.permutation(l)
			perturbation = 0.2
			self.populationValues[i] = {
				"mutation_prob" : self.mutation_proba + r0883878.floatToPercentage( np.random.uniform(-perturbation,perturbation) ) , 
				"crossover_prob": self.crossover_proba + r0883878.floatToPercentage( np.random.uniform(-perturbation,perturbation) ), 
				"k_selection" : np.random.randint(1,10),
				"k_elimination" : np.random.randint(1,10)
				}
		
	def floatToPercentage(val):
		return min(1, max(0 , val))


#-----------------CHOOSE OPERATORS--------------------------------------
	def selection(self,population):
		return self.k_tournament(population,self.k_selection)

	def mutation(self,individual,numberofswitches):
		return self.mutation_randomSwaps(individual,numberofswitches)
	
	def crossover(self,p1,p2,numberOfSwitchies):
		if np.random.rand() <= self.crossover_proba:
			off1,off2 = self.pmx_pair(p1,p2)
			return np.array([ self.mutation(off1,numberOfSwitchies),self.mutation(off2,numberOfSwitchies) ] )
		return None

	def elimination(self,population):
		return self.elimination_kTournament(population)


#-----------------SELECTION--------------------------------------
	def k_tournament(self,population,k):
		random_index_Sample = np.ndarray(k,dtype=int)
		costSample = np.ndarray(k)
		for i in range(k):
			random_index_Sample[i] = np.random.randint(0,len(population))

			costSample[i] = self.cost(population[random_index_Sample[i]],self.distanceMatrix)

		best_index = np.argmin(costSample)

		return population[random_index_Sample[best_index]]

#-----------------MUTATION--------------------------------------
#https://www.uio.no/studier/emner/matnat/ifi/INF3490/h16/exercises/inf3490-sol2.pdf
	def pmx(self,a,b, start, stop):
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
		return child

	def pmx_pair(self,a,b):
		half = len(a) // 2
		start = random.randint(0, len(a)-half)
		stop = start + half
		return ( self.pmx(a,b,start,stop) , self.pmx(b,a,start,stop) )

	def mutation_randomSwaps(self,individual,numberofswitches):
		numberOfCities = len(individual)
		for i in range(numberofswitches):
			if np.random.rand() <= self.mutation_proba:
				index1 = np.random.randint(numberOfCities-1)
				index2 = np.random.randint(numberOfCities-1)
				temp = individual[index1]
				individual[index1] = individual[index2]
				individual[index2] = temp
		return individual

	# mutation_scramble(self,individual):


#-----------------ELIMINATION--------------------------------------
	def elimination_kTournament(self,population):
		newPopulation = np.ndarray((self.populationsize, self.numberOfCities-1), dtype=int)
		for i in range(self.populationsize):
			newPopulation[i] = self.k_tournament(population, self.k_elimination)
		return newPopulation


#-----------------QUALITY--------------------------------------
	#Calculate the cost of a path(array)
	def cost(self,path,distanceMatrix):
		c = distanceMatrix[0][path[0]]
		for i in range(len(path)-1):
			c += distanceMatrix[path[i]][path[i+1]]
		c+= distanceMatrix[path[len(path)-1]][0]
		return c

	def accessQualityOfGeneration(self,population):
		#Values for each iteration
			fitness = np.ndarray((self.populationsize))

			for i in range(self.populationsize):
				fitness[i] = self.cost(population[i],self.distanceMatrix)
	
			meanObjective = np.mean(fitness)
			bestObjective = np.min(fitness)
			bestSolution = population[np.argmin(fitness)].tolist()
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

		self.initialisation(self.numberOfCities)

		(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
		lastMeans = np.zeros(self.genForConvergence)

		f = open('plot.csv', 'w',newline='')
		writer = csv.writer(f)
		writer.writerow(['Iteration','MeanValue','BestValue'])
		
		self.print_param()
		

		i=0
		lastMeans = self.addNewMean(lastMeans,meanObjective)
		
		while( i<self.iterations and self.stoppingCriteria(lastMeans,i)):

			#crossover + mutation
			offsprings = np.ndarray((self.populationsize,self.numberOfCities-1),dtype=int)
			nbr_offspring = 0
			for j in range(self.populationsize//2):
				#crossover
				p1 = self.selection(self.population)
				p2 = self.selection(self.population)
				numberOfSwitches = int(self.percentageOfSwitches * self.numberOfCities)
				new_individuals = self.crossover(p1,p2,numberOfSwitches)
				if isinstance(new_individuals,np.ndarray): #mutate the offsprings
					offsprings[nbr_offspring] = self.mutation(new_individuals[0],numberOfSwitches)
					offsprings[nbr_offspring+1] = self.mutation( new_individuals[1],numberOfSwitches)
					nbr_offspring += 2


			offsprings.resize((nbr_offspring,self.numberOfCities-1))
	
			newPopulation = np.concatenate((self.population,offsprings))

			#elimination
			self.population = self.elimination(newPopulation)

			(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(self.population)
			if self.printEveryIter :
				print("I:", i, "meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective,end=" ")
				#print(bestSolution)


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
		return 0


if __name__== "__main__":

	numberOfCities = 0

	printEveryIter = False

	
	r = r0883878(
		populationsize = 300, k_selection = 2, percentageOfSwitches = 0.1, k_elimination = 3,
	 	mutation_proba = 0.4, crossover_proba = 0.8, 
	 	iterations = 200, genForConvergence = 5, stoppingConvergenceSlope = 0.0001)
	r.optimize("tourData/tour29.csv")
