from os import access
import Reporter
import numpy as np
import random
import csv


# Modify the class name to match your student number.
class r0883878:

	numberOfCities = 0

	printEveryIter = False

	populationsize = 300
	#selection
	k_selection = 2
	percentageOfSwitches = 0.1

	k_elimination = 3
	mutation_proba = 0.4
	#crossover
	crossover_proba = 0.8

	iterations = 500
	genForConvergence = 5
	stoppingConvergenceSlope = 0.00001



	def __init__(self):
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

	def initialisation(self,numberOfCities):

		l = [i for i in range(1,numberOfCities)]
		population = np.ndarray(dtype=int ,shape=(self.populationsize,numberOfCities-1))
		for i in range(self.populationsize):
			population[i] =  np.random.permutation(l)

		return population

	def cost(self,path,distanceMatrix):

		c = distanceMatrix[0][path[0]]
		for i in range(len(path)-1):
			c += distanceMatrix[path[i]][path[i+1]]
		c+= distanceMatrix[path[len(path)-1]][0]
		return c


	def selection(self,population,distanceMatrix):
		return self.k_tournament(population,distanceMatrix,self.k_selection)

	def k_tournament(self,population,distanceMatrix,k):
		numberOfCities = len(distanceMatrix)

		random_index_Sample = np.ndarray(k,dtype=int)
		costSample = np.ndarray(k)
		for i in range(k):
			random_index_Sample[i] = np.random.randint(0,len(population))

			costSample[i] = self.cost(population[random_index_Sample[i]],distanceMatrix)

		best_index = np.argmin(costSample)

		return population[random_index_Sample[best_index]]


	def crossover(self,p1,p2,numberOfSwitchies):
		if np.random.rand() <= self.crossover_proba:
			off1,off2 = self.pmx_pair(p1,p2)
			return np.array([ self.mutation(off1,numberOfSwitchies),self.mutation(off2,numberOfSwitchies) ] )
		return None
		#else:
		#	return np.array([ self.mutation(p1,numberOfSwitchies),self.mutation(p2,numberOfSwitchies) ])

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


	def mutation(self,individual,numberofswitches):
		numberOfCities = len(individual)
		for i in range(numberofswitches):
			if np.random.rand() <= self.mutation_proba:
				index1 = np.random.randint(numberOfCities-1)
				index2 = np.random.randint(numberOfCities-1)
				temp = individual[index1]
				individual[index1] = individual[index2]
				individual[index2] = temp
		return individual



	def elimination(self,population,distanceMatrix):
		numberOfCities = len(distanceMatrix)
		newPopulation = np.ndarray((self.populationsize,numberOfCities-1),dtype=int)
		for i in range(self.populationsize):
			newPopulation[i] = self.k_tournament(population,distanceMatrix,self.k_elimination)
		return newPopulation

	def accessQualityOfGeneration(self,population,distanceMatrix):
		#Values for each iteration
			fitness = np.ndarray((self.populationsize))

			for i in range(self.populationsize):
				fitness[i] = self.cost(population[i],distanceMatrix)
	
			meanObjective = np.mean(fitness)
			bestObjective = np.min(fitness)
			bestSolution = population[np.argmin(fitness)].tolist()
			bestSolution.append(0)
			return (meanObjective, bestObjective, np.array(bestSolution))

	def stoppingCriteria(self, means, index):
		flag = True
		if index>self.genForConvergence:
			indexes = np.arange(float(self.genForConvergence))
			slope = np.polyfit(indexes, means, 1)
			if self.printEveryIter :
				print(slope[0]/np.mean(means))
			if abs(slope[0]/np.mean(means)) < self.stoppingConvergenceSlope:
				print("slope:",slope[0]/np.mean(means),"lastmeans:",means)
				flag = False
		return flag

	def addNewMean(self, means, newMean):
		means = np.roll(means, 1)
		means[0] = newMean
		return means
	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		numberOfCities = len(distanceMatrix)
		self.numberOfCities = numberOfCities

		population = self.initialisation(numberOfCities)

		(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(population,distanceMatrix)
		lastMeans = np.zeros(self.genForConvergence)

		f = open('plot.csv', 'w',newline='')
		writer = csv.writer(f)
		writer.writerow(['Iteration','MeanValue','BestValue'])
		
		self.print_param()
		

		i=0
		lastMeans = self.addNewMean(lastMeans,meanObjective)
		
		while( i<self.iterations and self.stoppingCriteria(lastMeans,i)):

			#crossover + mutation
			offsprings = np.ndarray((self.populationsize,numberOfCities-1),dtype=int)
			nbr_offspring = 0
			for j in range(self.populationsize//2):
				#crossover
				p1 = self.selection(population,distanceMatrix)
				p2 = self.selection(population,distanceMatrix)
				numberOfSwitches = int(self.percentageOfSwitches * numberOfCities)
				new_individuals = self.crossover(p1,p2,numberOfSwitches)
				if isinstance(new_individuals,np.ndarray):
					offsprings[nbr_offspring] = self.mutation(new_individuals[0],numberOfSwitches)
					offsprings[nbr_offspring+1] = self.mutation( new_individuals[1],numberOfSwitches)
					nbr_offspring += 2


			offsprings.resize((nbr_offspring,numberOfCities-1))
	
			newPopulation = np.concatenate((population,offsprings))

	

			#elimination
			population = self.elimination(newPopulation,distanceMatrix)

	


			(meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration(population,distanceMatrix)
			if self.printEveryIter :
				print("meanObjective:",meanObjective,", bestObjective:",bestObjective,"diff:",meanObjective-bestObjective)
				#print(bestSolution)
				print("I:", i)


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
		return 0


if __name__== "__main__":

	r = r0883878()
	r.optimize("tour29.csv")
