import random as rnd
import numpy as np
import numpy.random as rnd
import time

#1h26

class KnapsackProblem :
    
    def __init__(self,numObjects,k_selection,populationSize,offSpringSize,nbIterations):
        self.values = rnd.rand(numObjects)
        self.weights = rnd.rand(numObjects)
        self.capacity = 0.2 * np.sum(self.weights)
        self.numObjects = numObjects

        #parameters
        self.k_selection = k_selection
        self.populationSize = populationSize
        self.offSpringSize = offSpringSize
        self.nbIterations = nbIterations
        

    def initialise(self,populationSize):
        self.populationSize = populationSize
        self.individuals = np.ndarray((populationSize,self.numObjects), dtype=int)
        self.alphas = np.ndarray(populationSize)
        for i in range(populationSize):
            self.individuals[i] = np.random.permutation(self.numObjects)
            self.alphas[i] = max(0.01, 0.1+0.02*rnd.rand())

    def printPopulation(self):
        print("Population")
        for i in range(self.populationSize):
            print("{_1} {_11} alpha:{_2:.5f} fitness:{_3:.5f}".format(_1=self.individuals[i],_11=self.inKnapsack(i), _2=self.alphas[i], _3=self.fitness(i)))
        print("----------------------------------------")

    def printParam(self):
        print("ProblemParam:")
        print("numObjects: ",self.numObjects," populationSize: ",self.populationSize," offSpringSize: ",self.offSpringSize," k_selection: ",self.k_selection,"nbIterations: ",self.nbIterations)
    
   

    def printProblem(self):
        print("ProblemValues:")
        #print("index: \n",np.arange(self.numObjects))
        print("values: \n",self.values)
        print("weights: \n",self.weights)
        print("capacity: ",self.capacity)

    def generationInfo(self):
        fitness = np.ndarray(self.populationSize)
        for i in range(self.populationSize):
            fitness[i] = self.fitness(i)
        meanFitness = np.mean(fitness)
        BestFitness = np.max(fitness)
        bestIndex = np.argmax(fitness)
        return meanFitness,BestFitness,bestIndex

    def fitness(self,indexIndividual):
        totalValue = 0
        remainingCapacity = self.capacity
        individual = self.individuals[indexIndividual]
        alpha = self.alphas[indexIndividual]
       
        for i in range(self.numObjects):
            indexObject = individual[i]
            if self.weights[indexObject] < remainingCapacity : 
                totalValue += self.values[indexObject]
                remainingCapacity -= self.weights[indexObject]
        return totalValue

    def fitnessIndividual(self,individual):
        totalValue = 0
        remainingCapacity = self.capacity
       
        for i in range(self.numObjects):
            indexObject = individual[i]
            if self.weights[indexObject] < remainingCapacity : 
                totalValue += self.values[indexObject]
                remainingCapacity -= self.weights[indexObject]
        return totalValue

    def inKnapsack(self,indexIndividual):
        totalValue = 0
        elementsInKnapsack = []
        remainingCapacity = self.capacity
        individual = self.individuals[indexIndividual]
        alpha = self.alphas[indexIndividual]
       
        for i in range(self.numObjects):
            indexObject = individual[i]
            if self.weights[indexObject] < remainingCapacity : 
                totalValue += self.values[indexObject]
                elementsInKnapsack.append(indexObject)
                remainingCapacity -= self.weights[indexObject]
        return elementsInKnapsack

    def inKnapsackIndividual(self,individual):
        totalValue = 0
        elementsInKnapsack = []
        remainingCapacity = self.capacity
        alpha = None
       
        for i in range(self.numObjects):
            indexObject = individual[i]
            if self.weights[indexObject] < remainingCapacity : 
                totalValue += self.values[indexObject]
                elementsInKnapsack.append(indexObject)
                remainingCapacity -= self.weights[indexObject]
        return elementsInKnapsack
    
    def recombination(self,p1,p2):
        set1 = self.inKnapsack(p1)
        set2 = self.inKnapsack(p2)

        # Copy intersection to offspring
        offspring = np.intersect1d(set1,set2)
        symdiff = np.concatenate((np.setdiff1d(set1,set2),np.setdiff1d(set2,set1)))

        # Copy in symmetric difference with 50% probability
        for x in symdiff:
            if rnd.random() < 0.5:
                offspring = np.concatenate((offspring,np.array([x])))
 
        # shuffle the misingObject and the offspirng
        missingObjects = np.setdiff1d(np.arange(self.numObjects),offspring)
        rnd.shuffle(offspring)
        rnd.shuffle(missingObjects)

        #new alpha
        a1 = self.alphas[p1]
        a2 = self.alphas[p2]

        beta = 2*rnd.random() - 0.5
        newAlpha = a1 + beta * (a2-a1)
        return np.concatenate((offspring,missingObjects)),newAlpha

    def mutate(self,individual,alpha):
        if rnd.random() < alpha:
            i1 = rnd.randint(0,self.numObjects)
            i2 = rnd.randint(0,self.numObjects)
            tmp = individual[i1]
            individual[i1] = individual[i2]
            individual[i2] = tmp
        return individual


    def lsoInsert(self,individual):
        bestFitness = self.fitnessIndividual(individual)
        bestInd = individual
        copyInd = np.copy(individual)
        firstloc = individual[0]
        for i in range(1,self.numObjects):
            #Insert
            copyInd[0] = individual[i]
            copyInd[1:i+1] = individual[0:i]
            copyInd[i+1:] = individual[i+1:]
            #Check if better
            newFitness = self.fitnessIndividual(copyInd)
            if newFitness > bestFitness : 
                bestInd = np.copy(copyInd)
                bestFitness = newFitness

        return bestInd

    def lsoNeighbours1opt(self,individual):
        bestFitness = self.fitnessIndividual(individual)
        bestInd = individual
        copyInd = np.copy(individual)
        for i in range(self.numObjects):
            for j in range(i+1,self.numObjects):
                #Swap position
                copyInd[j] = individual[i]
                copyInd[i] = individual[j]
                #Check if better
                newFitness = self.fitnessIndividual(copyInd)
                if newFitness > bestFitness : 
                    bestInd = np.copy(copyInd)
                    bestFitness = newFitness
                #Unswap
                copyInd[i] = individual[i]
                copyInd[j] = individual[j]

        return bestInd

    def localSearchOp(self,individual):
        return self.lsoNeighbours1opt(individual)


    
    def selection(self):
        randomIndexes = rnd.randint(self.populationSize,size=self.k_selection)
        fitness = [self.fitness(i) for i in randomIndexes]
        return randomIndexes[np.argmax(fitness)]

    #(mu+lambda) elimination
    def elimination(self,offspringPop,alphaOffspring):
        newPopulation = np.concatenate((self.individuals,offspringPop))
        newAlphas = np.concatenate((self.alphas,alphaOffspring))
        fitness = [self.fitnessIndividual(individual) for individual in newPopulation]
        ind = np.argpartition(fitness,self.populationSize)[-self.populationSize:]
        return newPopulation[ind],newAlphas[ind]

    def optimise(self):
        startTime = time.time()
        self.initialise(self.populationSize)
        #heuristic Solution
        heuristic = (self.values /self.weights)
        heuristicIndividual = np.argsort(heuristic)[::-1]
        fitnessHeuristic = self.fitnessIndividual(heuristicIndividual)
        print("Heuristic solution: {_1} heuristicfitness_inKnapsack:{_2:.5f}".format(_1=np.sort(self.inKnapsackIndividual(heuristicIndividual)), _2=fitnessHeuristic))

        
        offsprings = np.ndarray((self.offSpringSize,self.numObjects), dtype=int)
        alphaOffsprings = np.ndarray(self.populationSize)

        i=0
        
        while (i<self.nbIterations):
            #Create offspring from selected individuals
            for o in range(self.offSpringSize):
                p1 = self.selection()
                p2 = self.selection()
                (offsprings[o],alphaOffsprings[o]) = self.recombination(p1,p2)
                offsprings[o] = self.mutate(offsprings[o],alphaOffsprings[o])
        
            #mutate every individual
            for ind in range(self.populationSize):
                self.individuals[ind] = self.mutate(self.individuals[ind],self.alphas[ind])
                self.individuals[ind] = self.localSearchOp(self.individuals[ind])

            #elimination
            self.individuals,self.alphas = self.elimination(offsprings,alphaOffsprings)

            #prints
            meanFitness,BestFitness,bestIndex = self.generationInfo()
            print("iter:{_0} mean:{_1:.5f} best:{_2:.5f}".format(_0=i, _1=meanFitness, _2=BestFitness))
           
            i+= 1

        finalTime = time.time()
        #calculate heuristic
        heuristic = (self.values /self.weights)
        heuristicIndividual = np.argsort(heuristic)[::-1]
        fitnessHeuristic = self.fitnessIndividual(heuristicIndividual)
        print("Heuristic solution: {_1} heuristicfitness_inKnapsack:{_2:.5f}".format(_1=np.sort(self.inKnapsackIndividual(heuristicIndividual)), _2=fitnessHeuristic))
        print("iter:{_0} mean:{_1:.5f} best:{_2:.5f} | bestIndividual_inKnapsack:{_3} alpha:{_4:.5f}".format(_0=i, _1=meanFitness, _2=BestFitness, _3=np.sort(self.inKnapsack(bestIndex)),_4=self.alphas[bestIndex]))
        print("Total time:",finalTime-startTime)

if __name__== "__main__":
    
    #kn = KnapsackProblem(numObjects=10,populationSize=50,offSpringSize=50,k_selection=5,nbIterations=20)
    #kn = KnapsackProblem(numObjects=50,populationSize=100,offSpringSize=100,k_selection=5,nbIterations=30)
    kn = KnapsackProblem(numObjects=100,populationSize=200,offSpringSize=200,k_selection=5,nbIterations=30)


    kn.printParam()
    #kn.printProblem()
    kn.optimise()

    



