from os import path
from r0883878 import r0883878
from individual import Individual
import numpy as np

def setup29():
    r = r0883878(
		populationsize = 50, init_k_selection = 5, percentageOfSwitches = 0.1, init_k_elimination = 5,
	 	init_mutation_proba = 0.6, init_crossover_proba = 0.8, perturbation_prob = 0.2,
	 	iterations = 300, genForConvergence = 5, stoppingConvergenceSlope = 0.0001)
    r.numberOfCities = 29
    # Read distance matrix from file.
    file = open("tourData/tour29.csv")
    r.distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return r

def setup5():
    r = r0883878(
		populationsize = 10, init_k_selection = 5, percentageOfSwitches = 0.1, init_k_elimination = 5,
	 	init_mutation_proba = 0.6, init_crossover_proba = 0.8, perturbation_prob = 0.2,
	 	iterations = 50, genForConvergence = 5, stoppingConvergenceSlope = 0.0001)
    r.numberOfCities = 5
    # Read distance matrix from file.
    file = open("tourData/tour5.csv")
    r.distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    r.initialisation(0.1)
    return r

def resetIndividual29():
    return Individual(numberOfCities=29, perturbation_prob=0.2, 
        base_mutation_proba=0.4, base_crossover_proba=0.8, 
        init_k_selection=5, init_k_elimination=5)

def resetIndividual5():
    return Individual(numberOfCities=29, perturbation_prob=0.2, 
        base_mutation_proba=0.4, base_crossover_proba=0.8, 
        init_k_selection=5, init_k_elimination=5)


def test_mutation_randomSwaps(r : r0883878, indi : Individual, numberOfSwitchies : int):
    print("\nTEST test_mutation_randomSwaps-----------------------------------------------------------------------")

    print("indi:   ",indi.path)
    r.mutation_randomSwaps(indi, numberOfSwitchies)
    print("newIndi:",indi.path)

def test_mutation_scramble(r : r0883878, indi : Individual):
    print("\nTEST test_mutation_scramble-----------------------------------------------------------------------")

    print("indi:   ",indi.path)
    r.mutation_scramble(indi)
    print("newIndi:",indi.path)

def test_k_tournament():
    print("\nTEST test_k_tournament-----------------------------------------------------------------------")

    r = setup5()
    costPopu = np.ndarray(r.populationsize,dtype=float)
    print("population:")
    for i in range(r.populationsize):
        costPopu[i] = r.population[i].cost(r.distanceMatrix)
        print("individual:", r.population[i].path,"cost:",costPopu[i])

    for i in range(2):
        indi = r.k_tournament(5, r.population)
        print("k_tournament 5 indi:",indi.path,"costIndi",indi.cost(r.distanceMatrix))
    for i in range(2):
        indi = r.k_tournament(10, r.population)
        print("k_tournament 10 indi:",indi.path,"costIndi",indi.cost(r.distanceMatrix))


def test_lsoSwap(r : r0883878, indi : Individual):
    print("\nTEST test_lsoSwap-----------------------------------------------------------------------")

    print("indi:       ", indi.path, "cost:", indi.cost(r.distanceMatrix))
    fisrtbetter = r.lsoSwap(indi,"first")
    print("fisrtbetter:", fisrtbetter.path, "cost:", fisrtbetter.cost(r.distanceMatrix))
    best = r.lsoSwap(indi,"best")
    print("best:       ", best.path, "cost:", best.cost(r.distanceMatrix))

def test_lsoInterchange(r : r0883878, indi : Individual):
    print("\nTEST test_lsoInterchange-----------------------------------------------------------------------")

    print("indi:       ", indi.path, "cost:", indi.cost(r.distanceMatrix))
    fisrtbetter = r.lsoInterchange(indi,"first")
    print("fisrtbetter:", fisrtbetter.path, "cost:", fisrtbetter.cost(r.distanceMatrix))
    best = r.lsoInterchange(indi,"best")
    print("best:       ", best.path, "cost:", best.cost(r.distanceMatrix))

def test_lsoInsert(r : r0883878, indi : Individual):
    print("\nTEST test_lsoInsert-----------------------------------------------------------------------")

    print("indi:       ", indi.path, "cost:", indi.cost(r.distanceMatrix))
    fisrtbetter = r.lsoInsert(indi,"first")
    print("fisrtbetter:", fisrtbetter.path, "cost:", fisrtbetter.cost(r.distanceMatrix))
    best = r.lsoInsert(indi,"best")
    print("best:       ", best.path, "cost:", best.cost(r.distanceMatrix))

def test_getHammingDist():
    print("\nTEST test_getHammingDist-----------------------------------------------------------------------")

    r = setup5()
    ind1 = r.population[0]
    ind2 = r.population[1]
    print("ind1.path:", ind1.path)
    print("ind2.path:", ind2.path)
    print("hamming distance:",r.getHammingDist(ind1,ind2))
    ind1.path = np.array([1,2,3,4])
    ind2.path = np.array([1,2,3,4])
    print("ind1.path:", ind1.path)
    print("ind2.path:", ind2.path)
    print("hamming distance (0):",r.getHammingDist(ind1,ind2))

if __name__== "__main__":
    #SETUP-----------------------------------------
    r = setup29()

    #NewIndidividual--------------------------------
    indi = resetIndividual29()
    
    #TESTS-----------------------------------------
    print("LAUNCH TESTS-----------------------------------------------------------------------------------")
    #MUTATION
    if False :
        indi = resetIndividual()
        test_mutation_randomSwaps(r,indi, numberOfSwitchies=2)
        indi = resetIndividual()
        test_mutation_scramble(r,indi)

    #K tournaments
    if False :
        test_k_tournament()

    #LSO
    if False :
        test_lsoSwap(r,indi)
        test_lsoInterchange(r,indi)
        test_lsoInsert(r,indi)

    #Diversity
    if False :
        test_getHammingDist()