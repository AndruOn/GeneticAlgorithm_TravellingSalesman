import numpy as np
from scipy.spatial import distance


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
                totalCost += 10^20
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
        return "Individual: mutation:{mutation} crossover:{crossover} k_selection:{k_selection} k_elimination:{k_elimination} path:{path}".format(
            mutation = self.mutation_proba, crossover=self.crossover_proba, k_selection=self.k_selection, k_elimination=self.k_elimination, path=self.path)


if __name__== "__main__":
    indi = Individual(29,0.2,.4,.8)

    print(indi.floatToPercentage(-0.2))
    print(114//30)
    print(57.62489368127279 // 18)
    print("(round(0.2):",round(0.2))
    print("(round(0.8):",round(0.8))

    array1 = np.array([1,2,3,4])
    array2 = np.array([0,5,7,4])
    print("hamming distance",distance.hamming(array1, array2))
    
