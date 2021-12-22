import numpy as np


class Individual:

    def __init__(self, numberOfCities=None, perturbation_prob=None, 
        base_mutation_proba=None, base_crossover_proba=None):
        self.path = np.random.permutation(np.arange(1,numberOfCities))
        if (base_mutation_proba != None):
            self.mutation_proba = self.floatToPercentage( base_mutation_proba + np.random.uniform(-perturbation_prob,perturbation_prob) )
        if (base_crossover_proba != None):
            self.crossover_proba = self.floatToPercentage( base_crossover_proba + np.random.uniform(-perturbation_prob,perturbation_prob) )
        self.k_selection = np.random.randint(1,8)
        self.k_elimination = np.random.randint(1,8)


    def cost(self, distanceMatrix) -> np.double:
        c = distanceMatrix[0][self.path[0]]
        for i in range(len(self.path)-1):
            c += distanceMatrix[self.path[i]][self.path[i+1]]
        c+= distanceMatrix[self.path[len(self.path)-1]][0]
        return c

    def floatToPercentage(self,val) -> float:
        return min(1.0, max(0.0 , val))

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
