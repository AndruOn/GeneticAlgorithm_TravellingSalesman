"""
An evolutionary algorithm for solving the Knapsack problem

In the binary knapsack problem, you are given a set of objects
    o₁, o₂, …, oₖ
along with their values
    v₁, v₂, …, vₖ
and their weights
    w₁, w₂, …, wₖ
The goal is to maximize the total value of the selected objects, subject to
a weight constraint W.
"""
module Knapsack

    using Random
    using Statistics
    using Plots

	""" Problem specific section """

    """ Representation of the binary knapsack problem """
    struct KnapsackProblem
        value :: Vector{Float64}
        weight :: Vector{Float64}
        capacity :: Float64

	# Generates a random problem instance with a given number of items.
        function KnapsackProblem(numObjects :: Integer)
            value = (2).^randn(numObjects)
            weight = (2).^randn(numObjects)
            capacity = 0.2 * sum(weight)
            new(value, weight, capacity)
        end
    end

    """ Candidate solution representation """
    mutable struct Individual
        order :: Vector{Int64}
        α :: Float64

	# Create a random individual
        Individual(kp :: KnapsackProblem) = new( randperm(length(kp.value)), max(0.01, 0.1+0.02*randn()) )

	# Create an individual with the given order and mutation rate.
        Individual(order :: Vector{Int64}, α :: Float64) = new(order, α)
    end

    """ Computes the objective value of the given individual for the given
    knapsack problem instance """
    function fitness(kp :: KnapsackProblem, ind :: Individual) :: Float64
        value = 0
        remainingCapacity = kp.capacity
        for i ∈ ind.order
            if kp.weight[ i ] <= remainingCapacity
                value += kp.value[ i ]
                remainingCapacity -= kp.weight[ i ]
            end
        end
        return value
    end

    """ Determines the items in the knapsack of the given individual """
    function inKnapsack(kp :: KnapsackProblem, ind :: Individual) :: Set{Int64}
        kpi = Set{Int64}()
        remainingCapacity = kp.capacity
        for i ∈ ind.order
            if kp.weight[ i ] <= remainingCapacity
                push!(kpi, i)
                remainingCapacity -= kp.weight[ i ]
            end
        end
        return kpi
    end

	""" Solution specific section """

    """ A structure for representing the parameters of the evolutionary algorithm """
    struct Parameters
        λ :: Integer
        μ :: Integer
        k :: Integer
        its :: Integer
    end

    """ Solve a knapsack problem instance using an evolutionary algorithm """
    function evolutionaryAlgorithm(kp :: KnapsackProblem, p :: Parameters)

	# Initialize population
        population = initialize(kp, p.λ)

	# Evaluate fitness and reporting
        fitnesses = map(x->fitness(kp, x), population)
        (tmp,i) = findmax(fitnesses)
        kns = inKnapsack(kp, population[i])
        println(0, ": Mean fitness = ", mean(fitnesses), "\t Best fitness = ", maximum(fitnesses), "\t Knapsack: ", kns)


        for i = 1 : p.its
            # Selection, recombination, and mutation
            offspring = Vector{Individual}(undef, p.μ)
            for jj = 1 : p.μ
                p₁ = selection(kp, population, p.k)
                p₂ = selection(kp, population, p.k)
                offspring[jj] = recombination(kp, p₁, p₂)
                mutate!(offspring[jj])
            end

            # Mutation of seed population
            for ind ∈ population
                mutate!(ind)
            end

            # Elimination
            population = elimination(kp, population, offspring, p.λ)

	    # Evaluate fitness and reporting
            fitnesses = map(x->fitness(kp, x), population)
            (tmp,ii) = findmax(fitnesses)
            kns = inKnapsack(kp, population[ii])
            println(i, ": Mean fitness = ", mean(fitnesses), "\t Best fitness = ", maximum(fitnesses)) # , "\t Knapsack: ", kns
        end
    end

    """ Randomly initialize the population """
    function initialize(kp :: KnapsackProblem, λ) :: Vector{Individual}
        return map(x->Individual(kp), 1:λ)
    end

    """ Randomly mutates an individual """
    function mutate!(ind :: Individual)
        if rand() < ind.α
            # Randomly swap elements
            i = rand(1:length(ind.order))
            j = rand(1:length(ind.order))
            tmp = ind.order[i]
            ind.order[i] = ind.order[j]
            ind.order[j] = tmp
        end
    end

    """ Subset-based recombination """
    function recombination(kp :: KnapsackProblem, p₁ :: Individual, p₂ :: Individual) :: Individual
	# Determine knapsack contents
        s₁ = inKnapsack(kp, p₁)
        s₂ = inKnapsack(kp, p₂)

        # Copy intersection to offspring
        offspring = intersect(s₁, s₂)

        # Copy in symmetric difference with 50% probability
        for ind ∈ symdiff(s₁, s₂)
            if rand() <= 0.5
                push!(offspring, ind)
            end
        end

	# Transform subsets back into an order
        n = length(kp.value)
        order = Vector{Int64}(undef, n)
        i = 1
        for obj ∈ offspring
            order[ i ] = obj
            i += 1
        end
        rem = setdiff(Set{Int64}(1:n), offspring)
        for obj ∈ rem
            order[ i ] = obj
            i += 1
        end

        # Randomly permute the elements
        order[ 1 : length(offspring) ] = order[ randperm(length(offspring)) ]
        order[ length(offspring)+1 : n ] = order[ length(offspring) .+ randperm(n-length(offspring)) ]

	# Randomly combine the mutation rates within the interval:
	# <----x----y---->
        β = 2*rand() - 0.5
        α = p₁.α + β * (p₂.α - p₁.α)

        return  Individual(order, α)
    end

    """ k-tournament selection """
    function selection(kp :: KnapsackProblem, population :: Vector{Individual}, k :: Integer) :: Individual
        selected = rand(population, k)
        (tmp, i) = findmax( map(x->fitness(kp, x), selected) )
        return selected[i]
    end

    """ λ+μ selection/elimination """
    function elimination(kp :: KnapsackProblem, population :: Vector{Individual}, offspring :: Vector{Individual}, λ :: Integer) :: Vector{Individual}
        combined = vcat(population, offspring)
        order = reverse(sortperm(map( x->fitness(kp, x), combined )))
        return combined[order[1 : λ]]
    end

    n = 25
    kp = KnapsackProblem(n)
    println("KP values = ", kp.value)
    println("KP weight = ", kp.weight)
    println("KP capacity = ", kp.capacity)

    heurOrder = reverse(sortperm(kp.value ./ kp.weight))
    heurBest = Individual(heurOrder, 0.0)

    println("Heuristic objective value = ", fitness(kp, heurBest))

    p = Parameters( 200, 100, 5, 100 )
    evolutionaryAlgorithm(kp, p)

    # Some additional reporting

    # for ind ∈ population
    #     println("Item order = ", ind.order)
    #     println("Knapsack items = ", inKnapsack(kp, ind))
    #     println("Objective value = ", fitness(kp, ind))
    # end

    # offspring = recombination(kp, population[1], population[2])
    # println("Offspring order = ", offspring.order)
    # println("Offspring knapsack = ", inKnapsack(kp, offspring))
    # println("Offspring α = ", offspring.α)

    # for i = 1 : 20
    #     mutate!(population[1])
    #     println(population[1].order)
    # end
end
