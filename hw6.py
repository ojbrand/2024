import evo
import random as rnd
import numpy as np
from numba import njit


@njit
# operators
def calculate_overallocation_penalty(assignments, tas):
    """Calculates the overallocation penalty across all TAs."""
    max_assigned = tas['max_assigned'] 
    actual_assigned = assignments.sum(axis=1) 
    overallocation_num = (-1 * np.maximum(0, actual_assigned - max_assigned))
    overallocation = np.sum(overallocation_num)
    return overallocation

def calculate_undersupport_penalty(assignments, sections):
    """Calculates penalty for sections that are understaffed."""
    min_ta_required = sections['min_ta']
    ta_assigned = assignments.sum(axis=0)  
    undersupport = np.maximum(0, min_ta_required - ta_assigned)
    return np.sum(undersupport)

def calculate_unwilling_penalty(assignments, preferences):
    """Calculates penalty for assigning TAs sections they're unwilling"""
    unwilling  = np.sum((preferences == -1) & (assignments == 1))
    return unwilling 

def calculate_unpreferred_penalty(assignments, preferences):
    """Calculates penalty for assigning TAs sections they're willing but not preferring"""
    unpreferred  = np.sum((preferences == 0) & (assignments == 1))
    return unpreferred


# agents
def mutate(solutions):
    """ An agent that swaps two random values """
    L = solutions[0]
    shape = L.shape
    i = rnd.randrange(0, shape[0])
    j = rnd.randrange(0, shape[1])
    L[i, j] = 1 - L[i, j]
    return L

def crossover(parent1, parent2):
    """ An agent that swaps two random values """
    mask = np.random.randint(2, size=parent1.shape).astype(np.bool)
    not_mask = np.logical_not(mask)
    
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    
    return offspring1, offspring2



def main():

    # create the environment
    E = evo.Environment()

    # register the fitness functions
    E.add_fitness_criteria("overallocation", calculate_overallocation_penalty)
    E.add_fitness_criteria("undersupport", calculate_undersupport_penalty)
    E.add_fitness_criteria("unwilling", calculate_unwilling_penalty)
    E.add_fitness_criteria("unpreferred", calculate_unpreferred_penalty)
    
    # register the agents
    E.add_agent("mutate", mutate)
    E.add_agent("crossover", crossover)

    # Adding 1 or more initial solution
    L = np.zeros(47, 17)
    E.add_solution(L)

    # Run the evolver
    E.evolve(1000000, 100, 100)

    # Print the final result
    print(E)


if __name__ == '__main__':
    main()

