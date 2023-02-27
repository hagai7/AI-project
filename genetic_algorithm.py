from __future__ import annotations
import numpy as np
from random import random
from typing import Tuple

from utils import weighted_choice, log_info
from configuration import GeneticAlgorithmConf, MainConf


class Individual:
    def __init__(self):
        pass

    def reproduce(self, other: Individual) -> Individual:
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError


class GeneticAlgorithm:
    def __init__(self, fitness_func, population_generation_func,
                 fitness_threshold, mutation_rate, num_iterations, mutation_decay=1.0):
        """
        :param fitness_func: float 0 to 1, the higher the better (more fit)
        :param population_generation_func: receives a seed, returns a list of individuals
        :param fitness_threshold: algorithm will stop if an individual reaches this threshold
        :param mutation_rate: float 0 to 1
        :param num_iterations: algorithm will stop after n iterations
        :param mutation_decay: float 0 to 1 by which to multiply the mutation rate every iteration
        """
        self.fitness_func = fitness_func
        self.population_generation_func = population_generation_func
        self.mutation_rate = mutation_rate
        self.num_iterations = num_iterations
        self.fitness_threshold = fitness_threshold
        self.mutation_decay = mutation_decay

    @staticmethod
    def _best_individual(population, fitnesses: np.ndarray) -> Tuple[Individual, float]:
        """
        :return: the most fit individual
        """
        best_index = fitnesses.argmax()
        return population[best_index], fitnesses[best_index]

    def _get_new_individual(self, population, fitnesses):
        a, b = np.random.choice(population, 2, False, fitnesses / fitnesses.sum())
        offspring = a.reproduce(b)
        if random() < self.mutation_rate:
            offspring.mutate()

        return offspring

    def _select_new_population(self,
                               parent_population, parent_fitnesses,
                               offspring_population, offspring_fitnesses):
        """
        Select the new population out of the parent and offspring populations
        :param offspring_population: doesn't have to be the same size as the parent population
        :return: tuple of (population, fitnesses)
        """
        all_population = parent_population + offspring_population
        all_fitnesses = np.hstack((parent_fitnesses, offspring_fitnesses))

        # Use the top fraction first
        sorted_indices = all_fitnesses.argsort()
        top_indices = sorted_indices[sorted_indices.size - int(parent_fitnesses.size * GeneticAlgorithmConf.TOP_FRACTION):]
        top_population = [all_population[i] for i in top_indices]
        top_fitnesses = all_fitnesses[top_indices]

        # Randomly (weighted) select the rest
        sub_all_bottom_indices = sorted_indices[:sorted_indices.size - int(parent_fitnesses.size * GeneticAlgorithmConf.TOP_FRACTION)]

        all_bottom_population = [all_population[i] for i in sub_all_bottom_indices]
        all_bottom_fitnesses = all_fitnesses[sub_all_bottom_indices]
        bottom_selected_indices = weighted_choice(np.arange(sorted_indices.size - int(parent_fitnesses.size * GeneticAlgorithmConf.TOP_FRACTION)),
                                                  size=int(parent_fitnesses.size * (1 - GeneticAlgorithmConf.TOP_FRACTION)),
                                                  weights=all_bottom_fitnesses)

        bottom_population = [all_bottom_population[i] for i in bottom_selected_indices]
        bottom_fitnesses = all_bottom_fitnesses[bottom_selected_indices]

        # Join the top and bottom
        population = top_population + bottom_population
        fitnesses = np.hstack((top_fitnesses, bottom_fitnesses))

        return population, fitnesses

    def generate(self, pop_size: int):
        population = self.population_generation_func(pop_size)
        fitnesses = np.array([self.fitness_func(individual) for individual in population])

        for i in range(1, self.num_iterations + 1):
            assert len(population) == MainConf.POP_SIZE

            log_info(f"Generation {i}/{self.num_iterations}")
            yield self._best_individual(population, fitnesses)

            if np.max(fitnesses) >= self.fitness_threshold:
                return

            # Create a population of offsprings
            offspring_population = [self._get_new_individual(population, fitnesses) for _ in range(len(population))]
            offspring_fitnesses = np.array([self.fitness_func(individual) for individual in offspring_population])

            # Get the new population, generated based on both parents and offsprings
            population, fitnesses = self._select_new_population(population, fitnesses,
                                                                offspring_population, offspring_fitnesses)

            # Update mutation rate
            self.mutation_rate *= self.mutation_decay

        yield self._best_individual(population, fitnesses)
