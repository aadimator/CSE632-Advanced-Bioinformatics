# Class that implements the proposed genetic algorithm
# The initial population, selection, substitution, adaptation function,
# crossover and mutation operators are implemented in this class.

import math
import random
from dataclasses import dataclass
import numpy as np

from utils import Utils
from Bio.pairwise2 import align
from vose_sampler import VoseAlias


class GA_MSA:
    """Class that implements genetic algorithm for MSA"""

    def __init__(self, population_size=10, generations=100, termination_generations=50,
                 min_generations=50, mutation_rate=0.05, gap_open_score=-0.5,
                 gap_extend_score=-0.1, use_affine_gap_penalty=True, score_matrix_path=None):
        """Class initialization.

        Keyword arguments:
        population_size -- the population size (default 10)
        generations -- at most iterations (default 100)
        min_generations -- at least iterations (default 50)
        termination_generations -- number of generations after no improvement (default 50)
        mutation_rate -- mutation rate (default 0.05)
        gap_open_score -- gap open penalty (default -0.5)
        gap_extend_score -- gap extend penalty (default -0.1)
        use_affine_gap_penalty -- use affine gap penalty, otherwise only use gap_open_score (default True)
        score_matrix_path -- path to the score matrix, if None provided, BLOSUM62 will be used (default None)
        """
        self.population_size = population_size
        self.generations = generations
        self.min_generations = min_generations
        self.termination_generations = termination_generations
        self.mutation_rate = mutation_rate
        self.gap_open_score = gap_open_score
        self.gap_extend_score = gap_extend_score
        self.use_affine_gap_penalty = use_affine_gap_penalty
        self.score_matrix_path = score_matrix_path

    @staticmethod
    def compute_pairwise_alignments(sequences):
        """Computes the pairwise alignments between the sequences provided.

        Keyword arguments:
        sequences -- list of strings

        Return:
        alignments -- list of strings (alignments)
        """
        seq_len = len(sequences)
        alignments = np.empty((seq_len, seq_len), dtype=object)
        gop = -2
        gep = -0.5

        for i in range(seq_len):
            for j in range(i, seq_len):
                seq1 = sequences[i]
                seq2 = sequences[j]
                pair_align = align.globalds(seq1, seq2,
                                            Utils.get_score_matrix(),
                                            gop, gep)[0]
                alignments[i][j] = pair_align[0]
                alignments[j][i] = pair_align[1]
        return alignments

    def init_pop(self, sequences):
        """Create an initial population. First, computes pairwise alignments of all the sequences
        and then generated `self.population_size` alignments by randomly selecting each sequence
        alignment. After that, adds gaps to the generated organism so they all have same size.

        Keyword arguments:
        sequences -- list of strings"""
        pairwise_alignments = GA_MSA.compute_pairwise_alignments(sequences)
        population = list()
        for i in range(self.population_size):
            alignments = list()
            for j in range(len(sequences)):
                alignments.append(
                    pairwise_alignments[j, random.randint(0, len(sequences) - 1)])

            # alignments = Utils.add_gaps(alignments)
            # alignments = Utils.remove_useless_gaps(alignment)
            population.append(Organism(alignments))
            print("\nPopulation " + str(i + 1) + ":")
            Utils.print_sequences(alignments)
        return Population(population)

    def score_pairwise(self, seq1, seq2, matrix, gap=True):
        """Pairwise score generator.

        Keyword arguments:
        seq1 -- first alignment
        seq2 -- second alignment
        matrix -- substitution matrix"""
        for A, B in zip(seq1, seq2):
            diag = ('-' == A) or ('-' == B)
            yield (self.gap_extend_score if gap else self.gap_open_score) if diag else matrix[(A, B)]
            gap = diag

    def calculate_fitness(self, alignment):
        """Calculate the fitness score of a particular alignment. The objective
        function used is the sum-of-pairs.

        Keyword arguments:
        alignment -- list of alignments"""
        sum_score = 0
        matrix = Utils.get_score_matrix(self.score_matrix_path)
        # create full dict from half dict
        matrix.update(dict(((b, a), val) for (a, b), val in matrix.items()))

        for i in range(len(alignment) - 1):
            for j in range(i + 1, len(alignment)):
                sum_score += sum(self.score_pairwise(
                    alignment[i], alignment[j], matrix))

        return round(sum_score, 2)

    def get_probability_distribution(self, population):
        """Get probability distribution of organisms from the population based on
        their fitness scores.

        Keyword arguments:
        population -- Population

        Returns:
        dict -- {index_organism:probability}
        """
        dist = dict()
        for i in range(len(population.organisms)):
            dist[i] = round(population.organisms[i].fitness /
                            population.fitness, 2)
        return dist

    def apply_crossover(self, population):
        """Apply crossover on the population. Selects parents from 
        probabilty distrbution, based on their fitnees score, and perform
        horizontal, vertical or neither recombination by a certain probability."""
        new_population = list()
        prob_dist = self.get_probability_distribution(population)
        sampler = VoseAlias(prob_dist)
        for _ in range(len(population.organisms)):
            index_1, index_2 = sampler.sample_n(2)
            p1 = population.organisms[index_1].alignments
            p2 = population.organisms[index_2].alignments

            # print()
            # print("P1")
            # print(p1)
            # print("P2")
            # print(p2)
            # print()

            prob = round(random.uniform(0, 1), 2)
            h_prob = 0.3
            v_prob = 0.5
            if prob <= h_prob:
                # print("Horizontal")
                child = self._horizontal_recombination(p1, p2)
            elif prob <= v_prob:
                # print("Vertical")
                child = self._vertical_recombination(p1, p2)
            else:
                child = random.choice([p1, p2])
            # print(child)
            new_population.append(Organism(child))
        return Population(new_population)

    def _horizontal_recombination(self, p1, p2):
        """Apply horizontal recombination, i.e. build an offspring by 
        randomly selecting each sequence from one of the parents."""
        return [random.choice([i, j]) for i, j in zip(p1, p2)]

    # TODO: implement single point crossover, from Vertical decomposition with Genetic Algorithm for Multiple Sequence Alignment paper
    def _vertical_recombination(self, p1, p2):
        """Apply vertical recombination, i.e. randomly define a cut point 
        in the sequence and build the offspring by copying the sequence 
        from position 1 up to the cut point from one parent and from the 
        cut point to the end of the sequence from the other parent.

        Keep track of the gaps before split point to maintain the integrity
        of the sequence. If split point isn't valid, select either of the parents
        as the offspring."""
        split_point = random.randint(1, min(len(p1[0]), len(p2[0])) - 1)
        if (not GA_MSA.valid_split_point(p1, p2, split_point)):
            return random.choice([p1, p2])
        p1_half = [alignment[:split_point] for alignment in p1]
        p2_half = [alignment[split_point:] for alignment in p2]
        return [i+j for i, j in zip(p1_half, p2_half)]

    @staticmethod
    def valid_split_point(p1, p2, split_point):
        """Returns if a split point is valid or not.
        Valid split points are those that maintain the integrity of the sequence structure.
        If the number of gaps before the split point is same in all the sequences in the alignments,
        then the split point is valid."""
        for a, b in zip(p1, p2):
            if Utils.number_of_gaps(a[:split_point]) != Utils.number_of_gaps(b[:split_point]):
                return False
        return True

    # Apply a mutation on child
    def mutate(self, population):
        mutation_operators = ["gap_open",
                              "gap_extension", "gap_reduction", "none"]
        for org_index in range(len(population.organisms)):
            if round(random.uniform(0, 1), 2) < self.mutation_rate:
                alignments_len = len(
                    population.organisms[org_index].alignments)
                operation = random.choice(mutation_operators)
                if operation == mutation_operators[0]:
                    # Open gap; a position in the sequence is randomly selected,
                    # and a block of gaps of variable size is inserted into the sequence.

                    align_index = random.randint(0, alignments_len - 1)
                    alignment = population.organisms[org_index].alignments[align_index]
                    pos = random.randint(0, len(alignment) - 1)
                    # 20% of the sequence length
                    max_gap_block_size = round(Utils.max_seq_length(
                        population.organisms[org_index].alignments) * 0.20)
                    gaps = "-" * random.randint(1, max_gap_block_size)
                    print("Gap Open Mutation")
                    print(alignment)
                    print(pos)
                    print(max_gap_block_size)
                    print(gaps)
                    alignment = alignment[:pos] + gaps + alignment[pos:]
                    print(alignment)
                    population.organisms[org_index].alignments[align_index] = alignment

                elif operation == mutation_operators[1]:
                    # extend gap
                    align_index, start, _ = Utils.get_gap_block(
                        population.organisms[org_index].alignments)

                    alignment = population.organisms[org_index].alignments[align_index]
                    print("Gap Extend Mutation")
                    print(alignment)
                    alignment = alignment[:start] + "-" + alignment[start:]
                    print(alignment)
                    population.organisms[org_index].alignments[align_index] = alignment

                elif operation == mutation_operators[2]:
                    # reduce gap
                    align_index, start, _ = Utils.get_gap_block(
                        population.organisms[org_index].alignments)

                    alignment = population.organisms[org_index].alignments[align_index]
                    print("Gap Reduce Mutation")
                    print(alignment)
                    alignment = alignment[:start] + alignment[start + 1:]
                    print(alignment)
                    population.organisms[org_index].alignments[align_index] = alignment

        return population

    def run(self, sequences=None, input_path=None):
        """Runs the GA on the provided sequences to find MSA.

        Keyword arguments:
        sequences -- list of sequences (default None)
        input_path -- Path to the input data file (default None)"""

        if not input_path and not sequences:
            print("No sequences provided")
            return

        if input_path:
            sequences = Utils.prepare_input(input_path)

        # Prints the original sequence
        print("Input matrix:")
        Utils.print_sequences(sequences)

        # Create the initial population
        population = self.init_pop(sequences)

        # Repeat for all generations or until a good solution appears
        best_val = 0
        best_organism = None
        counter = 0
        print()

        for g in range(self.generations):
            counter += 1

            for i in range(self.population_size):
                population.organisms[i].alignments = Utils.add_gaps(
                    population.organisms[i].alignments)
                population.organisms[i].alignments = Utils.remove_useless_gaps(
                    population.organisms[i].alignments)

                score = self.calculate_fitness(
                    population.organisms[i].alignments)
                population.organisms[i].fitness = score
                population.fitness += score

            Utils.print_population(population)

            max_index = max(enumerate(population.organisms),
                            key=lambda org: org[1].fitness)[0]
            print(max_index)
            max_fitness = population.organisms[max_index].fitness
            print(max_fitness)
            if (best_val < max_fitness):
                best_val = max_fitness
                best_organism = population.organisms[max_index]
                counter = 0

            if (g > self.min_generations and counter > self.termination_generations):
                break

            population = self.apply_crossover(population)
            population = self.mutate(population)

        # Best solution
        print("\nBest solution:")
        Utils.print_sequences(best_organism.alignments)
        return best_val


@dataclass
class Organism:
    alignments: list
    fitness: float = 0


@dataclass
class Population:
    organisms: list
    fitness: float = 0
