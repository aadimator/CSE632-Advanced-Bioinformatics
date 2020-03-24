import random
from decimal import Decimal


class VoseAlias(object):
    """ A probability distribution for discrete weighted random variables and its probability/alias
    tables for efficient sampling via Vose's Alias Method (a good explanation of which can be found at
    http://www.keithschwarz.com/darts-dice-coins/).
    Taken from : https://github.com/asmith26/Vose-Alias-Method/blob/master/vose_sampler/vose_sampler.py
    """

    def __init__(self, dist):
        """ (VoseAlias, dict) -> NoneType """
        self.dist = dist
        self.alias_initialisation()
        self.table_prob_list = list(self.table_prob)

    def alias_initialisation(self):
        """ Construct probability and alias tables for the distribution. """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        for o, p in self.dist.items():
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)

    def alias_generation(self):
        """ Return a random outcome from the distribution. """
        # Determine which column of table_prob to inspect
        col = random.choice(self.table_prob_list)

        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """ Return a sample of size n from the distribution."""
        # Ensure a non-negative integer as been specified
        n = int(size)
        if n <= 0:
            raise ValueError(
                "Please enter a non-negative integer for the number of samples desired: %d" % n)

        return [self.alias_generation() for i in range(n)]
