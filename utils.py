# Class that handles utility methods
#
# It handles the addition, counting and
# removal of gaps and spaces from a matrix.
# It also reads and prepares the input matrix for the GA.
# It also reads a custom score matrix.

import os
import sys
import random
import numpy as np
import pandas as pd

from Bio.SubsMat import MatrixInfo
from Bio.SubsMat import SeqMat
from Bio.Align.substitution_matrices import load, Array

from contextlib import contextmanager


class Utils:

    # @staticmethod
    # def get_score_matrix(input_path=None):
    #     """Returns a substitution_matrices.Array (Score Matrix) that can be
    #     passed to evalutation function.

    #     If the input_path is not provided, then BLOSUM62
    #     scoring matrix will be returned instead.

    #     Parameters:
    #     input_path (str): Path to the csv score matrix

    #     Returns:
    #     Array: Score Matrix

    #     """
    #     if (not input_path):
    #         return load("BLOSUM62")
    #     return Utils.read(input_path)

    @staticmethod
    def get_score_matrix(input_path=None):
        """Returns a SeqMat (Score Matrix) that can be
        passed to biopython functions.

        If the input_path is not provided, then BLOSUM62
        scoring matrix will be returned instead.

        Parameters:
        input_path (str): Path to the csv score matrix

        Returns:
        SeqMat: Score Matrix

        """
        if (not input_path):
            return load("BLOSUM62")
        df = pd.read_csv(input_path, delimiter=",",
                         header=0, index_col=0)
        data = df.stack().to_dict()
        return SeqMat(data)

    @staticmethod
    def print_sequences(sequences):
        '''Prints the sequences provided.'''
        for seq in sequences:
            print(seq)

    @staticmethod
    def print_population(population):
        print()
        print("Population Fitness Score: " + str(population.fitness))
        print()
        for i, organism in enumerate(population.organisms):
            print("Organism #" + str(i + 1))
            print("Fitness Score: " + str(organism.fitness))
            Utils.print_sequences(organism.alignments)
            print()

    @staticmethod
    def max_seq_length(sequences):
        '''Returns the length of the largest sequence
        in a list of sequences.'''
        return len(max(sequences, key=len))

    @staticmethod
    def add_gaps(sequences):
        """
        Add the required gaps to produce an initial alignment.
        Inserts gaps at random positions inside the sequence
        such that the length of all the sequences is same (max
        length of the sequence).

        Parameters:
        sequences (list): List of sequences (str)

        Returns:
        alignment (list): Matrix of sequences
        """
        max_length = Utils.max_seq_length(sequences)
        alignment = list()
        for seq in sequences:
            s = list(seq)
            while len(s) < max_length:
                r = random.randint(0, len(s))
                s.insert(r, '-')
            alignment.append(''.join(s))

        return alignment

    @staticmethod
    def remove_useless_gaps(sequences):
        """
        Remove the gap-only columns from the sequences.

        Parameters:
        sequences (list): List of sequences (str)

        Returns:
        sequences (list): Sequences without any gap-only columns
        """
        s = np.array([[c for c in seq] for seq in sequences])
        only_gap_cols = np.all(s == '-', axis=0).nonzero()[0]
        s = np.delete(s, only_gap_cols, axis=1)
        return ["".join(row) for row in s]

    @staticmethod
    def number_of_gaps(sequence):
        """
        Get the number of gaps in a sequence.

        Parameters:
        sequence (str)

        Returns:
        number of gaps (int)
        """
        return sequence.count('-')

    @staticmethod
    def get_gap_block(sequences):
        """
        Returns a randomly selected block of gaps from a list of sequences (alignments).

        Parameters:
        sequences (list)

        Returns:
        position of the gap block, (alignment_index, start_pos, end_pos)
        """
        while True:
            align_index = random.randint(0, len(sequences) - 1)
            start_pos = random.randint(0, len(sequences[align_index]) - 1)

            if (sequences[align_index][start_pos] == "-"):
                break
        gap_indices = Utils.get_interval_gaps(
            sequences[align_index], start_pos)
        return align_index, gap_indices[0], gap_indices[-1]

    @staticmethod
    def get_interval_gaps(sequence, start_pos):
        gaps = []

        for i in range(start_pos, len(sequence)):
            if sequence[i] == "-":
                gaps.append(i)
            else:
                break

        for i in range(start_pos - 1, -1, -1):
            if sequence[i] == "-":
                gaps.append(i)
            else:
                break

        return sorted(gaps)

    @staticmethod
    def prepare_input(input_path):

        # Read input file string
        with open(input_path) as f:
            input_str = f.read()

        # Get lines
        input_str = input_str.replace(" ", "")
        lines_list = input_str.split("\n")
        
        return lines_list

    @staticmethod
    def read(handle, dtype=float):
        """Parse the file and return an Array object.
        Modified version of the default `read` function provided
        in Bio.Align.substitution_matrices to work with csv files.

        Original link:
                https://github.com/biopython/biopython/blob/master/Bio/Align/substitution_matrices/__init__.py"""
        try:
            fp = open(handle)
            lines = fp.readlines()
        except TypeError:
            fp = handle
            try:
                lines = fp.readlines()
            except Exception as e:
                raise e from None
            finally:
                fp.close()
        header = []
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
        rows = [line.strip().split(',') for line in lines[i:]]
        if len(rows[0]) == len(rows[1]) == 2:
            alphabet = [key for key, value in rows]
            for key in alphabet:
                if len(key) > 1:
                    alphabet = tuple(alphabet)
                    break
            else:
                alphabet = "".join(alphabet)
            matrix = Array(alphabet=alphabet, dims=1, dtype=dtype)
            matrix.update(rows)
        else:
            alphabet = rows.pop(0)
            alphabet = alphabet[1:]
            for key in alphabet:
                if len(key) > 1:
                    alphabet = tuple(alphabet)
                    break
            else:
                alphabet = "".join(alphabet)
            matrix = Array(alphabet=alphabet, dims=2, dtype=dtype)
            for letter1, row in zip(alphabet, rows):
                assert letter1 == row.pop(0)
                for letter2, word in zip(alphabet, row):
                    matrix[letter1, letter2] = float(word)
        matrix.header = header
        return matrix

    @staticmethod
    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
