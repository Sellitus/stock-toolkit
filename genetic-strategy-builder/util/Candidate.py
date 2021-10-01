
import copy
import random

from .TrendIndicators import TrendIndicators as ti
from .MomentumIndicators import MomentumIndicators as mi


class Candidate:
    """
    Member of the population. Contains DNA, which consists of individual technical library
    """

    DNA = []
    randomize_default = 0.2

    def __init__(self, dna_to_mix=None, randomize=False, remove_duplicates=True):
        if randomize is True:
            randomize = self.randomize_default

        if dna_to_mix is None:
            self.generate_random_dna(randomize=randomize, remove_duplicates=remove_duplicates)
        else:
            self.splice_together_dna(dna_to_mix)

    def set_dna(self, dna):
        self.DNA = list(set(dna))

    def get_dna(self):
        return self.DNA

    def generate_random_dna(self, minimum=3, maximum=10, randomize=False, remove_duplicates=True):
        num_strategies = random.randrange(minimum, maximum)

        all_dna = ti.trend_dna + mi.momentum_dna

        new_dna = []
        for _ in range(num_strategies):
            new_dna.append(copy.deepcopy(all_dna[random.randint(0, len(all_dna) - 1)]))
            new_dna[-1].set_settings(randomize=randomize)

        self.DNA = new_dna
        if remove_duplicates:
            no_dupes = {}
            for ind in self.DNA:
                if str(ind) not in no_dupes:
                    no_dupes[str(ind)] = ind
            self.DNA = no_dupes.values()
        return self.DNA

    def splice_together_dna(self, dna_to_mix, remove_duplicates=True):
        dna_mutable = dna_to_mix.copy()
        # Splice together the strands randomly
        total = 0
        for dna in dna_mutable:
            total += len(dna)
        average = round(total / len(dna_mutable))

        spliced_dna = []
        random.shuffle(dna_mutable)

        for i in range(average):
            i = i % len(dna_mutable)
            dna = dna_mutable[i]
            if len(dna) - 1 > 0:
                idx = random.randint(0, len(dna) - 1)
            else:
                idx = 0
            # NOTE: Have this add an additional element somehow when a DNA list is empty
            if len(dna_mutable[i]) > 0:
                spliced_dna.append(dna_mutable[i][idx])
                dna_mutable[i].pop(idx)

        self.DNA = spliced_dna
        if remove_duplicates:
            no_dupes = {}
            for ind in self.DNA:
                if str(ind) not in no_dupes:
                    no_dupes[str(ind)] = ind
            self.DNA = no_dupes.values()
        return self.DNA


