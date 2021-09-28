import random

from .TrendIndicators import TrendIndicators as ti
from .MomentumIndicators import MomentumIndicators as mi


class Candidate:
    """
    Member of the population. Contains DNA, which consists of individual technical library
    """

    DNA = []

    def __init__(self, dna_to_mix=None):
        if dna_to_mix is None:
            self.generate_random_dna()
        else:
            self.splice_together_dna(dna_to_mix)

    def set_dna(self, dna):
        self.DNA = dna

    def get_dna(self):
        return self.DNA

    def generate_random_dna(self, minimum=2, maximum=10):
        num_strategies = random.randrange(minimum, maximum)

        all_dna = ti.trend_dna + mi.momentum_dna

        new_dna = []
        for _ in range(num_strategies):
            new_dna.append(all_dna[random.randint(0, len(all_dna) - 1)])

        self.DNA = new_dna
        return self.DNA

    def splice_together_dna(self, dna_to_mix):
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
        return self.DNA


