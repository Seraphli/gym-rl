import numpy as np


class Epsilon(object):
    def get(self, step):
        return


class LinearAnnealEpsilon(Epsilon):
    """Linear anneal epsilon"""

    def __init__(self, initial, final, total):
        self.initial = initial
        self.final = final
        self.total = total
        self.decay = (self.final - self.initial) / self.total

    def get(self, step):
        return max(self.final, min(self.initial + self.decay * step, self.initial))


class ExpDecayEpsilon(Epsilon):
    """Exponential decay epsilon
    
    Using formula f(t) = _N*a^(-_lambda*t)
    """

    def __init__(self, initial, final, total, base):
        self.initial = initial
        self.final = final
        self.total = total
        self.base = base
        self._N = self.initial
        self._lambda = -np.log(0.05) / np.log(self.base) / self.total

    def get(self, step):
        return max(self.final, min(self.initial, self._N * np.power(self.base, -self._lambda * step)))


class MultiStageEpsilon(Epsilon):
    def __init__(self, l_eps):
        """Use list of epsilon for multistage calculation
        
        Args:
            l_eps (list of Epsilon): List of epsilon
        """
        self.l_eps = l_eps

    def get(self, step):
        for i in range(len(self.l_eps)):
            if step > self.l_eps[i].total:
                step -= self.l_eps[i].total
            else:
                return self.l_eps[i].get(step)
        return self.l_eps[-1].get(step + self.l_eps[-1].total)
