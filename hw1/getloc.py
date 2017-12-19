import numpy as np

class Loc:
    def __init__(self, filepath=''):
        lines = open(filepath).readlines()
        self.coords = []
        self.patterns = []

        for line in lines:
            row = [ float(i) for i in line.split()]
            self.coords.append((row[0], row[1]))
            self.patterns.append(row[2:])
        self.patterns = np.array(self.patterns)

    def getMatchCoord(self, pattern):
        pattern = np.array(pattern)
        dis = np.sum(np.power(self.patterns - pattern, 2), 1)
        idx = np.argmin(dis)
        return self.coords[idx]




