##########
# Import #
##########

import numpy as np


################
# Memory Class #
################

class Memory(object):
    def __init__(self, max_memory=100000):
        self.__max_memory = max_memory
        self.__memory = list()

    def remember(self, item):
        if len(self.__memory) == self.__max_memory:
            self.__memory.pop(0)
        self.__memory.append(item)

    def random_access(self, n):
        index = np.random.randint(0, len(self.__memory), size=n)
        return [self.__memory[i] for i in index]

    def get_memory_size(self):
        return len(self.__memory)
