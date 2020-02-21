##########
# Import #
##########

import numpy as np


################
# Memory Class #
################

class Memory(object):
    def __init__(self, max_memory=100):
        self.__max_memory = max_memory
        self.__memory = list()

    def remember(self, item):
        if len(self.__memory) == self.__max_memory:
            self.__memory.pop(0)
        self.__memory.append(item)

    def random_access(self):
        index = np.random.randint(0, len(self.__memory))
        return self.__memory[index]

    def get_memory_size(self):
        return len(self.__memory)
