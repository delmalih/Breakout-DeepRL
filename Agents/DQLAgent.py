##########
# Import #
##########

import json
import numpy as np
from keras.models import model_from_json
from .BaselineAgent import BaselineAgent
from ..Memory import Memory


#############
# DQL Class #
#############

class DQLAgent(BaselineAgent):
    def __init__(self, env, epsilon=0.1, memory_size=100, batch_size=16,
                 discount=0.99):
        self.__env = env
        self.__epsilon = epsilon
        self.__memory = Memory(memory_size)
        self.__batch_size = batch_size
        self.__discount = 0.99

    def learned_act(self, state):
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def reinforce(self, state, next_state, action, reward, done):
        self.__memory.remember([state, next_state, action, reward, done])
        if len(self.__memory.get_memory_size()) < self.__batch_size:
            return 0.0

        minibatch = [self.__memory.random_access()
                     for i in range(self.__batch_size)]

        input_states = np.array([values[0] for values in minibatch])
        new_states = np.array([values[1] for values in minibatch])
        input_qs_list = self.model.predict(input_states)
        future_qs_list = self.model.predict(new_states)
        target_q = np.zeros((self.batch_size, 4))

        for i, mem_item in enumerate(minibatch):
            state, next_state, action, reward, done = mem_item
            if done:
                target_q_value = reward
            else:
                future_reward = np.max(future_qs_list[i])
                target_q_value = reward + self.discount * future_reward

            target_q[i] = input_qs_list[i]
            target_q[i, action] = target_q_value

        target_q = np.clip(target_q, -3, 3)
        loss = self.model.train_on_batch(input_states, target_q)
        return loss

    def save(self, name_weights='model.h5', name_model='model.json'):
        self.__model.save_weights(name_weights, overwrite=True)
        with open(name_model, "w") as outfile:
            json.dump(self.__model.to_json(), outfile)

    def load(self, name_weights='model.h5', name_model='model.json'):
        with open(name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(name_weights)
        model.compile(optimizer="adam", loss="mse")
        self.__model = model
