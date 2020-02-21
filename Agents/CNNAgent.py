##########
# Import #
##########

import sys
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from .BaselineAgent import BaselineAgent
sys.path.append("..")
from Memory import Memory


#############
# CNN Class #
#############

class CNNAgent(BaselineAgent):
    def __init__(self, env, epsilon=0.1, memory_size=100, discount=0.9):
        self.__env = env
        self.__epsilon = epsilon
        self.__memory = Memory(memory_size)
        self.__discount = discount
        self.create_model()

    def create_model(self):
        nA = self.__env.get_number_of_actions()
        input_model = keras.layers.Input(shape=self.__env.get_state_shape())
        vgg = keras.applications.VGG16(include_top=False)
        encodings = keras.layers.Flatten()(vgg(input_model))
        output_model = keras.layers.Dense(nA)(encodings)
        self.__model = keras.models.Model(input_model, output_model)
        self.__model.compile(optimizer="adam", loss="mse")

    def learned_act(self, state):
        q_values = self.__model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def reinforce(self, state, next_state, action, reward, done,
                  batch_size=32):
        self.__memory.remember([state, next_state, action, reward, done])
        if self.__memory.get_memory_size() < batch_size:
            return 0.0

        minibatch = [self.__memory.random_access()
                     for i in range(batch_size)]

        input_states = np.array([values[0] for values in minibatch])
        new_states = np.array([values[1] for values in minibatch])
        input_qs_list = self.__model.predict(input_states)
        future_qs_list = self.__model.predict(new_states)
        target_q = np.zeros((batch_size, 4))

        for i, mem_item in enumerate(minibatch):
            state, next_state, action, reward, done = mem_item
            if done:
                target_q_value = reward
            else:
                future_reward = np.max(future_qs_list[i])
                target_q_value = reward + self.__discount * future_reward

            target_q[i] = input_qs_list[i]
            target_q[i, action] = target_q_value

        target_q = np.clip(target_q, -3, 3)
        loss = self.__model.train_on_batch(input_states, target_q)
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

    def train(self, n_epochs=20, batch_size=32, output_path=""):
        score = 0
        loss = 0
        for e in range(n_epochs):
            state = self.__env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.__env.step(action)
                score += reward
                loss = self.reinforce(state, next_state, action, reward, done)
            self.__env.draw_video(output_path + str(e))
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Score {}".format(
                e, n_epochs, loss, score))
            self.save(name_weights=output_path + 'model.h5',
                      name_model=output_path + 'model.json')
