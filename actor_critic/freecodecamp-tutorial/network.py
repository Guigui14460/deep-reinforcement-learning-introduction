import os
from typing import Tuple

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions: int, fc1_dims: int = 1024, fc2_dims: int = 512,
                 name: str = "actor_critic", chkpt_dir: str = "tmp/actor_critic"):
        super(ActorCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, name + "_ac.h5")

        # neural nets
        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")

        # value function
        self.v = Dense(1, activation=None)

        # policy function (probability distribution to choose the best action to take)
        self.pi = Dense(n_actions, activation="softmax")

    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)
        return v, pi
