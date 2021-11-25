import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer


class DeepQNetwork:
    def __init__(self, env, optimizer):
        self.n = env.action_space.n
        self._optimizer = optimizer
        self._network = Sequential([
                InputLayer(env.observation_space.shape),
                Conv2D(32, kernel_size=8, strides=4, activation='relu'),
                Conv2D(64, kernel_size=4, strides=2, activation='relu'),
                Conv2D(64, kernel_size=3, strides=1, activation='relu'),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(self.n),
            ])
        self._vars = self._network.trainable_variables

    def _preprocess_states(self, states):
        if states.dtype == tf.uint8:
            states = tf.cast(states, tf.float32) / 255.0
        return states

    @tf.function
    def predict(self, states):
        return self._network(self._preprocess_states(states))

    @tf.function
    def train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            Q = self.predict(states)
            mask = tf.one_hot(actions, depth=Q.shape[1])
            Q = tf.reduce_sum(mask * Q, axis=1)
            loss = tf.keras.losses.MSE(returns, Q)

        gradients = tape.gradient(loss, self._vars)
        self._optimizer.apply_gradients(zip(gradients, self._vars))
