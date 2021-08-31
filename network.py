
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class Critic:
    def __init__(self, input_dims, critic_learning_rate):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.observation = tf.placeholder(shape=[None, input_dims], dtype=tf.float32, name='obs')
        self.hidden1 = tf.layers.dense(self.observation, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)
        self.hidden2 = tf.layers.dense(self.hidden1, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)
        self.Value_stream = tf.layers.dense(self.hidden2, 1,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            trainable=True)
        self.Value_output = self.Value_stream

        # Optimisation
        self.returns_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='returns_placeholder')

        self.critic_loss = tf.reduce_mean(
            tf.squared_difference(tf.squeeze(self.Value_output), self.returns_placeholder))
        self.critic_optimiser = tf.train.AdamOptimizer(critic_learning_rate,
                                                       name='actor_optimizer_main_engine')
        # self.critic_gradients, self.critic_variables = zip(*self.critic_optimiser.compute_gradients(self.critic_loss))
        # self.critic_gradients, _ = tf.clip_by_global_norm(self.critic_gradients, 5.0)
        # self.training_op_critic = self.critic_optimiser.apply_gradients(
        #     zip(self.critic_gradients, self.critic_variables))
        self.training_op_critic = self.critic_optimiser.minimize(self.critic_loss)


class Actor:
    def __init__(self, input_dims, clip_param, learning_rate):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.observation = tf.placeholder(shape=[None, input_dims], dtype=tf.float32, name='obs')
        self.hidden1 = tf.layers.dense(self.observation, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)
        self.hidden2 = tf.layers.dense(self.hidden1, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)
        self.main_engine_stream, self.lateral_engine_stream = tf.split(self.hidden2, 2, 1)

        self.main_engine_mu = tf.layers.dense(self.main_engine_stream, 1, activation=tf.nn.tanh,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              trainable=True)
        self.main_engine_sigma = tf.layers.dense(self.main_engine_stream, 1,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 trainable=True)
        self.main_engine_sigma = self.bounded_output(self.main_engine_sigma, 0, 1)

        self.lateral_engine_mu = tf.layers.dense(self.lateral_engine_stream, 1, activation=tf.nn.tanh,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 trainable=True)
        self.lateral_engine_sigma = tf.layers.dense(self.lateral_engine_stream, 1,
                                                    kernel_initializer=tf.orthogonal_initializer,
                                                    trainable=True)
        self.lateral_engine_sigma = self.bounded_output(self.lateral_engine_sigma, 0, 1)

        self.main_engine_dist = tf.distributions.Normal(self.main_engine_mu, self.main_engine_sigma)
        self.main_engine_sample = tf.squeeze(self.main_engine_dist.sample(1), axis=0)
        # self.main_engine_prob = self.main_engine_dist.prob(self.main_engine_sample)
        # self.main_engine_log_prob = tf.log(self.main_engine_prob + 1e-5)
        self.main_engine_log_prob = self.main_engine_dist.log_prob(self.main_engine_sample)

        self.lateral_engine_dist = tf.distributions.Normal(self.lateral_engine_mu, self.lateral_engine_sigma)
        self.lateral_engine_sample = tf.squeeze(self.lateral_engine_dist.sample(1), axis=0)
        # self.lateral_engine_prob = self.lateral_engine_dist.prob(self.lateral_engine_sample)
        # self.lateral_engine_log_prob = tf.log(self.lateral_engine_prob + 1e-5)
        self.lateral_engine_log_prob = self.lateral_engine_dist.log_prob(self.lateral_engine_sample)

        # Training
        self.main_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='action_placeholder1')
        self.lateral_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='action_placeholder2')

        self.old_log_prob_main_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32,
                                                                   name='old_log_prob_main_engine')
        self.old_log_prob_lateral_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32,
                                                                      name='old_log_prob_lateral_engine')

        self.new_log_prob_main_engine = self.main_engine_dist.log_prob(self.main_engine_placeholder)
        self.new_log_prob_lateral_engine = self.lateral_engine_dist.log_prob(self.lateral_engine_placeholder)

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')
        self.rewards_to_go_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards_to_go')

        # main_engine
        self.main_engine_ratio = tf.exp(
            self.new_log_prob_main_engine - self.old_log_prob_main_engine_placeholder)
        self.main_engine_surrogate_loss_1 = tf.math.multiply(self.main_engine_ratio, self.scaled_advantage_placeholder)
        # self.main_engine_surrogate_loss_2 = tf.where(self.scaled_advantage_placeholder > 0,
        #                                              (1 + clip_param)*self.scaled_advantage_placeholder,
        #                                              (1 - clip_param)*self.scaled_advantage_placeholder)
        self.main_engine_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.main_engine_ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.main_engine_loss = -tf.reduce_mean(
            tf.minimum(self.main_engine_surrogate_loss_1, self.main_engine_surrogate_loss_2))
        # self.main_engine_optimiser = tf.train.AdamOptimizer(actor_learning_rate_main_engine,
        #                                                     name='actor_optimizer_main_engine')
        # self.main_engine_gradients, self.main_engine_variables = zip(
        #     *self.main_engine_optimiser.compute_gradients(self.main_engine_loss))
        # self.main_engine_gradients, _ = tf.clip_by_global_norm(self.main_engine_gradients, 5.0)
        # self.training_op_actor_main_engine = self.main_engine_optimiser.apply_gradients(
        #     zip(self.main_engine_gradients, self.main_engine_variables))

        # lateral_engine
        self.lateral_engine_ratio = tf.exp(
            self.new_log_prob_lateral_engine - self.old_log_prob_lateral_engine_placeholder)
        self.lateral_engine_surrogate_loss_1 = tf.math.multiply(self.lateral_engine_ratio,
                                                                self.scaled_advantage_placeholder)
        self.lateral_engine_surrogate_loss_2 = tf.where(self.scaled_advantage_placeholder > 0,
                                                     (1 + clip_param)*self.scaled_advantage_placeholder,
                                                     (1 - clip_param)*self.scaled_advantage_placeholder)
        # self.lateral_engine_surrogate_loss_2 = tf.math.multiply(
        #     tf.clip_by_value(self.lateral_engine_ratio, 1 - clip_param, 1 + clip_param),
        #     self.scaled_advantage_placeholder)
        self.lateral_engine_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.lateral_engine_ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)

        self.lateral_engine_loss = -tf.reduce_mean(
            tf.minimum(self.lateral_engine_surrogate_loss_1, self.lateral_engine_surrogate_loss_2))

        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate,
                                                      name='actor_optimizer_main_engine')
        # self.lateral_engine_gradients, self.lateral_engine_variables = zip(
        #     *self.actor_optimizer.compute_gradients(self.total_loss))
        # self.lateral_engine_gradients, _ = tf.clip_by_global_norm(self.lateral_engine_gradients, 5.0)
        # self.action_op = self.actor_optimizer.apply_gradients(
        #     zip(self.lateral_engine_gradients, self.lateral_engine_variables))

        self.total_loss = tf.add(self.main_engine_loss, self.lateral_engine_loss)
        self.action_op = self.actor_optimizer.minimize(self.total_loss)

        # self.lateral_engine_gradients, self.lateral_engine_variables = zip(
        #     *self.actor_optimizer.compute_gradients(self.total_loss))
        # self.lateral_engine_gradients, _ = tf.clip_by_global_norm(self.lateral_engine_gradients, 5.0)
        # self.action_op = self.actor_optimizer.apply_gradients(
        #     zip(self.lateral_engine_gradients, self.lateral_engine_variables))

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower



