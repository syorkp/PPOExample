class ActorCritic:
    def __init__(self, input_dims, clip_param, actor_learning_rate_main_engine, actor_learning_rate_lateral_engine,
                 critic_learning_rate):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.observation = tf.placeholder(shape=[None, input_dims], dtype=tf.float32, name='obs')
        self.hidden1 = tf.layers.dense(self.observation, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)
        self.hidden2 = tf.layers.dense(self.hidden1, 400, activation=tf.nn.relu,
                                       kernel_initializer=tf.orthogonal_initializer,
                                       trainable=True)

        self.Value_stream = tf.layers.dense(self.hidden2, 1, activation=tf.nn.relu,
                                            kernel_initializer=tf.orthogonal_initializer,
                                            trainable=True)
        self.Value_output = self.Value_stream

        self.main_engine_mu = tf.layers.dense(self.hidden2, 1, activation=tf.nn.relu,
                                              kernel_initializer=tf.orthogonal_initializer,
                                              trainable=True)
        self.main_engine_sigma = tf.layers.dense(self.hidden2, 1, activation=tf.nn.relu,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 trainable=True)
        self.lateral_engine_mu = tf.layers.dense(self.hidden2, 1, activation=tf.nn.relu,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 trainable=True)
        self.lateral_engine_sigma = tf.layers.dense(self.hidden2, 1, activation=tf.nn.relu,
                                                    kernel_initializer=tf.orthogonal_initializer,
                                                    trainable=True)

        self.main_engine_dist = tf.distributions.Normal(self.main_engine_mu, self.main_engine_sigma)
        self.main_engine_sample = tf.squeeze(self.main_engine_dist.sample(1), axis=0)
        self.main_engine_log_prob = self.main_engine_dist.prob(self.main_engine_sample[0][0])

        self.lateral_engine_dist = tf.distributions.Normal(self.main_engine_mu, self.main_engine_sigma)
        self.lateral_engine_sample = tf.squeeze(self.lateral_engine_dist.sample(1), axis=0)
        self.lateral_engine_log_prob = self.lateral_engine_dist.pro(self.lateral_engine_sample[0][0])

        # Training
        self.action_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='action_placeholder')

        self.old_log_prob_main_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32,
                                                                   name='old_log_prob_main_engine')
        self.log_prob_main_engine_placeholder = tf.log(
            self.main_engine_dist.prob(self.action_placeholder[0][0]), name="main_log_placeholder")

        self.old_log_prob_lateral_engine_placeholder = tf.placeholder(shape=[None], dtype=tf.float32,
                                                                      name='old_log_prob_lateral_engine')
        self.log_prob_lateral_engine_placeholder = tf.log(
            self.lateral_engine_dist.prob(self.action_placeholder[0][1]))

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')
        self.rewards_to_go_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards_to_go')

        # main_engine
        self.main_engine_ratio = tf.exp(
            self.log_prob_main_engine_placeholder - self.old_log_prob_main_engine_placeholder)
        self.main_engine_surrogate_loss_1 = tf.math.multiply(self.main_engine_ratio, self.scaled_advantage_placeholder)
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
            self.log_prob_lateral_engine_placeholder - self.old_log_prob_lateral_engine_placeholder)
        self.lateral_engine_surrogate_loss_1 = tf.math.multiply(self.lateral_engine_ratio,
                                                                self.scaled_advantage_placeholder)
        self.lateral_engine_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.lateral_engine_ratio, 1 - clip_param, 1 + clip_param),
            self.scaled_advantage_placeholder)
        self.lateral_engine_loss = -tf.reduce_mean(
            tf.minimum(self.lateral_engine_surrogate_loss_1, self.lateral_engine_surrogate_loss_2))

        # self.actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate_lateral_engine,
        #                                               name='actor_optimizer_main_engine')
        # self.lateral_engine_gradients, self.lateral_engine_variables = zip(
        #     *self.actor_optimizer.compute_gradients(self.total_loss))
        # self.lateral_engine_gradients, _ = tf.clip_by_global_norm(self.lateral_engine_gradients, 5.0)
        # self.action_op = self.actor_optimizer.apply_gradients(
        #     zip(self.lateral_engine_gradients, self.lateral_engine_variables))

        # Critic (state-value) loss function
        self.critic_loss = tf.reduce_mean(
            tf.squared_difference(tf.squeeze(self.Value_output), self.rewards_to_go_placeholder))

        self.total_loss = self.main_engine_loss + self.lateral_engine_loss + self.critic_loss

        self.critic_optimiser = tf.train.AdamOptimizer(critic_learning_rate,
                                                       name='actor_optimizer_main_engine')
        self.critic_gradients, self.critic_variables = zip(*self.critic_optimiser.compute_gradients(self.total_loss))
        self.critic_gradients, _ = tf.clip_by_global_norm(self.critic_gradients, 5.0)
        self.training_op_critic = self.critic_optimiser.apply_gradients(
            zip(self.critic_gradients, self.critic_variables))

def compute_advantage(reward_buffer, value_buffer):
    # UNUSED
    g = 0
    gamma = 0.99
    lmda = 0.95
    returns = []
    for i in reversed(range(1, len(reward_buffer))):
        delta = reward_buffer[i - 1] + gamma * value_buffer[i] - value_buffer[i - 1]
        g = delta + gamma * lmda * g
        returns.append(g + value_buffer[i - 1])
    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - value_buffer[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    returns = np.array(returns, dtype=np.float32)
    return returns, adv


def compute_rewards_to_go(rewards):
    # UNUSED
    gamma = 0.99
    rewards_to_go = []
    current_discounted_reward = 0
    for i, reward in enumerate(reversed(rewards)):
        current_discounted_reward = reward + current_discounted_reward * gamma
        rewards_to_go.insert(0, current_discounted_reward)
    return rewards_to_go