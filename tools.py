import gym
import numpy as np
import tensorflow.compat.v1 as tf
import scipy.signal as sig


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return sig.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_advantages_and_returns(reward_buffer, value_buffer, gamma, lmbda):
    delta = reward_buffer[:-1] + gamma * value_buffer[1:] - value_buffer[:-1]
    advantage = discount_cumsum(delta, gamma * lmbda)
    returns = discount_cumsum(reward_buffer, gamma)[:-1]
    return advantage, returns


def train_network_batches(reward_buffer, observation_buffer, action_buffer, main_engine_log_prob_buffer,
                          lateral_engine_log_prob_buffer, value_buffer, actor, critic, sess, batch_size, advantages, returns):
    # reward_buffer = np.array(reward_buffer)
    # observation_buffer = np.array(observation_buffer)
    # action_buffer = np.array(action_buffer)
    # main_engine_log_prob_buffer = np.squeeze(np.array(main_engine_log_prob_buffer))
    # lateral_engine_log_prob_buffer = np.squeeze(np.array(lateral_engine_log_prob_buffer))
    # value_buffer = np.squeeze(np.array(value_buffer))
    #
    # advantages, returns = get_advantages_and_returns(reward_buffer, value_buffer, 0.99, 0.9)
    # advantages = np.array(advantages)
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    # total_actor_loss = 0
    # total_critic_loss = 0
    if np.isnan(np.sum(advantages + returns)):
        print("Nan detected")

    for i in range(5):
        loss_main, loss_lateral, _,me_sigma, le_sigma = sess.run(
            [actor.main_engine_loss, actor.lateral_engine_loss, actor.action_op, actor.main_engine_sigma, actor.lateral_engine_sigma],
            feed_dict={
                actor.observation: observation_buffer.reshape((batch_size, 8)),
                actor.main_engine_placeholder: action_buffer[:, 0],
                actor.lateral_engine_placeholder: action_buffer[:, 1],

                actor.old_log_prob_main_engine_placeholder: main_engine_log_prob_buffer.reshape((batch_size)),
                actor.old_log_prob_lateral_engine_placeholder: lateral_engine_log_prob_buffer.reshape((batch_size)),

                actor.scaled_advantage_placeholder: advantages,
            }
        )
        # total_actor_loss += loss_main + loss_lateral

        loss_critic, _ = sess.run(
            [critic.critic_loss, critic.training_op_critic],
            feed_dict={
                critic.observation: observation_buffer,
                critic.returns_placeholder: returns,
                # critic.rewards_to_go_placeholder: rewards_to_go,
            }
        )

        # total_critic_loss += loss_critic
    # print(me_sigma, le_sigma)

    # return total_actor_loss/5, total_critic_loss
