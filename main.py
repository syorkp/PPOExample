import sklearn
import sklearn.preprocessing
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import math
from network import Actor, Critic
import scipy
from tools import *
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

env = gym.envs.make("LunarLanderContinuous-v2")
env._max_episode_steps = 200

# network = ActorCritic(env.observation_space.shape[0], 0.2, 0.001, 0.001, 0.002)
actor = Actor(env.observation_space.shape[0], 0.2, 0.00025)
critic = Critic(env.observation_space.shape[0], 0.00025)

episodes = 4000
batch_size = 20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reward_list = []
    reward_list2 = []
    actor_loss_buffer = []
    critic_loss_buffer = []
    done = False

    for episode in range(episodes):
        if done:
            reward_buffer = np.array(reward_buffer)
            observation_buffer = np.array(observation_buffer)
            action_buffer = np.array(action_buffer)
            main_engine_log_prob_buffer = np.squeeze(np.array(main_engine_log_prob_buffer))
            lateral_engine_log_prob_buffer = np.squeeze(np.array(lateral_engine_log_prob_buffer))
            value_buffer = np.squeeze(np.array(value_buffer))

            advantages, returns = get_advantages_and_returns(reward_buffer, value_buffer, 0.99, 0.9)
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            for batch_num in range(batch_size, len(advantages), batch_size):
                train_network_batches(reward_buffer[batch_num-batch_size:batch_num],
                                      observation_buffer[batch_num-batch_size:batch_num],
                                      action_buffer[batch_num-batch_size:batch_num],
                                      main_engine_log_prob_buffer[batch_num-batch_size:batch_num],
                                      lateral_engine_log_prob_buffer[batch_num-batch_size:batch_num],
                                      value_buffer[batch_num-batch_size:batch_num],
                                      actor,
                                      critic,
                                      sess,
                                      batch_size,
                                      advantages[batch_num-batch_size:batch_num],
                                      returns[batch_num-batch_size:batch_num])

        state = env.reset()
        reward_total = 0
        steps = 0
        done = False

        observation_buffer = []
        action_buffer = []
        reward_buffer = []
        value_buffer = []
        main_engine_log_prob_buffer = []
        lateral_engine_log_prob_buffer = []

        while not done:
            # if steps != 0 and steps % batch_size == 0:
            #     # Check for nans and wrong p
            #     to_check = np.concatenate((np.array(action_buffer).flatten(), main_engine_log_prob_buffer, lateral_engine_log_prob_buffer))
            #     if np.isnan(np.sum(to_check)):
            #         print("Nan detected")
            #
            #     train_network_batches(reward_buffer, observation_buffer, action_buffer, main_engine_log_prob_buffer,
            #                           lateral_engine_log_prob_buffer, value_buffer, actor, critic, sess, batch_size)
            #
            #     observation_buffer = []
            #     action_buffer = []
            #     reward_buffer = []
            #     value_buffer = []
            #     main_engine_log_prob_buffer = []
            #     lateral_engine_log_prob_buffer = []
                # actor_loss_buffer.append(actor_loss)
                # critic_loss_buffer.append(critic_loss)

            main_engine, lateral_engine, main_engine_log_prob, lateral_engine_log_prob = sess.run(
                [actor.main_engine_sample, actor.lateral_engine_sample, actor.main_engine_log_prob,
                 actor.lateral_engine_log_prob],
                feed_dict={
                    actor.observation: state.reshape(1, 8),
                }
            )

            value = sess.run(critic.Value_output,
                             feed_dict={
                                 critic.observation: state.reshape(1, 8),
                             }
                             )

            action = np.array([main_engine[0][0], lateral_engine[0][0]])

            next_state, reward, done, _ = env.step(action)

            steps += 1
            reward_total += reward

            # Add to buffers
            reward_buffer.append(reward)
            observation_buffer.append(state)
            action_buffer.append(action)
            main_engine_log_prob_buffer.append(main_engine_log_prob[0][0])
            lateral_engine_log_prob_buffer.append(lateral_engine_log_prob[0][0])
            value_buffer.append(value[0])

            state = next_state

        reward_list.append(reward_total)
        reward_list2.append(reward_total)
        print("Episode: {}, Number of Steps : {}, Total reward: {:0.2f}".format(
            episode, steps, reward_total))
        # if episode % 50 == 0 and episode != 0:
        #     print("Episode: {}, Number of Steps : {}, Average reward: {:0.2f}".format(
        #         episode, steps, np.mean(reward_list)))
        #     reward_list = []

plt.plot(reward_list2)
plt.show()

plt.plot(actor_loss_buffer)
plt.show()
plt.plot(critic_loss_buffer)
plt.show()

