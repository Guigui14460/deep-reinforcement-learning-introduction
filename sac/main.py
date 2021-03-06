import gym
from gym import wrappers
import numpy as np
import pybullet_envs

from agent import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 250
    # uncomment this line and do a mkdir tmp && mkdir tmp/video if you want to
    # record video of the agent playing the game.
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    # env = wrappers.Monitor(
    #     env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    # load_checkpoint = True

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'Episode {i}, score: {score:.2f}, avg_score: {avg_score:.2f}')

    if not load_checkpoint:
        x = list(range(1, n_games + 1))
        plot_learning_curve(x, score_history, figure_file)
