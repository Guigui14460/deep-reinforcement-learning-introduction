import gym
import torch.multiprocessing as mp

from networks import ActorCritic


class Worker(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_id, n_games, t_max):
        super(Worker, self).__init__()
        self.local_actor_critic = ActorCritic(
            input_dims, n_actions, gamma=gamma)
        self.global_actor_critic = global_actor_critic
        self.name = f"w{name:02d}"
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.n_games = n_games
        self.t_max = t_max

    def run(self):
        t_step = 1
        while self.episode_idx.value < self.n_games:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.t_max == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param.grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(
                f"{self.name}  episode: {self.episode_idx.value} reward: {score:.2f}")
