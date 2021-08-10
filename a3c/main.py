import torch.multiprocessing as mp

from agent import Worker
from networks import ActorCritic
from optimizer import SharedAdam


if __name__ == '__main__':
    lr = 1e-4
    gamma = 0.99
    env_id = "CartPole-v0"
    n_actions = 2
    input_dims = [4]
    n_games = 3000
    t_max = 5
    global_actor_critic = ActorCritic(input_dims, n_actions, gamma)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(),
                       lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Worker(global_actor_critic, optim, input_dims, n_actions, gamma,
                      lr, i, global_ep, env_id, n_games, t_max) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]
