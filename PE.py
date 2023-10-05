# Imports
import gym
import minerl
import torch.multiprocessing as mp
import numpy as np
from collections import OrderedDict


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make("MineRLObtainDiamondShovel-v0")
    #env.seed(worker_id)

    while True:
        cmd, act = worker_end.recv()
        if cmd == 'step':
            # ob, reward, done, info = env.step(data)
        
            ob, reward, done, truncated, info = env.step(act)
            
            if done:
                ob, info = env.reset(options=dict(mode='random'))
            worker_end.send((ob['pov'], reward, done, truncated, info))
            
        elif cmd == 'reset':
            ob, info = env.reset(options=dict(mode='random'))
            worker_end.send(ob['pov'])
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob['pov'])
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, act):
        for master_end, a in zip(self.master_ends, act):
            master_end.send(('step', a))
        self.waiting = True
        

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, truncated, infos = zip(*results)
        
        # print(np.stack(obs))
        # print(np.stack(obs[0]['rgb_camera']).shape)
        return np.stack(obs['pov']), np.stack(rews), np.stack(dones), np.stack(truncated), np.stack(infos)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None, None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, a):
        self.step_async(a)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None, None))
        for worker in self.workers:
            worker.join()
            self.closed = True