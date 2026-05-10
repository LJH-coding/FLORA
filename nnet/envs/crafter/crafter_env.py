import torch
import crafter
from nnet.structs import AttrDict

class CrafterEnv:
    def obs_space(self):
        return (["image", (3, self.size, self.size), torch.uint8],)

    def __init__(self, size, reward, ignore_health_reward, max_episode_steps, action_repeat = 1):
        self.env = crafter.Env(size=(size, size), length=max_episode_steps, reward=reward)
        self.ignore_health_reward = ignore_health_reward
        self.num_actions = self.env.action_space.n
        self.size = size
        self.action_repeat = action_repeat
        assert self.action_repeat == 1, "Action repeat must be 1 for Crafter"

    def sample(self):
        return torch.nn.functional.one_hot(torch.randint(low=0, high=self.num_actions, size=()), num_classes=self.num_actions).type(torch.float32)

    def preprocess(self, state, reward, done):
        state = torch.tensor(state)
        # (C, H, W)
        state = state.permute(2, 0, 1)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        is_last = done
        return state, reward, done, is_last

    def reset(self):

        state, _, _, _ = self.preprocess(self.env.reset(), 0, 0)
        reward = torch.tensor(0.0, dtype=torch.float32)
        done = torch.tensor(False, dtype=torch.float32)
        is_last = torch.tensor(False, dtype=torch.float32)
        is_first = torch.tensor(True, dtype=torch.float32)
        self.unlocked = self.env._unlocked.copy()
        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

    def step(self, action):
        state, reward, done, infos = self.env.step(action.item())
        if self.ignore_health_reward:
            reward = int(len(self.env._unlocked) != len(self.unlocked))
        self.unlocked = self.env._unlocked.copy()
        state, reward, done, is_last = self.preprocess(state, reward, done)
        is_first = torch.tensor(False, dtype=torch.float32)
        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)
    