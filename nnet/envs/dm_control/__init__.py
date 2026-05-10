# DeepMind Control Envs
from .deep_mind_control_env import DeepMindControlEnv
from .acrobot import Acrobot
from .ball_in_cup import BallInCup
from .cartpole import Cartpole
from .cheetah import Cheetah
from .finger import Finger
from .hopper import Hopper
from .pendulum import Pendulum
from .quadruped import Quadruped
from .reacher import Reacher
from .walker import Walker

# DeepMind Control Envs Dictionary
dm_control_dict = {
    "Cheetah": Cheetah,
    "Walker": Walker,
    "Hopper": Hopper,
    "Pendulum": Pendulum,
    "Cartpole": Cartpole,
    "Reacher": Reacher,
    "Quadruped": Quadruped,
    "Acrobot": Acrobot,
    "Finger": Finger,
    "BallInCup": BallInCup,
}