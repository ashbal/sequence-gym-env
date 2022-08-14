import gym
from gym.envs.registration import register


register(
    id='sequence-v0',
    entry_point='gym_sequence.envs:SeqEnv',
    nondeterministic=True,
)
# env = gym.make('sequence-v0')
env = gym.make('gym_sequence:sequence-v0')