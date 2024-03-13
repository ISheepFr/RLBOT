import gym
import rlgym
import stable_baselines3
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import common_values

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards import EventReward
from stable_baselines3.common.callbacks import ProgressBarCallback

from typing import Any

from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.obs_builders import AdvancedObs

from rlgym.utils.terminal_conditions import TerminalCondition

from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition,TimeoutCondition,GoalScoredCondition
from rlgym.utils.reward_functions import CombinedReward

import numpy as np

from rlgym.envs import Match
#from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs



frame_skip = 8          # Number of ticks to repeat an action
half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5
fps = 120 / frame_skip
def get_match():
    return Match(
        terminal_conditions=[NoTouchTimeoutCondition(fps*45),TimeoutCondition(fps*300),GoalScoredCondition()],
        obs_builder=AdvancedObs(),
        action_parser=DefaultAction(),
        state_setter=DefaultState(),
        reward_function=CombinedReward(
(
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=15.0,
                    save=30.0,
                    demo=10.0,

                ),
            ),
        (0.1, 1.0, 1.0)),


    )

#def get_match():
#    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
#    return Match(
#        reward_function=DefaultReward(),
#        terminal_conditions=[TimeoutCondition(225)],
#        obs_builder=DefaultObs(),
#        state_setter=DefaultState(),
#            action_parser=DefaultAction()
#    )

def test():

    env = rlgym.make(game_speed=1,obs_builder=AdvancedObs())
    model = stable_baselines3.PPO.load(
        "models/3_1e6+2.5e6+6.5e6+4.6e6_rl_model_7250000_steps.zip",
        env,
        device="auto"
    )

    print("Model loaded")

    while True:
        obs = env.reset()
        done = False

        while not done:
            action = model.predict(obs)
            #print("MODELE: ",action)
            #action = env.action_space.sample()
            action  = action[0]

            next_obs, reward, done, gameinfo = env.step(action)
            obs = next_obs

if __name__ == "__main__":
    #env  = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=1, wait_time=30)
   #learner = PPO(policy="MlpPolicy", env=env, verbose=1)
   #learner.learn(1_000_000)


#max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

#terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],

    #test()

    env = rlgym.make(
    #spawn_opponents=True,
    obs_builder=AdvancedObs(),
    terminal_conditions=[NoTouchTimeoutCondition(fps*45),TimeoutCondition(fps*300),GoalScoredCondition()],
    reward_fn=CombinedReward(
(
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=15.0,
                    save=30.0,
                    demo=10.0,

                ),
            ),
        (0.1, 1.0, 1.0)),
)
# TODO

    #env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=2, wait_time=30)
    #learner = stable_baselines3.PPO(policy="MlpPolicy", env=env, verbose=1)
    #learner.learn(1_000_000)

    try:
        model = stable_baselines3.PPO.load(
            "models/3_1e6+2.5e6+6.5e6+4.6e6_rl_model_7250000_steps",
            env,
            device="auto"
        )
        print("Loaded previous model")
    except:
        model = stable_baselines3.PPO(
        "MlpPolicy",
        env=env,
        n_epochs=10,
        learning_rate=5e-5,
        ent_coef=0.01,
        vf_coef=1.,
        verbose=3,
        device="auto"
    )
#
        print("Creating new model")
#
    callback = CheckpointCallback(round(100_000), save_path="models", name_prefix="3_1e6+2.5e6+6.5e6+4.6e6+7.25e6__rl_model")
    #print(terminal_conditions)
    #model.learn(total_timesteps=int(1e6))
#
    model.learn(total_timesteps=int(25e6),callback=callback)
    model.save("models/test_exit_save")
    #print("testzdpfzdfiodfoeijgfzkjgflfkejglmfkjmlzkgjmflkzjfglkjelfkmgjelmkfjgmlekzfjgmlzkjefg")
#
    #while True:
    #    obs = env.reset()
    #    done = False
#
    #    while not done:
    #        action = model.predict(obs)
    #        #print("MODELE: ",action)
    #        #action = env.action_space.sample()
    #        action  = action[0]
#
    #        next_obs, reward, done, gameinfo = env.step(action)
    #        obs = next_obs
