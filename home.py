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

from rlgym.utils.terminal_conditions import TerminalCondition

from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition,TimeoutCondition,GoalScoredCondition
from rlgym.utils.reward_functions import CombinedReward

import numpy as np


#class TouchBall(TerminalCondition):
#    def reset(self, initial_state: GameState):
#        pass
#
#    def is_terminal(self, current_state: GameState) -> bool:
#        return current_state.last_touch != -1
#
#class SpeedReward(RewardFunction):
#    def reset(self, initial_state: GameState):
#        pass
#
#    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#        linear_velocity = player.car_data.linear_velocity
#        reward = math.vecmag(linear_velocity)
#
#        return reward
#
#    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#        return 0
#
#default_tick_skip = 8
#physics_ticks_per_second = 120
#ep_len_seconds = 10
def exit_save(model):
    model.save("models/exit_save")
frame_skip = 8          # Number of ticks to repeat an action
half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5
fps = 120 / frame_skip

#max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],


env = rlgym.make(
    #spawn_opponents=True,
    terminal_conditions=[NoTouchTimeoutCondition(fps*45),TimeoutCondition(fps*300),GoalScoredCondition()],
    reward_fn=CombinedReward(
(
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0,
                ),
            ),
        (0.1, 1.0, 1.0)),
)
try:
    model = stable_baselines3.PPO.load(
        "models/exit_save.zip",
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

    print("Creating new model")

callback = CheckpointCallback(round(250_000 ), save_path="models", name_prefix="rl_model")
#print(terminal_conditions)
#model.learn(total_timesteps=int(1e6))

model.learn(total_timesteps=int(1000),callback=callback, progress_bar=True)
model.save("models/exit_save")
print("testzdpfzdfiodfoeijgfzkjgflfkejglmfkjmlzkgjmflkzjfglkjelfkmgjelmkfjgmlekzfjgmlzkjefg")

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