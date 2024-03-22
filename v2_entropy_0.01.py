import gym
import parser_necto
import rlgym as rlgym
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

import rlgym_sim as rlgym_sim

from rlgym_ppo import Learner
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.state_setters import RandomState


def rl_gym_fction():
    import parser_necto as parserNecto
    import reward_necto as rewardNecto
    import obs_necto as obsNecto

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = parser_necto.NectoAction()

    obs_builder = DefaultObs(
        pos_coef=np.array([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef= 1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef= 1 / common_values.CAR_MAX_ANG_VEL,
    )

    state_setter = RandomState(ball_rand_speed=True,
                               cars_rand_speed=True,
                               cars_on_ground=False)

    rewards = rewardNecto.NectoRewardFunction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        obs_builder=obs_builder,
        terminal_conditions=terminal_conditions,
        reward_fn=rewards,
        action_parser=action_parser,
        state_setter=state_setter,)

    return env

def test():
    import rlgym as rlgym
    import parser_necto as parserNecto
    import reward_necto as rewardNecto

    spawn_opponents = False
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = parser_necto.NectoAction()

    obs_builder = DefaultObs(
        pos_coef=np.array([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    state_setter = RandomState(ball_rand_speed=True,
                               cars_rand_speed=True,
                               cars_on_ground=False)

    rewards = rewardNecto.NectoRewardFunction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
    env = rlgym.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        obs_builder=obs_builder,
        terminal_conditions=terminal_conditions,
        reward_fn=rewards,
        action_parser=action_parser,
        state_setter=state_setter,
    )

    return env


from rlgym_ppo.util import MetricsLogger

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)

if __name__ == "__main__":
    metrics_logger =  ExampleLogger()
    n_proc = 15
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    checkpoints_save_folder = "GPU_model_save/"
    model_load_folder = "GPU_load_directory/"

    learner = Learner(rl_gym_fction,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      #render=True,
                      render_delay=0.025,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=10_000_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=False,
                      n_checkpoints_to_keep=10_000_000,
                      policy_layer_sizes=(2048, 1024, 1024, 1024),
                      critic_layer_sizes=(2048, 1024, 1024, 1024),
                      checkpoints_save_folder=checkpoints_save_folder,
                      checkpoint_load_folder=model_load_folder,
                      )
    learner.learn()

