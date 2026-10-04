# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# training environments with randomized terrain and/or friction
for _name, _cfg in [
    ("Terrain", "ant_terrain_env_cfg:AntTerrainEnvCfg"),
    ("Terrain-Only", "ant_terrain_env_cfg:AntTerrainOnlyEnvCfg"),
    ("Terrain-Push", "ant_terrain_env_cfg:AntTerrainPushEnvCfg"),
    ("Friction-Only", "ant_terrain_env_cfg:AntFrictionOnlyEnvCfg"),
    ("Wide-Terrain", "ant_curriculum_env_cfg:AntWideTerrainEnvCfg"),
    ("Curriculum", "ant_curriculum_env_cfg:AntCurriculumEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# held-out evaluation environments (not used for training)
for _name, _cfg in [
    ("Slippery", "AntUnseenSlipperyEnvCfg"),
    ("Boxes", "AntUnseenBoxesEnvCfg"),
    ("Boxes-Slippery", "AntUnseenBoxesSlipperyEnvCfg"),
    ("Tall-Boxes", "AntUnseenTallBoxesEnvCfg"),
    ("XTall-Boxes", "AntUnseenXTallBoxesEnvCfg"),
    ("Wave", "AntUnseenWaveEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-Unseen-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_unseen_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# environments with a height scanner in the observations (the original, training and held-out environments)
for _name, _cfg in [
    ("Scan", "AntScanEnvCfg"),
    ("Scan-Terrain", "AntScanTerrainEnvCfg"),
    ("Scan-Unseen-Slippery", "AntScanUnseenSlipperyEnvCfg"),
    ("Scan-Unseen-Boxes", "AntScanUnseenBoxesEnvCfg"),
    ("Scan-Unseen-Boxes-Slippery", "AntScanUnseenBoxesSlipperyEnvCfg"),
    ("Scan-Unseen-Tall-Boxes", "AntScanUnseenTallBoxesEnvCfg"),
    ("Scan-Unseen-XTall-Boxes", "AntScanUnseenXTallBoxesEnvCfg"),
    ("Scan-Unseen-Wave", "AntScanUnseenWaveEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_scan_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )
