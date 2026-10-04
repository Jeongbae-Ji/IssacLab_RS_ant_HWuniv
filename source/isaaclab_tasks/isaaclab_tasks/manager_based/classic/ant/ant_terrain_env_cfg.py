# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training environment for the Ant with randomized terrain shape and friction.

The rewards, observations, actions and terminations are the same as in :class:`AntEnvCfg`.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg, EventCfg

# Only terrains that stay close to z=0 are used: the episode terminates when the torso drops below an
# absolute height, so pits and downward slopes would terminate a robot that is walking fine.
# The Ant runs along +x for more than 100 m in one episode, so the rows extend far ahead of the start rows.
TRAIN_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=30,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.3, grid_height_range=(0.0, 0.10), platform_width=2.0
        ),
        "boxes_mid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.0, 0.10), platform_width=2.0
        ),
        "boxes_large": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.6, grid_height_range=(0.0, 0.10), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Mix of flat, box-grid and rough tiles in random order."""


@configclass
class FrictionEventCfg(EventCfg):
    """Configuration for events with randomized friction."""

    # the contact friction is the average of the robot and the terrain friction (terrain: 1.0),
    # so this range gives a contact friction of 0.7 to 1.125
    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.25),
            "dynamic_friction_range": (0.4, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class PushEventCfg(FrictionEventCfg):
    """Configuration for events with randomized friction, initial base state and pushes."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.3, 0.3)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class AntTerrainOnlyEnvCfg(AntEnvCfg):
    """Configuration for the Ant walking environment with randomized terrain only."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TRAIN_TERRAIN_CFG
        # start on the first rows so that at least 150 m of terrain lies ahead of every robot
        self.scene.terrain.max_init_terrain_level = 10


@configclass
class AntFrictionOnlyEnvCfg(AntEnvCfg):
    """Configuration for the Ant walking environment with randomized friction only (flat terrain)."""

    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class AntTerrainEnvCfg(AntTerrainOnlyEnvCfg):
    """Configuration for the Ant walking environment with randomized terrain and friction."""

    events: FrictionEventCfg = FrictionEventCfg()


@configclass
class AntTerrainPushEnvCfg(AntTerrainOnlyEnvCfg):
    """Configuration for the Ant walking environment with randomized terrain, friction, initial state and pushes."""

    events: PushEventCfg = PushEventCfg()
