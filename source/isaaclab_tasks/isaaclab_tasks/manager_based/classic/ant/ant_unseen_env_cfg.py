# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-out evaluation environments for the Ant.

Only the terrain shape and the terrain friction differ from :class:`AntEnvCfg`. These environments
are used for evaluation only and must not be used for training.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg

# The Ant runs along +x for more than 100 m in one episode, so the terrain is a long strip:
# all robots start on the first row and the rows extend 150 m ahead of them.
UNSEEN_BOXES_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=24,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.03, 0.10), platform_width=2.0
        ),
    },
)
"""Random grid of boxes, similar to the example terrain in the assignment."""

UNSEEN_TALL_BOXES_TERRAIN_CFG = UNSEEN_BOXES_TERRAIN_CFG.replace(
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.10, 0.15), platform_width=2.0
        ),
    },
)
"""Random grid of boxes that are taller than the ones of the 0-10 cm training terrain."""

UNSEEN_XTALL_BOXES_TERRAIN_CFG = UNSEEN_BOXES_TERRAIN_CFG.replace(
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.18, 0.22), platform_width=2.0
        ),
    },
)
"""Random grid of boxes that are taller than the ones of the 0-18 cm training terrain."""

UNSEEN_WAVE_TERRAIN_CFG = UNSEEN_BOXES_TERRAIN_CFG.replace(
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.03, 0.08), num_waves=4, border_width=0.25
        ),
    },
)
"""Smooth waves, a terrain type that is not used for training."""

SLIPPERY_FRICTION = 0.2
"""Friction coefficient of the slippery terrains (the original terrain uses 1.0)."""


@configclass
class AntUnseenSlipperyEnvCfg(AntEnvCfg):
    """Flat terrain with low friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.physics_material.static_friction = SLIPPERY_FRICTION
        self.scene.terrain.physics_material.dynamic_friction = SLIPPERY_FRICTION


@configclass
class AntUnseenBoxesEnvCfg(AntEnvCfg):
    """Random grid of boxes with the original friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = UNSEEN_BOXES_TERRAIN_CFG
        # start every robot on the first row so that the whole strip lies ahead of it
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AntUnseenBoxesSlipperyEnvCfg(AntUnseenBoxesEnvCfg):
    """Random grid of boxes with low friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.physics_material.static_friction = SLIPPERY_FRICTION
        self.scene.terrain.physics_material.dynamic_friction = SLIPPERY_FRICTION


@configclass
class AntUnseenTallBoxesEnvCfg(AntUnseenBoxesEnvCfg):
    """Random grid of tall boxes with the original friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = UNSEEN_TALL_BOXES_TERRAIN_CFG


@configclass
class AntUnseenXTallBoxesEnvCfg(AntUnseenBoxesEnvCfg):
    """Random grid of extra tall boxes with the original friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = UNSEEN_XTALL_BOXES_TERRAIN_CFG


@configclass
class AntUnseenWaveEnvCfg(AntUnseenBoxesEnvCfg):
    """Wave terrain with the original friction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = UNSEEN_WAVE_TERRAIN_CFG
