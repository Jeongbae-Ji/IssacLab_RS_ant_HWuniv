# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training environments for the Ant with a wider range of box heights, with and without a terrain curriculum.

The rewards, observations, actions and terminations are the same as in :class:`AntEnvCfg`.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg
from .ant_terrain_env_cfg import FrictionEventCfg
from .curriculum import LaneTerrainGeneratorCfg, terrain_levels_progress

# 10 lanes along x (240 m long, 16 m wide). The box height range grows from lane 0 to lane 9.
LANE_TERRAIN_CFG = LaneTerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=15,
    num_cols=10,
    num_start_rows=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.3, grid_height_range=(0.0, 0.18), platform_width=2.0
        ),
        "boxes_mid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.0, 0.18), platform_width=2.0
        ),
        "boxes_large": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.6, grid_height_range=(0.0, 0.18), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.0, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Lanes of flat, box-grid and rough tiles with increasing box height."""


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=terrain_levels_progress, params={"promote_distance": 30.0, "demote_distance": 15.0}
    )


@configclass
class AntWideTerrainEnvCfg(AntEnvCfg):
    """Configuration for the Ant walking environment with box heights up to 18 cm and no curriculum.

    Every robot stays in a randomly assigned lane.
    """

    events: FrictionEventCfg = FrictionEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = LANE_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None


@configclass
class AntCurriculumEnvCfg(AntWideTerrainEnvCfg):
    """Configuration for the Ant walking environment with box heights up to 18 cm and a terrain curriculum.

    Every robot starts in the easiest lane and moves to harder lanes as it walks further.
    """

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
