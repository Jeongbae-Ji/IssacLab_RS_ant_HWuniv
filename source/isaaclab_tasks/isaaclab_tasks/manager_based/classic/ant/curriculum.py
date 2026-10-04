# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain curriculum for the Ant.

The Ant runs along +x for more than 100 m in one episode. The terrain is therefore laid out in lanes:
the difficulty is constant along x and increases across the lanes (y). The terrain level of a robot
is the lane it starts in.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LaneTerrainGenerator(TerrainGenerator):
    """Terrain generator with lanes of constant difficulty along x.

    The difficulty increases linearly over the columns (y) and the sub-terrain type is sampled randomly
    for every tile. The terrain origins are exposed with the shape (num_cols, num_start_rows, 3), so that
    the terrain importer treats the lanes as terrain levels and the first rows as spawn locations.
    """

    cfg: LaneTerrainGeneratorCfg

    def __init__(self, cfg: LaneTerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg, device)
        # keep only the first rows as spawn locations and use the lane as the first index
        self.terrain_origins = np.ascontiguousarray(self.terrain_origins[: cfg.num_start_rows].transpose(1, 0, 2))

    def _generate_curriculum_terrains(self):
        """Add terrains with the difficulty based on the column index."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        lower, upper = self.cfg.difficulty_range
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # randomly sample terrain index
                sub_index = self.np_rng.choice(len(proportions), p=proportions)
                # vary the difficulty parameter linearly over the number of columns
                difficulty = (sub_col + self.np_rng.uniform()) / self.cfg.num_cols
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])


@configclass
class LaneTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the lane terrain generator."""

    class_type: type = LaneTerrainGenerator

    curriculum: bool = True

    num_start_rows: int = 1
    """Number of rows (from the start of the lanes) that are used as spawn locations."""


def terrain_levels_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    promote_distance: float,
    demote_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot moved along +x during the episode.

    Robots that moved further than ``promote_distance`` go to a harder lane and robots that moved less
    than ``demote_distance`` go to an easier lane.

    Returns:
        The mean terrain level over all environments.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # compute the distance the robot moved towards the target
    distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    # the reset at the start of the simulation is not the end of an episode
    played = env.episode_length_buf[env_ids] > 0
    move_up = (distance > promote_distance) & played
    move_down = (distance < demote_distance) & played
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
