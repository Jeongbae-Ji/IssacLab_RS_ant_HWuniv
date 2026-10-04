# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant environments with a height scanner added to the observations.

The height scanner measures the terrain height on a grid around the torso. Everything else is the same
as in the environments these configurations are derived from.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from . import ant_unseen_env_cfg as unseen
from .ant_env_cfg import AntEnvCfg
from .ant_terrain_env_cfg import AntTerrainEnvCfg


def add_height_scanner(cfg: AntEnvCfg):
    """Add a height scanner to the scene and its measurement to the policy observations."""
    # the sensor looks up the robot prim of every environment on the USD stage
    cfg.scene.clone_in_fabric = False
    # 2 m x 2 m grid around the torso with a spacing of 0.2 m (121 rays)
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    cfg.observations.policy.height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )


def _with_height_scanner(base_cfg: type) -> type:
    """Create a configuration class that adds the height scanner to the given configuration class."""

    @configclass
    class ScanEnvCfg(base_cfg):
        def __post_init__(self):
            super().__post_init__()
            add_height_scanner(self)

    return ScanEnvCfg


AntScanEnvCfg = _with_height_scanner(AntEnvCfg)
AntScanTerrainEnvCfg = _with_height_scanner(AntTerrainEnvCfg)
AntScanUnseenSlipperyEnvCfg = _with_height_scanner(unseen.AntUnseenSlipperyEnvCfg)
AntScanUnseenBoxesEnvCfg = _with_height_scanner(unseen.AntUnseenBoxesEnvCfg)
AntScanUnseenBoxesSlipperyEnvCfg = _with_height_scanner(unseen.AntUnseenBoxesSlipperyEnvCfg)
AntScanUnseenTallBoxesEnvCfg = _with_height_scanner(unseen.AntUnseenTallBoxesEnvCfg)
AntScanUnseenXTallBoxesEnvCfg = _with_height_scanner(unseen.AntUnseenXTallBoxesEnvCfg)
AntScanUnseenWaveEnvCfg = _with_height_scanner(unseen.AntUnseenWaveEnvCfg)
