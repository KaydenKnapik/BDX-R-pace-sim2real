# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
from .assets.robots.bdxr import BDX_R_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

BDXR_03_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee"],
    saturation_effort=60.0,
    effort_limit=42.0,
    velocity_limit=18.849,
    stiffness={".*": 78.957},
    damping={".*": 5.027},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},          # static friction coefficient (Nm)
    dynamic_friction={".*": 0.0},  # dynamic friction coefficient (Nm)
    viscous_friction={".*": 0.0},  # viscous friction coefficient (Nm s/rad)
    max_delay=10,                  # max delay in simulation steps
)

# 2. 02 Actuator: Ankles (17 Nm Saturation, 11.9 Nm Effort Limit)
BDXR_02_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_Ankle"],
    saturation_effort=17.0,
    effort_limit=11.9,
    velocity_limit=37.699,
    stiffness={".*": 16.581},
    damping={".*": 1.056},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)


@configclass
class BDXRPaceCfg(PaceCfg):
    """Pace configuration for BDX-R robot."""   
    robot_name: str = "bdxr_sim"
    data_dir: str = "bdxr_sim/chirp_data.pt"  # located in pace_sim2real/data/bdxr_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((41, 2))  # 10 + 10 + 10 + 10 + 1 = 41 parameters to optimize
    joint_order: list[str] =[
        "Left_Hip_Yaw",
        "Left_Hip_Roll",
        "Left_Hip_Pitch",
        "Left_Knee",
        "Left_Ankle",
        "Right_Hip_Yaw",
        "Right_Hip_Roll",
        "Right_Hip_Pitch",
        "Right_Knee",
        "Right_Ankle",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:10, 0] = 1e-5
        self.bounds_params[:10, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[10:20, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[20:30, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[30:40, 0] = -0.005
        self.bounds_params[30:40, 1] = 0.005  # bias between -0.005 - 0.005 [rad] (accurate encoders)
        self.bounds_params[40, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class BDXRPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for BDX-R robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = BDX_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators={"legs": BDXR_03_PACE_ACTUATOR_CFG, "ankles": BDXR_02_PACE_ACTUATOR_CFG})


@configclass
class BDXRPaceEnvCfg(PaceSim2realEnvCfg):

    scene: BDXRPaceSceneCfg = BDXRPaceSceneCfg()
    sim2real: PaceCfg = BDXRPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
