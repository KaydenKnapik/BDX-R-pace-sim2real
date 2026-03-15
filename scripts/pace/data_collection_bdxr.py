# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-BDX-R-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1)
parser.add_argument("--max_frequency", type=float, default=10.0)
parser.add_argument("--duration", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib.pyplot as plt  # <-- Added back for plotting
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root

def main():
    args_cli.task = "Isaac-Pace-BDX-R-v0"
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)
    num_joints = len(joint_ids)

    armature = torch.tensor([0.1] * num_joints, device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor([4.5] * num_joints, device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor([0.05] * num_joints, device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor([0.05] * num_joints, device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)
    
    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping)))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction
    
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = torch.arange(joint_ids.shape[0], device=joint_ids.device)[drive_indices]
        comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)

    phase = 2 * pi * (args_cli.min_frequency * t + ((args_cli.max_frequency - args_cli.min_frequency) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, num_joints), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    
    # --- YOUR PREFERRED TRAJECTORY SHAPING ---
    trajectory_bias = torch.tensor([0.0, 0.0, 0.3, -0.6, 0.3,   0.0, 0.0, 0.3, -0.6, 0.3], device=env.unwrapped.device)
    trajectory_scale = torch.tensor([0.2, 0.2, 0.5, 0.5, 0.5,   0.2, 0.2, 0.5, 0.5, 0.5], device=env.unwrapped.device)
    trajectory_directions = torch.tensor([1.0, 1.0, 1.0, -1.0, 1.0,   -1.0, -1.0, 1.0, -1.0, 1.0], device=env.unwrapped.device)

    trajectory[:, joint_ids] = (trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0)) * trajectory_directions.unsqueeze(0) * trajectory_scale.unsqueeze(0)

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias[0, joint_ids], joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros((1, num_joints), device=env.unwrapped.device), joint_ids=joint_ids)

    dof_pos_buffer = torch.zeros(num_steps, num_joints, device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, num_joints, device=env.unwrapped.device)
    
    counter = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dof_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids] - bias[0]
            
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[0, :num_joints] = trajectory[counter % num_steps, :num_joints]
            
            env.step(actions)
            dof_target_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"]._data.joint_pos_target[0, joint_ids]
            
            counter += 1
            # --- Added terminal prints back ---
            if counter % int(sample_rate) == 0:
                print(f"[INFO]: Step {counter/sample_rate:.1f} / {duration} seconds")
            
            if counter >= num_steps: break

    env.close()

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "time": t.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
    }, data_dir / "chirp_data.pt")
    
    print(f"🎉 Success! Sim data saved to: {data_dir}/chirp_data.pt")

    # --- Added Plotting back ---
    import matplotlib.pyplot as plt

    for i in range(num_joints):
        plt.figure()
        plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos")
        plt.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} target", linestyle='dashed')
        plt.title(f"Joint {joint_order[i]} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
    simulation_app.close()