from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import collections
from tqdm import tqdm
import torch
from model import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import traceback
from calvin_env.envs.tasks import Tasks

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

def init_min_max(stats_file_path):
    data = np.load(stats_file_path,allow_pickle=True)
    min_action = data['min_action']
    max_action = data['max_action']
    return min_action,max_action

def normalize_data(obs,min_obs,max_obs):
    n_obs = 2 * (obs - min_obs) / (max_obs - min_obs) - 1
    # n_action = 2 * (action - min_action) / (max_action - min_action) - 1
    return n_obs

def denormalize(naction,min_action,max_action):
    original_action = 0.5 * (naction + 1) * (max_action - min_action) + min_action
    return original_action

def main(cfg):
    tasks_dict = cfg.tasks['tasks']
    tasks_instance = Tasks(tasks_dict)
    max_steps = 1000
    obs_horizon = 2
    pred_horizon = 16
    action_horizon = 8
    action_dim = 7
    obs_dim = 39
    return True
    save_video = False
    min_action,max_action = init_min_max('/satassdscratch/scml-shared/calvin_data/task_D_D/stats_actions.npz')
    ckpt_path = "/satassdscratch/scml-shared/calvin_data/task_D_D/ema_noise_pred_net_nor_action.pth"
    
    if save_video:
        output_video_path = 'output_video.mp4'
        frame_size = (800,800)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    device = torch.device('cuda')

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    
    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_noise_pred_net = noise_pred_net
    # _ = ema_noise_pred_net.to(device)
    ema_noise_pred_net.load_state_dict(state_dict)
    print('Pretrained weights loaded.')
    _ = ema_noise_pred_net.to(device)

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset()
    obs = np.concatenate((observation['robot_obs'], observation['scene_obs']),axis=-1)
    obs_deque = collections.deque([obs]*obs_horizon,maxlen=obs_horizon)
    done = False
    step_idx = 0
    with tqdm(total=max_steps, desc="Eval Uncond_DiffPol") as pbar:
        while not done:
            B=1
            nobs = np.stack(obs_deque)
            # nobs = normalize_data(obs_seq,min_obs,max_obs)
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            # print(nobs,'nobs_outside')
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)
                # print(naction.device)
                # print(obs_cond.device)
                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            naction = denormalize(naction,min_action,max_action)
            start = obs_horizon - 1
            end = start + action_horizon
            action = naction[start:end,:]
            # print(action)
            for timestep in range(len(action)):
                action_to_take = action[timestep].copy()
                action_to_take[-1] = int((int(action[timestep][-1] >= 0) * 2) - 1)
                action_to_take_abs = ((action_to_take[0],action_to_take[1],action_to_take[2]),(action_to_take[3],action_to_take[4],action_to_take[5]),(action_to_take[-1],))
                # print(action_to_take_abs)
                observation, reward, done, info = env.step(action_to_take_abs)
                if save_video:
                    rgb = env.render(mode="rgb_array")[:,:,::-1]
                    video_writer.write(rgb)
                obs = np.concatenate((observation['robot_obs'], observation['scene_obs']),axis=-1)
                # print(obs)
                obs_deque.append(obs)
                step_idx += 1
                pbar.update(1)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    if save_video:
        # Release the VideoWriter
        video_writer.release()
        print(f"Video saved to {output_video_path}")

if __name__== "__main__":
    
    # /Users/shivikasingh/Desktop/ML-LS/Dataset/an_calvin/calvin/calvin_env/conf

    with initialize(config_path="conf"):
        print("config path:")
        cfg = compose(config_name="config_data_collection.yaml",overrides=["cameras=static_and_gripper"])
        cfg.env["use_egl"] = False
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
        # print(cfg.env)

    try:
        # Your code that may raise an exception here
        main(cfg)
    except Exception as e:
        # Handle the exception or simply do nothing to suppress the local variable display
        traceback.print_exc()