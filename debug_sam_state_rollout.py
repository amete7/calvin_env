from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import traceback
from tqdm import tqdm
import torch
import sys
sys.path.append('/home/amete7/diffusion_dynamics/diff_skill/code')
from model import SkillAutoEncoder
from dataset.dataset_calvin import CustomDataset_Cont
from calvin_env.envs.tasks import Tasks

from sampler_utils import get_task_label

def get_indices(init_obs,lang_prompt,task_oracle,prior_dict):
    task_idx = get_task_label(lang_prompt,task_oracle)
    print(task_idx,'task_idx')
    data_index = []
    for i,idx in enumerate(prior_dict['task_id']):
        if idx == task_idx:
            data_index.append(i)
    data_obs = prior_dict['init_obs_emb'][data_index][:,6:24]
    distances = np.linalg.norm(data_obs - init_obs[:,6:24], axis=1)
    closest_index = np.argmin(distances)
    print(distances[closest_index],'distance in obs')
    return prior_dict['codes'][data_index[closest_index]], prior_dict['init_obs_emb'][data_index[closest_index]][:24]

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    tasks_dict = cfg.tasks['tasks']
    task_oracle = Tasks(tasks_dict)
    max_steps = 1000
    save_video = cfg.save_video
    # idx = 399 + 20
    # idx = 57 + 20
    # idx = 285 + 20
    # idx = 1596 + 12
    # idx = 1710 + 8
    # idx = 171 + 31
    # idx = 1938 + 5
    # idx = 798 + 5
    # idx = 1197 + 5
    # idx = 1026 + 5
    # lang_prompt = "open the cabinet drawer"
    # lang_prompt = "slide the door to the left side"
    # lang_prompt = "push the switch upwards"
    # lang_prompt = "pick up the red block from the table"
    # lang_prompt = "pick up the blue block"
    # lang_prompt = "toggle the button to turn on the green light"
    # lang_prompt = "grasp the pink block, then rotate it left"
    # lang_prompt = "grasp the blue block and lift it up"
    # lang_prompt = "grasp the red block from the drawer"
    # lang_prompt = "take the red block and rotate it left"

    ## validation data below
    # idx = 23484 + 5 # pick pink from slider###################
    # idx = 2793 + 5 # pick pink from drawer####################
    # idx = 798 + 2 # slide red block left######################
    # idx = 1197 + 2 # stack blocks#############################
    # idx = 9405 +2 # rotate pink left############################
    # idx = 5

    # lang_prompt = "lift the pink block lying in the cabinet"
    lang_prompt = "pick up the pink block from the drawer"
    # lang_prompt = "go slide the red block to the left"
    # lang_prompt = "stack the blocks on top of each other"
    # lang_prompt = "rotate the pink block to the left"################
    # lang_prompt = "slide the door to the left"

    # idx = 20188 + 5 # pick pink from slider###################
    idx = 2401 + 5 # pick pink from drawer####################
    # idx = 686 + 2 # slide red block left######################
    # idx = 1029 + 2 # stack blocks#############################
    # idx = 8085 +2 # rotate pink left############################


    # idx = 2964 + 5 # stack
    # idx = 3363 + 5 # push pink left
    # idx = 12027 + 2 # pick red in slider
    # idx = 19152 + 2 # pick blue from slider
    # idx = 912 + 2 # rotate red left
    # lang_prompt = "place the block on top of another block"
    # lang_prompt = "push the pink block to the left"
    # lang_prompt = "pick up the red block from the slider"
    # lang_prompt = "in the slider grasp the blue block"
    # lang_prompt = "rotate the red block towards the left"

    model_ckpt = cfg.paths.model_weights_path
    prior_dict = np.load(cfg.paths.sample_prior_data, allow_pickle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = 'val' if 'val' in cfg.paths.processed_data_path else 'train'
    ndpt = 64+(cfg.action_horizon//2)-cfg.action_horizon+1
    if save_video:
        if cfg.use_prior:
            output_video_path = f'{cfg.num_rollout}_{data_type}_{cfg.prior_type}_roll_{idx//ndpt}.mp4'
        else:
            output_video_path = f'{cfg.num_rollout}_{data_type}_endec_roll_{idx//ndpt}.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_Cont(processed_data_path,pred_horizon=cfg.action_horizon, pad_length=cfg.action_horizon//2)
    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(model_ckpt, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print('model_loaded')

    data = custom_dataset[idx]
    env_state = data['complete_state'][0].numpy()
    robot_state = env_state[:15]
    scene_state = env_state[15:]
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset(robot_state,scene_state)

    action_horizon = cfg.action_horizon
    for i in range(cfg.num_rollout):
        data = custom_dataset[idx + action_horizon*i]
        obs = data['lowdim_obs'].unsqueeze(0).to(device)
        action = data['action'].unsqueeze(0).to(device)
        with torch.no_grad():
            latent = model.encode(obs, action)
            z, indices_1 = model.vq(latent)
        print(indices_1, 'indices from encoder')
        init_emb = obs[:, 0, ...]
        if cfg.use_prior:
            robot_state = observation['robot_obs']
            robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
            robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
            scene_state = torch.tensor(observation['scene_obs']).unsqueeze(0).to(device)
            init_emb = torch.cat((scene_state,robot_state),dim=-1).float().to(device)
            # print(init_emb.cpu().numpy().shape,'init_emb')
            with torch.no_grad():
                indices, scene_obs = get_indices(init_emb.cpu().numpy(),lang_prompt,task_oracle,prior_dict)
                if i==0:
                    robot_state = observation['robot_obs']
                    scene_state = scene_obs
                    observation = env.reset(robot_state,scene_state)
                    if save_video:
                        rgb = env.render(mode="rgb_array")[:,:,::-1]
                        for i in range(30):
                            video_writer.write(rgb)
                    data = custom_dataset[idx]
                    env_state = data['complete_state'][0].numpy()
                    robot_state = env_state[:15]
                    scene_state = env_state[15:]
                    env = hydra.utils.instantiate(cfg.env)
                    observation = env.reset(robot_state,scene_state)
                indices = torch.tensor(indices).unsqueeze(0).to(device).long()
                z = model.vq.indices_to_codes(indices)
            print(indices,'indices from prior')
            print('############################################')
        # return
        with torch.no_grad():
            action = model.decode(z, init_emb).squeeze(0).cpu().numpy()
        done = False
        step_idx = 0
        for timestep in tqdm(range(len(action))):
            action_to_take = action[timestep].copy()
            action_to_take[-1] = int((int(action[timestep][-1] >= 0) * 2) - 1)
            action_to_take_abs = ((action_to_take[0],action_to_take[1],action_to_take[2]),(action_to_take[3],action_to_take[4],action_to_take[5]),(action_to_take[-1],))
            # print(action_to_take_abs)
            observation, reward, done, info = env.step(action_to_take_abs)
            # print(info,'info')
            if save_video:
                rgb = env.render(mode="rgb_array")[:,:,::-1]
                video_writer.write(rgb)
            step_idx += 1
            if step_idx > max_steps:
                done = True
            # if done:
            #     break
    if save_video:
        video_writer.release()
        print(f"Video saved to {output_video_path}")

if __name__== "__main__":
    
    # /Users/shivikasingh/Desktop/ML-LS/Dataset/an_calvin/calvin/calvin_env/conf

    with initialize(config_path="conf"):
        # print("config path:")
        cfg = compose(config_name="config_data_collection_state.yaml", overrides=["cameras=static_and_gripper"])
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