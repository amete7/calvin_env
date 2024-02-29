from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import traceback
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import sys
import collections
from utils import top_k_sampling, top_p_sampling, greedy_sampling, get_top_k_probs, beam_search, random_sampling
sys.path.append('/home/amete7/diffusion_dynamics/diff_skill/code')
from model_v2 import SkillAutoEncoder
from gpt_prior_global import GPT, GPTConfig
from dataset.dataset_calvin import CustomDataset_Cont, CustomDataset_ABCD_lang_no_pad, CustomDataset_ABCD_lang

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

def get_language_features(text):
    inputs = processor(text=text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        language_features = clip_model.get_text_features(**inputs)
    # language_features = language_features.cpu()
    return language_features

def get_beam_search_indices(model, attach_emb, max_len, device):
    start_token = cfg.prior.pad_token[0]
    outputs = beam_search(start_token, model, attach_emb, max_len, device, 5, 1.0)
    print(outputs)
    return outputs

def get_indices(model, attach_emb, max_len, device):
    indices = [cfg.prior.pad_token[0]]
    with torch.no_grad():
        for i in range(max_len):
            x = torch.tensor(indices).unsqueeze(0).to(device)
            outs = model(x, None, attach_emb, [0,0])
            logits = outs[0,-1,:]
            index = top_p_sampling(logits.cpu().numpy(), 0.9, temperature=1.1)
            # index = random_sampling(logits.cpu().numpy())
            indices.append(index)
    print(indices)
    return indices[1:]

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    save_video = True
    # 49, 54, 57, 60, 66
    traj_idx = 49

    model_ckpt = cfg.paths.model_weights_path
    prior_ckpt_gpt = cfg.paths.prior_weights_path_gpt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = 'val' if 'val' in cfg.paths.processed_data_path else 'train'
    ndpt = 64+(cfg.action_horizon//2)-cfg.action_horizon+1
    if save_video:
        if cfg.use_prior:
            output_video_path = f'{cfg.action_horizon}_{data_type}_{cfg.prior_type}_roll_{traj_idx}_beam_30.mp4'
        else:
            output_video_path = f'{cfg.action_horizon}_{data_type}_endec_roll_{traj_idx}_top_k_t.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_ABCD_lang(processed_data_path,pred_horizon=cfg.action_horizon, pad_length=cfg.action_horizon//2)
    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(model_ckpt, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print('model_loaded')
    
    if cfg.prior_type == 'gpt':
        gpt_config = GPTConfig(
            vocab_size=cfg.prior.vocab_size, block_size=cfg.prior.block_size,
            output_dim=cfg.prior.output_dim, discrete_input=True,
            n_head=cfg.prior.n_head, n_layer=cfg.prior.n_layer, n_embd = cfg.prior.n_embd,
            obs_size=cfg.prior.obs_dim,
            )
        gpt_prior_model = GPT(gpt_config).to(device)
        state_dict = torch.load(prior_ckpt_gpt, map_location='cuda')
        gpt_prior_model.load_state_dict(state_dict)
        gpt_prior_model = gpt_prior_model.to(device)
        gpt_prior_model.eval()
        print('gpt_prior_model_loaded')
    else:
        raise NotImplementedError(f'prior type {cfg.prior_type} not implemented')
    idx = traj_idx*ndpt + 2
    data = custom_dataset[idx]
    lang_emb = data['lang_emb'].unsqueeze(0).to(device)
    # lang_emb = get_language_features('pull the switch downwards to turn off the yellow bulb')
    env_state = data['complete_state'][0].numpy()
    robot_state = env_state[:15]
    scene_state = env_state[15:]
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset(robot_state,scene_state)
    # print('env_reset')
    robot_state = observation['robot_obs']
    robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
    robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
    scene_state = torch.tensor(observation['scene_obs']).unsqueeze(0).to(device)
    obs = torch.cat((scene_state,robot_state),dim=-1).float().to(device)
    for i in range(cfg.num_steps):
        with torch.no_grad():
            prior_obs = model.obs_mlp(obs)
            # indices = get_indices(gpt_prior_model, (lang_emb, prior_obs), cfg.action_horizon, device)
            indices = get_beam_search_indices(gpt_prior_model, (lang_emb, prior_obs), cfg.action_horizon, device)
        print(indices,'indices from prior')
        z = model.vq.indices_to_codes(torch.tensor(indices).unsqueeze(0).to(device))
        # print(z.shape,'z shape')
        with torch.no_grad():
            action = model.decode(z, obs).squeeze(0).cpu().numpy()
        # print(action,'action')
        for timestep in tqdm(range(cfg.action_horizon)):
            action_to_take = action[timestep].copy()
            # print(action_to_take,'action_to_take')
            # return
            action_to_take[-1] = int((int(action[timestep][-1] >= 0) * 2) - 1)
            action_to_take_abs = ((action_to_take[0],action_to_take[1],action_to_take[2]),(action_to_take[3],action_to_take[4],action_to_take[5]),(action_to_take[-1],))
            # print(action_to_take_abs)
            observation, reward, done, info = env.step(action_to_take_abs)
            # print(info,'info')
            if save_video:
                rgb = env.render(mode="rgb_array")[:,:,::-1]
                video_writer.write(rgb)
            robot_state = observation['robot_obs']
            robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
            robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
            scene_state = torch.tensor(observation['scene_obs']).unsqueeze(0).to(device)
            obs = torch.cat((scene_state,robot_state),dim=-1).float().to(device)
            # if done:
            #     break
    print('##############################################')
    print(data['lang_prompt'])
    print('##############################################')
    
    if save_video:
        video_writer.release()
        print(f"Video saved to {output_video_path}")

if __name__== "__main__":
    
    # /Users/shivikasingh/Desktop/ML-LS/Dataset/an_calvin/calvin/calvin_env/conf

    with initialize(config_path="conf"):
        # print("config path:")
        cfg = compose(config_name="config_data_collection_m2_state.yaml", overrides=["cameras=static_and_gripper"])
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