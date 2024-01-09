from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import traceback
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
import sys
sys.path.append('/home/amete7/diffusion_dynamics/diff_skill/code')
from model_conv_final import SkillAutoEncoder
from gpt_prior import GPT, GPTConfig
from dataset.dataset_calvin import CustomDataset_Cont

import matplotlib.pyplot as plt
import imageio
import os

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)
                
def get_clip_features(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    # image_features = image_features.cpu()
    return image_features

def get_language_features(text):
    inputs = processor(text=text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        language_features = clip_model.get_text_features(**inputs)
    # language_features = language_features.cpu()
    return language_features

def get_indices(gpt_prior_model, attach_emb, attach_pos, device):
    max_indices = 9
    dummy_indices = [cfg.prior.pad_token[0],cfg.prior.pad_token[1]]
    indices = dummy_indices
    for _ in range(max_indices):
        x = torch.tensor(indices).unsqueeze(0).to(device)
        # print(attach_emb[1].dtype)
        with torch.no_grad():
            logits = gpt_prior_model(x, None, attach_emb, attach_pos)
        next_token = logits[0,-1,:].argmax().item()
        indices.append(next_token)
        # print(indices)
    return torch.tensor(indices[2:]).unsqueeze(0).to(device)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    save_video = False
    idx = 0
    # idx = 245
    # idx = 105
    # idx = 1132
    # idx = 46785
    # idx = 785
    lang_prompt = "push the switch upwards"

    model_ckpt = cfg.paths.model_weights_path
    priot_ckpt = cfg.paths.prior_weights_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_video:
        output_video_path = 'encoder_decoder.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_Cont(processed_data_path)
    # attach_pos = cfg.prior.attach_pos
    # gpt_config = GPTConfig(vocab_size=cfg.prior.vocab_size, block_size=cfg.prior.block_size, output_dim=cfg.prior.output_dim, discrete_input=True)
    # gpt_prior_model = GPT(gpt_config).to(device)
    # state_dict = torch.load(priot_ckpt, map_location='cuda')
    # gpt_prior_model.load_state_dict(state_dict)
    # gpt_prior_model = gpt_prior_model.to(device)
    # gpt_prior_model.eval()
    # print('gpt_prior_model_loaded')
    # # print(observation.keys())
    # front_rgb = observation['rgb_obs']['rgb_static']
    # gripper_rgb = observation['rgb_obs']['rgb_gripper']
    # robot_state = observation['robot_obs']
    # # print(robot_state[:6])
    # robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
    # robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
    # # print(robot_state.shape,'robot_state_shape')
    # front_emb = get_clip_features(front_rgb)
    # gripper_emb = get_clip_features(gripper_rgb)
    # lang_emb = get_language_features(lang_prompt)
    # init_emb = torch.cat((front_emb,gripper_emb,robot_state),dim=-1).float().to(device)
    # attach_emb = (lang_emb,init_emb)
    # # print(lang_emb.shape,'lang_emb_shape')
    # # print(lang_emb.device,'lang_emb_device')
    # # print(init_emb.device,'init_emb_device')
    # with torch.no_grad():
    #     indices = get_indices(gpt_prior_model, attach_emb, attach_pos, device)
    # print(indices,'indices')
    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(model_ckpt, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print('model_loaded')

    for og_idx in range(11):
        image_paths = []
        start_index = 33*og_idx
        for idx in range(start_index, start_index+33):
            if idx%33==0:
                complete_indices = []
                print('next skill')
            data = custom_dataset[idx]
            obs = data['obs'].unsqueeze(0).to(device)
            action = data['action'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                latent = model.encode(obs, action)
                z, indices = model.vq(latent)
            indices = indices.squeeze(0).cpu().numpy()
            
            if idx%33==0 or idx%33==32:
                complete_indices.append(indices)
            if idx%33==32:
                complete_indices = np.concatenate(complete_indices, axis=0)
                # Create a bar plot of the indices
                plt.bar(np.arange(len(complete_indices)), complete_indices)
                plt.title(f'Indices at skill {og_idx}')
                plt.xlabel('Index')
                plt.ylabel('Value')

                # Save the plot as an image
                image_path = f'indices_32_res_{og_idx}.png'
                plt.savefig(image_path)
                plt.close()

                # # Add the image path to the list
                # image_paths.append(image_path)

        # # Create a GIF from the images
        # images = [imageio.imread(image_path) for image_path in image_paths]
        # imageio.mimsave(f'indices{og_idx}.gif', images, duration=500)

        # # Remove the image files
        # for image_path in image_paths:
        #     os.remove(image_path)
        
    # return
    # indices = torch.randint(0, 1000, (skill_block_size,)).to(device)
    # with torch.no_grad():
    #     z = model.vq.indices_to_codes(indices)
    # z = z.unsqueeze(0).to(device)
    # print(z.shape,'z_shape')
    # init_emb = obs[:, 0, ...]
    # with torch.no_grad():
    #     action = model.decode(z, init_emb).squeeze(0).cpu().numpy()
    #     # action = model.decode_eval(z, front_emb).squeeze(0).cpu().numpy()
    # # print(action)
    # # return
    # done = False
    # step_idx = 0
    # for timestep in tqdm(range(len(action))):
    #     action_to_take = action[timestep].copy()
    #     action_to_take[-1] = int((int(action[timestep][-1] >= 0) * 2) - 1)
    #     action_to_take_abs = ((action_to_take[0],action_to_take[1],action_to_take[2]),(action_to_take[3],action_to_take[4],action_to_take[5]),(action_to_take[-1],))
    #     # print(action_to_take_abs)
    #     observation, reward, done, info = env.step(action_to_take_abs)
    #     if save_video:
    #         rgb = env.render(mode="rgb_array")[:,:,::-1]
    #         video_writer.write(rgb)
    #     step_idx += 1
    #     if step_idx > max_steps:
    #         done = True
    #     # if done:
    #     #     break
    # if save_video:
    #     video_writer.release()
    #     print(f"Video saved to {output_video_path}")

if __name__== "__main__":
    
    # /Users/shivikasingh/Desktop/ML-LS/Dataset/an_calvin/calvin/calvin_env/conf

    with initialize(config_path="conf"):
        # print("config path:")
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
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