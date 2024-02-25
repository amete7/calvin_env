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
from model import SkillAutoEncoder
from gpt_prior_global import GPT, GPTConfig
from dataset.dataset_calvin import CustomDataset_Cont

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
    save_video = True
    # idx = 57 + 5 #20
    # idx = 23484 + 5 # pick pink from slider
    # idx = 19152 + 2 # pick blue from slider
    idx = 2793 + 5 # pick pink from drawer
    # idx = 2964 + 5
    # idx = 3363 + 5
    # idx = 8850 + 20
    # idx = 285 + 20
    # idx = 1596 + 12
    # idx = 1710 + 8
    # idx = 171 + 31
    # lang_prompt = "push the switch upwards"

    model_ckpt = cfg.paths.model_weights_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_video:
        output_video_path = f'val_encdec_{idx//57}.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_Cont(processed_data_path)

    data = custom_dataset[idx]
    obs = data['lowdim_obs'].unsqueeze(0).to(device)
    action = data['action'].unsqueeze(0).to(device)
    env_state = data['complete_state'][0].numpy()
    robot_state = env_state[:15]
    scene_state = env_state[15:]
    print(obs.shape,'obs_shape')
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset(robot_state,scene_state)

    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(model_ckpt, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print('model_loaded')
    with torch.no_grad():
        latent = model.encode(obs, action)
        z, indices = model.vq(latent)
    print(indices)
    init_emb = obs[:, 0, ...]
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