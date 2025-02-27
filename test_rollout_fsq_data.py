from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import traceback
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms as torch_transforms
import sys
sys.path.append('/home/amete7/diffusion_dynamics/diff_skill/code')
from model import SkillAutoEncoder
from gpt_prior_global import GPT, GPTConfig
from dataset.dataset_calvin import CustomDataset_Cont
import torchvision

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)
                
net = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
net.eval()
net = torch.nn.Sequential(*list(net.children())[:-1]) # before linear
net = net.to(device)

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

resize_transform = torch_transforms.Compose([
        torch_transforms.ToPILImage(),
        torch_transforms.Resize((224, 224)),
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalization parameters for pretrained torchvision models
    ])

def get_resnet18_features(image_tensor):
    image_tensor = torch.tensor(image_tensor).permute(2,0,1)
    image_tensor = resize_transform(image_tensor).unsqueeze(0)
    with torch.no_grad():
        return net(image_tensor.to(device)).squeeze(2).squeeze(2)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    save_video = True
    # idx = 57 + 5 #20
    idx = 23484 + 5 # pick pink from slider
    # idx = 19152 + 2 # pick blue from slider
    # idx = 2793 + 5 # pick pink from drawer
    # idx = 798 + 2 # slide red block left
    # idx = 912 + 2 # rotate red left
    # idx = 1197 + 2 # stack blocks
    # idx = 2964 + 5 # stack
    # idx = 3363 + 5 # push pink left
    # idx = 9405 +2 # rotate pink left
    # idx = 12027 + 2 # pick red in slider
    # idx = 8850 + 20
    # idx = 285 + 20
    # idx = 1596 + 12
    # idx = 1710 + 8
    # idx = 171 + 31
    # lang_prompt = "push the switch upwards"

    model_ckpt = cfg.paths.model_weights_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_video:
        output_video_path = f'val_roll_endec_{idx//57}.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_Cont(processed_data_path)
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
    # print(observation['rgb_obs']['rgb_static'].shape)
    # return
    # print(observation['scene_obs'].shape)
    action_horizon = 8
    for i in range(64//action_horizon):
        data = custom_dataset[idx + action_horizon*i]
        obs = data['obs'].unsqueeze(0).to(device)
        action = data['action'].unsqueeze(0).to(device)
        with torch.no_grad():
            latent = model.encode(obs, action)
            z, indices = model.vq(latent)
        print(indices)
        init_emb = obs[:, 0, ...]
        if i != 0:
            front_rgb = observation['rgb_obs']['rgb_static']
            gripper_rgb = observation['rgb_obs']['rgb_gripper']
            robot_state = observation['robot_obs']
            # print(robot_state[:6])
            robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
            robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
            # scene_state = torch.tensor(observation['scene_obs']).unsqueeze(0).to(device)
            # print(robot_state.shape,'robot_state_shape')
            # front_emb = get_clip_features(front_rgb)
            # gripper_emb = get_clip_features(gripper_rgb)
            front_emb = get_resnet18_features(front_rgb)
            gripper_emb = get_resnet18_features(gripper_rgb)
            init_emb = torch.cat((front_emb,gripper_emb,robot_state),dim=-1).float().to(device)
            # init_emb = torch.cat((scene_state,robot_state),dim=-1).float().to(device)
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