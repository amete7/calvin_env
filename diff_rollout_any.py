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
# from gpt_prior_global import GPT, GPTConfig
from diffusion_prior import get_sample
from unet import ConditionalUnet1D
from diffusers import DDPMScheduler, DDIMScheduler
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
    max_indices = 8
    dummy_indices = [cfg.prior.pad_token[0]]
    indices = dummy_indices
    for _ in range(max_indices):
        x = torch.tensor(indices).unsqueeze(0).to(device)
        # print(attach_emb[1].dtype)
        with torch.no_grad():
            logits = gpt_prior_model(x, None, attach_emb, attach_pos)
        next_token = logits[0,-1,:].argmax().item()
        indices.append(next_token)
        # print(indices)
    return torch.tensor(indices[1:]).unsqueeze(0).to(device)

def get_init_state(idx):
    processed_data_path = cfg.paths.processed_data_path
    custom_dataset = CustomDataset_Cont(processed_data_path)
    data = custom_dataset[idx]
    env_state = data['env_state'][0].numpy()
    robot_state = env_state[:15]
    scene_state = env_state[15:]
    return robot_state, scene_state

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    save_video = True
    idx = 285 + 20
    # lang_prompt = "open the cabinet drawer"
    lang_prompt = "slide the door to the left side"
    # lang_prompt = "push the switch upwards"
    # lang_prompt = "pick up the red block from the table"
    # lang_prompt = "pick up the blue block"
    # lang_prompt = "toggle the button to turn on the green light"

    model_ckpt = cfg.paths.model_weights_path
    priot_ckpt = cfg.paths.prior_weights_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_video:
        output_video_path = f'diff_prior_data_{idx//57}.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # attach_pos = cfg.prior.attach_pos
    # gpt_config = GPTConfig(vocab_size=cfg.prior.vocab_size, block_size=cfg.prior.block_size, output_dim=cfg.prior.output_dim, discrete_input=True)
    # gpt_prior_model = GPT(gpt_config).to(device)
    # state_dict = torch.load(priot_ckpt, map_location='cuda')
    # gpt_prior_model.load_state_dict(state_dict)
    # gpt_prior_model = gpt_prior_model.to(device)
    # gpt_prior_model.eval()
    # print('gpt_prior_model_loaded')
    net = ConditionalUnet1D(
        input_dim=11, 
        local_cond_dim=None,
        global_cond_dim=512+1031,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True
    )
    ckpt = torch.load(priot_ckpt)
    net.load_state_dict(ckpt)
    net = net.to(device)
    print('net_loaded')
    noise_scheduler = DDIMScheduler(num_train_timesteps=100)

    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(model_ckpt, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print('model_loaded')

    # robot_init_state, scene_state = get_init_state(idx)
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset()
    # print(observation.keys())
    for i in range(6):
        front_rgb = observation['rgb_obs']['rgb_static']
        gripper_rgb = observation['rgb_obs']['rgb_gripper']
        robot_state = observation['robot_obs']
        # print(robot_state[:6])
        robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
        robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
        # print(robot_state.shape,'robot_state_shape')
        front_emb = get_clip_features(front_rgb)
        gripper_emb = get_clip_features(gripper_rgb)
        lang_emb = get_language_features(lang_prompt)
        init_emb = torch.cat((front_emb,gripper_emb,robot_state),dim=-1).float().to(device)
        # attach_emb = (lang_emb,init_emb)
        global_cond = torch.cat([init_emb, lang_emb], -1)
        # print(global_cond.shape,'global_cond_shape')
        # print(lang_emb.shape,'lang_emb_shape')
        # print(lang_emb.device,'lang_emb_device')
        # print(init_emb.device,'init_emb_device')
        with torch.no_grad():
            indices = get_sample(noise_scheduler, net, global_cond, num_inference_steps=1000, batch_size=1, shape=(8, 11), device=0, codebook_size=1000)
        print(indices,'indices')
        # return
        # indices = torch.randint(0, 1000, (skill_block_size,)).to(device)
        with torch.no_grad():
            z = model.vq.indices_to_codes(indices)
        # z = z.unsqueeze(0).to(device)
        # print(z.shape,'z_shape')
        with torch.no_grad():
            action = model.decode(z, init_emb).squeeze(0).cpu().numpy()
            # action = model.decode_eval(z, front_emb).squeeze(0).cpu().numpy()
        # print(action)
        action[:,-1] = (((action[:,-1] >= 0) * 2) - 1).astype(int)
        # return
        done = False
        step_idx = 0
        for timestep in range(len(action)):
            action_to_take = action[timestep].copy()
            # action_to_take[-1] = int((int(action[timestep][-1] >= 0) * 2) - 1)
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