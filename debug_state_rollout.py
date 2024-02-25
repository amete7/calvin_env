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
from model_v3 import SkillAutoEncoder
from gpt_prior_global import GPT, GPTConfig
from dataset.dataset_calvin import CustomDataset_Cont
from unet import ConditionalUnet1D
from diffusers import DDPMScheduler, DDIMScheduler
from diffusion_prior import get_sample

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

def get_indices(gpt_prior_model, attach_emb, attach_pos, device, max_indices=4):
    max_indices = max_indices
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

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    save_video = True
    # idx = 399 + 20
    # idx = 57 + 20
    # idx = 285 + 20
    # idx = 1596 + 12
    # idx = 1710 + 8
    # idx = 171 + 31
    # idx = 1938 + 5
    # idx = 798 + 5
    # idx = 1197 + 5
    idx = 1026 + 5
    # lang_prompt = "open the cabinet drawer"
    # lang_prompt = "slide the door to the left side"
    # lang_prompt = "push the switch upwards"
    # lang_prompt = "pick up the red block from the table"
    # lang_prompt = "pick up the blue block"
    # lang_prompt = "toggle the button to turn on the green light"
    # lang_prompt = "grasp the pink block, then rotate it left"
    # lang_prompt = "grasp the blue block and lift it up"
    # lang_prompt = "grasp the red block from the drawer"
    lang_prompt = "take the red block and rotate it left"

    ## validation data below
    # idx = 23484 + 5 # pick pink from slider###################
    # idx = 2793 + 5 # pick pink from drawer####################
    # idx = 798 + 2 # slide red block left######################
    # idx = 1197 + 2 # stack blocks#############################
    # idx = 9405 +2 # rotate pink left############################

    # lang_prompt = "lift the pink block lying in the cabinet"
    # lang_prompt = "pick up the pink block from the drawer"
    # lang_prompt = "go slide the red block to the left"
    # lang_prompt = "stack the blocks on top of each other"
    # lang_prompt = "rotate the pink block to the left"################

    # idx = 20188 + 5 # pick pink from slider###################
    # idx = 2401 + 5 # pick pink from drawer####################
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
    prior_ckpt = cfg.paths.prior_weights_path
    prior_ckpt_gpt = cfg.paths.prior_weights_path_gpt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = 'val' if 'val' in cfg.paths.processed_data_path else 'train'
    ndpt = 64+(cfg.action_horizon//2)-cfg.action_horizon+1
    if save_video:
        if cfg.use_prior:
            output_video_path = f'{cfg.num_rollout}_{data_type}_{cfg.prior_type}_roll_{idx//57}.mp4'
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

    if cfg.prior_type == 'diff':
        net = ConditionalUnet1D(
            input_dim=cfg.diff_prior.input_dim, 
            local_cond_dim=None,
            global_cond_dim=cfg.diff_prior.cond_dim,
            diffusion_step_embed_dim=cfg.diff_prior.time_emb,
            down_dims=cfg.diff_prior.down_dims,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True
        )
        ckpt = torch.load(prior_ckpt, map_location='cuda')
        net.load_state_dict(ckpt)
        net = net.to(device)
        net.eval()
        print('net_loaded')
        if cfg.diff_prior.schedule_type == 'ddpm':
            noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.diff_prior.diffusion_steps_train,beta_schedule=cfg.diff_prior.beta_schedule,)
        else:
            noise_scheduler = DDIMScheduler(num_train_timesteps=cfg.diff_prior.diffusion_steps_train,beta_schedule=cfg.diff_prior.beta_schedule,)

    elif cfg.prior_type == 'gpt':
        gpt_config = GPTConfig(vocab_size=cfg.prior.vocab_size, block_size=cfg.prior.block_size, 
                               output_dim=cfg.prior.output_dim, discrete_input=True, obs_size=cfg.prior.obs_size,
                               n_head=cfg.prior.n_head, n_layer=cfg.prior.n_layer, n_embd=cfg.prior.n_embd,)
        gpt_prior_model = GPT(gpt_config).to(device)
        state_dict = torch.load(prior_ckpt_gpt, map_location='cuda')
        gpt_prior_model.load_state_dict(state_dict)
        gpt_prior_model = gpt_prior_model.to(device)
        gpt_prior_model.eval()
        print('gpt_prior_model_loaded')
    else:
        raise NotImplementedError(f'prior type {cfg.prior_type} not implemented')

    data = custom_dataset[idx]
    env_state = data['complete_state'][0].numpy()
    robot_state = env_state[:15]
    scene_state = env_state[15:]
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset(robot_state,scene_state)

    # print(observation['scene_obs'].shape)
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
        lang_emb = get_language_features(lang_prompt)
        # if i != 0:
        if cfg.use_prior:
            robot_state = observation['robot_obs']
            robot_state = np.concatenate([robot_state[:6],[robot_state[14]]])
            robot_state = torch.tensor(robot_state).unsqueeze(0).to(device)
            scene_state = torch.tensor(observation['scene_obs']).unsqueeze(0).to(device)
            init_emb = torch.cat((scene_state,robot_state),dim=-1).float().to(device)
            with torch.no_grad():
                if cfg.prior_type == 'gpt':
                    indices = get_indices(gpt_prior_model, (lang_emb,init_emb), cfg.prior.attach_pos, device, max_indices=cfg.prior.block_size-1)
                else:
                    global_cond = torch.cat([init_emb, lang_emb], -1)
                    indices = get_sample(noise_scheduler, net, global_cond, num_inference_steps=cfg.diff_prior.diffusion_steps_eval, batch_size=1, shape=(cfg.diff_prior.code_length, cfg.diff_prior.input_dim), device=0, codebook_size=1000)
                z = model.vq.indices_to_codes(indices)
            print(indices,'indices from prior')
         
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
            print(info,'info')
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