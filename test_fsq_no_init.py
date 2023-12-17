from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2
import traceback
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
from model_sv_no_init import SkillAutoEncoder


model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
                
def get_clip_features(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

def main(cfg):
    max_steps = 1000
    skill_block_size = 32
    save_video = True
    ckpt_path = "/satassdscratch/scml-shared/calvin_data/task_D_D/ckpt_sv_no_init_450.bin"
    if save_video:
        output_video_path = 'output_video.mp4'
        frame_size = (400,400)
        fps = 15
        # Initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'xvid' for MP4, 'MJPG' for AVI
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset()
    front_rgb = observation['rgb_obs']['rgb_static']

    model = SkillAutoEncoder(cfg.model)
    state_dict = torch.load(ckpt_path, map_location='cuda')
    state_dict.pop("vq._levels", None)
    state_dict.pop("vq._basis", None)
    state_dict.pop("vq.implicit_codebook", None)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    indices = torch.randint(0, 1000, (skill_block_size,)).to(device)
    z = model.vq.indices_to_codes(indices)
    z = z.unsqueeze(0).to(device)
    # print(z.shape,'z_shape')

    front_emb = get_clip_features(front_rgb)
    with torch.no_grad():
        action = model.decode_eval(z, None).squeeze(0).cpu().numpy()
    print(action)
    done = False
    step_idx = 0
    for timestep in range(len(action)):
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
        # Release the VideoWriter
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