from hydra import initialize, compose
import time
import hydra
import numpy as np
import cv2



def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    observation = env.reset()
    # print(env.get_info())
    # print(observation)
    #The observation is given as a dictionary with different values
    print(observation.keys())
    print(observation['robot_obs'])
    # print(observation['scene_obs'])

    for i in range(1000):
        # The action consists in a pose displacement (position and orientation)
        action_displacement = np.random.uniform(low=-1, high=1, size=6)
        # And a binary gripper action, -1 for closing and 1 for oppening
        action_gripper = np.random.choice([-1, 1], size=1)
        action = np.concatenate((action_displacement, action_gripper), axis=-1)
        observation, reward, done, info = env.step(action)
        # print(np.concatenate((observation['robot_obs'], observation['scene_obs']),axis=-1))
        # rgb = env.render(mode="rgb_array")[:,:,::-1]
        print(info['scene_info'].keys())
        break
        # rgb = rgb.astype(np.uint8)
        # cv2.imwrite('img.png', rgb)
        # print("img")

if __name__== "__main__":
    
    # /Users/shivikasingh/Desktop/ML-LS/Dataset/an_calvin/calvin/calvin_env/conf

    with initialize(config_path="conf"):
        print("config path:")
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        cfg.env["use_egl"] = False
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
        print(cfg.env)

    
    main(cfg)