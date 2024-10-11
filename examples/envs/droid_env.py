import copy
import time

from typing import Dict

import numpy as np
import gym
import cv2

from droid.robot_env import RobotEnv
from droid.misc.parameters import hand_camera_id, varied_camera_1_id

class DroidEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env, ## RobotEnv
        camera_names_dict: Dict[str, str], ## image observation name to camera id mapping
        im_size: int = 256,
        blocking: bool = False,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    ## number of image observations can be change with fine-tuning
                    for i in camera_names_dict.keys()
                },

                ## joint angles (7 dof for franka arm) and gripper (not too sure)
                "proprio": gym.spaces.Box(
                    low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float32
                ),
            }
        )

        ## actions for end efector pose [x, y, z, roll, pitch, yaw] and gripper
        self.action_space = gym.spaces.Box(
            low=np.ones((7,)) * -1, high=np.ones((7,)), dtype=np.float32
        )
        self.camera_names_dict = camera_names_dict
        self._im_size = im_size
        self.blocking = blocking

    def get_obs(self):
        ## RobotEnv returns obsdict with keys: ['timestamp', 'robot_state', 
        ##                                      'image', 'camera_type', 
        ##                                      'camera_extrinsics', 
        ##                                      'camera_intrinsics'])
        env_obs = self._env.get_observation()
        
        ## image dict with keys: {camera serial}_{left, right}
        image_dict = env_obs["image"]

        ## get images from cameras given in self.camera_names_dict
        image_obs = {f"image_{i}": cv2.cvtColor(image_dict[self.camera_names_dict[i]], cv2.COLOR_BGRA2RGB) 
                     for i in self.camera_names_dict.keys()}

        ## robot state dict with keys: ['cartesian_position', 'gripper_position', 
        ##                              'joint_positions', 'joint_velocities', 
        ##                              'joint_torques_computed', 'prev_joint_torques_computed', 
        ##                              'prev_joint_torques_computed_safened', 'motor_torques_measured', 
        ##                              'prev_controller_latency_ms', 'prev_command_successful']
        robot_state = env_obs["robot_state"]

        ## not sure about gripper position
        ## CHECK: in bridge dataset/octo? 1 means open, in droid interface 1 means closed
        proprio = np.array(robot_state["joint_positions"] + [robot_state["gripper_position"]])        

        ## obs, info
        return {**image_obs, "proprio": proprio}, env_obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        randomize = (options is not None) and ("randomize" in options.keys()) and options["randomize"]
        self._env.reset(randomize=randomize)
        time.sleep(1)
        
        obs, obs_info = self.get_obs()

        return obs, {"obs_info": obs_info}

    def step(self, action):
        ## RobotEnv already has a step function
        ## but it does not pass the blocking param 
        ## to update robot call

        # Check Action
        assert len(action) == self._env.DoF
        if self._env.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)
        
        # Update Robot
        action_info = self._env.update_robot(
            action,
            action_space=self._env.action_space,
            gripper_action_space=self._env.gripper_action_space,
            blocking=self.blocking,
        )
        time.sleep(1)

        obs, obs_info = self.get_obs()

        ## how to set these ??
        reward, terminated, truncated = 0, False, False

        return obs, reward, terminated, truncated, {"action_info": action_info, "obs_info": obs_info}


if __name__ == "__main__":
    ## set image size to size octo trained with
    ## 256 for primary image
    ## 128 for wrist image
    
    ## apply camera_kwargs to these types of cameras (all cameras for droid)
    camera_kwargs = {"hand_camera": {"resolution": (128, 128),
                                       "resize_func": "cv2"},
                     "varied_camera": {"resolution": (256, 256),
                                       "resize_func": "cv2"}}
    
    ## image obs name to camera id mapping
    ## choose camera ids and left-right
    camera_names_dict = {"primary": varied_camera_1_id + "_left", 
                         "wrist": hand_camera_id + "_left"}

    robot_env = RobotEnv(action_space="cartesian_position", 
                         camera_kwargs=camera_kwargs)
    droid_env = DroidEnv(env=robot_env, 
                         camera_names_dict=camera_names_dict,
                         blocking=True)

    # print(droid_env)

    _, info = droid_env.reset()

    print(info["obs_info"]["robot_state"]["gripper_position"])
    robot_state = info["obs_info"]["robot_state"]
    action = robot_state["cartesian_position"] + [robot_state["gripper_position"]]
    action[0] -= 0.06
    action[-1] = 0.5
    obs, _, _, _, info = droid_env.step(np.array(action))

    print(info["obs_info"]["robot_state"]["gripper_position"])