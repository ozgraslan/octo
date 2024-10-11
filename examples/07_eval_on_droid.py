"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from droid.robot_env import RobotEnv
from droid.misc.parameters import hand_camera_id, varied_camera_1_id
from envs.droid_env import DroidEnv

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     "checkpoint_weights_path", None, "Path to checkpoint", required=True
# )
# flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

# custom to droid robot
flags.DEFINE_bool("blocking", True, "Use the blocking controller")


flags.DEFINE_integer("im_size", 256, "Image size")
flags.DEFINE_string("video_save_path", "./", "Path to save video")
flags.DEFINE_integer("num_timesteps", 15, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 4, "Length of action sequence to execute/ensemble"
)


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """ no message for now. """
STEP_DURATION = 1

##############################################################################


def main(_):
    # set up the droid env
    camera_kwargs = {"hand_camera": {"resolution": (128, 128),
                                       "resize_func": "cv2"},
                     "varied_camera": {"resolution": (256, 256),
                                       "resize_func": "cv2"}}
    
    ## image obs name to camera id mapping
    ## choose camera ids and left-right
    camera_names_dict = {"primary": varied_camera_1_id + "_left"}

    robot_env = RobotEnv(action_space="cartesian_position", 
                         camera_kwargs=camera_kwargs)
    env = DroidEnv(env=robot_env, 
                   camera_names_dict=camera_names_dict,
                   blocking=True)
    
    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    # load models
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    print(model.get_pretty_spec())
    # wrap the robot environment
    env = HistoryWrapper(env, FLAGS.window_size)
    env = TemporalEnsembleWrapper(env, FLAGS.action_horizon)
    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    # env = RHCWrapper(env, FLAGS.action_horizon)

    # create policy functions
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            unnormalization_statistics=None,
        )
        # remove batch dim
        return actions[0]

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        modality = click.prompt(
            "Language or goal image?", type=click.Choice(["l", "g"])
        )

        if modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")
            # Format task for the model
            task = model.create_tasks(texts=[text])
            # For logging purposes
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        input("Press [Enter] to start.")

        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
