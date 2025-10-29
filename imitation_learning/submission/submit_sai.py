import os
import numpy as np
from sai_rl import SAIClient
from model import BoosterModel
## Initialize the SAI client
sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")

print("\033[93m"
      "⚠️  WARNING: Models must be retrained before submission.\n"
      "   Submitting the same model again will result in a duplicate model error in SAI."
      "\033[0m")

## Make the environment
env = sai.make_env()

script_dir = os.path.dirname(os.path.abspath(__file__))
## Create the model
model = BoosterModel(model_path=f"{script_dir}/model.pt")

class Preprocessor():

    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis = 0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis = 0)
        
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.concatenate([
            obs,
            project_gravity,
            base_ang_vel,
        ], dtype= np.float32, axis=1)

        return obs

## Watch
# sai.watch(model, preprocessor_class=Preprocessor)

## Benchmark the model locally
sai.benchmark(model, preprocessor_class=Preprocessor,)

sai.submit(name="baseline", model=model, preprocessor_class=Preprocessor)