import os

import gym
import gym_sokoban

from agac.agac import AGAC
from core.cmd_util import make_vec_env
from core.tf_util import linear_schedule
from core.callbacks import EvalCallback

load_path = None#"best/best_model.zip"
is_render = True
eval_num = 10

for env_id in ["Boxoban-Train-v0"]:
    for seed in [123]:
        log_dir = "./logs/%s/AGAC_seed%s" % (env_id, seed)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        if load_path is not None:
            print("loading model...")
            model = AGAC.load(os.path.join(log_dir, load_path))
            for i in range(eval_num):
                obs = env.reset()
                dones = False
                while not dones:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = env.step(action)
                    env.render()

        else:
            os.makedirs(log_dir, exist_ok=True)
            # CnnPolicy, MlpPolicy
            model = AGAC('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=2048, nminibatches=32, agac_c=linear_schedule(0.0004), beta_adv=0.0004,
                     learning_rate=0.0005, ent_coef=0.01, episodic_count=False)
            callback = EvalCallback(eval_env = env, eval_freq=10000, log_path=os.path.join(log_dir, "eval_logs"), best_model_save_path=os.path.join(log_dir, "best"), render=is_render)
            model.learn(total_timesteps=10000000, tb_log_name="fixed_choice_room_10M", callback=callback)

