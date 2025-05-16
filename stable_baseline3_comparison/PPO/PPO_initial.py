import os
import random
import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 10_000
RENDER_FREQ = 50_000
ALGO = "PPO"
VIDEO_DIR = f"{ALGO}_initial_training_videos"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(render=False):
    env = MetaDriveEnv(dict(
        use_render=render,
        map="OSX",
        start_seed=SEED,
        image_observation=render,
        manual_control=render
    ))
    return env

def train():
    set_seed(SEED)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    train_env = make_env(render=False)
    obs, _ = train_env.reset()
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="logs/", seed=SEED)

    rewards = []
    timesteps = []

    for step in range(0, TOTAL_TIMESTEPS, EVAL_FREQ):
        model.learn(total_timesteps=EVAL_FREQ, reset_num_timesteps=False)

        # Evaluate
        mean_reward, _ = evaluate_policy(model, train_env, n_eval_episodes=5)
        rewards.append(mean_reward)
        timesteps.append(step + EVAL_FREQ)
        print(f"Step: {step + EVAL_FREQ}, Mean Reward: {mean_reward:.2f}")

        # Record a video every RENDER_FREQ timesteps
        if (step + EVAL_FREQ) % RENDER_FREQ == 0:
            print("\nRecording video...")
            train_env.config["use_render"] = True
            obs, _ = train_env.reset()
            frames = []

            for _ in range(1000):
                img = train_env.render(mode="topdown")
                if img is not None:
                    frames.append(img)
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = train_env.step(action)
                done = terminated or truncated
                if done:
                    obs, _ = train_env.reset()

            train_env.config["use_render"] = False

            video_path = os.path.join(VIDEO_DIR, f"video_{step + EVAL_FREQ}_{ALGO}.mp4")
            imageio.mimsave(video_path, frames, fps=15)
            print(f"[INFO] Saved video to {video_path}\n")

    train_env.close()

    # Plot rewards
    plt.plot(timesteps, rewards)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Reward")
    plt.title(f"{ALGO} Training Performance")
    plt.savefig(f"reward_curve_{ALGO}.png")
    plt.show()

    model.save(f"{ALGO}_metadrive_model")

if __name__ == "__main__":
    train()
