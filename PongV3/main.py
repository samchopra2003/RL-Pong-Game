import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from ursina import *

from PPOTrainer import PPOTrainer
from PolicyNetwork import PolicyNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 2000
LOGGING_FREQ = 8
NRAND = 5
reward_file = './rewards.txt'

COMPUTER_DIFFICULTY = 0.88
DX = 0.1
DZ = -0.3


def update():
    global dx, dz, score_A, score_B
    ball_speed_inc = 1.05
    computer_diff = COMPUTER_DIFFICULTY

    # computer player
    computer_paddle_speed = abs(computer_diff * dx)
    if paddle_B.x < ball.x:
        if paddle_B.x < 0.36:
            paddle_B.x = paddle_B.x + computer_paddle_speed * time.dt
    else:
        if paddle_B.x > -0.36:
            paddle_B.x = paddle_B.x - computer_paddle_speed * time.dt

    global rollout_iter
    if rollout_iter < NRAND:
        action = np.random.choice([0, 1, 2])
    else:
        # image preprocessing
        dr = base.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()
        data = tex.getRamImage()
        img = np.array(data, dtype=np.uint8)
        img = img.reshape((tex.getYSize(), tex.getXSize(), 4))
        img = img[::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        downsampled_img = img
        for _ in range(3):
            downsampled_img = cv2.pyrDown(downsampled_img)
        trimmed_image = downsampled_img[40:-20, 40:-40]
        # cv2.imshow('img', trimmed_image)
        # print(downsampled_img.shape)

        trimmed_image = torch.tensor([trimmed_image], dtype=torch.float32, device=DEVICE)

        # probs = policy(torch.tensor([trimmed_image], dtype=torch.float32, device=DEVICE)) \
        #     .squeeze().cpu().detach().item()
        probs = policy(trimmed_image).squeeze().cpu().detach().item()
        action = np.where(np.random.rand(1) < probs, 0, 1)[0]
        # print("action = ", action)

        global ep_states, ep_actions, ep_old_probs
        ep_old_probs.append(probs)
        ep_actions.append(action)
        ep_states.append(trimmed_image)

    # AI player
    if action == 0:
        if paddle_A.x < 0.36:
            paddle_A.x = paddle_A.x + computer_paddle_speed * time.dt
            # print("RIGHT")
    else:
        if paddle_A.x > -0.36:
            paddle_A.x = paddle_A.x - computer_paddle_speed * time.dt
            # print("LEFT")

    ball.x = ball.x + time.dt * dx
    ball.z = ball.z + time.dt * dz

    # Boundary checking
    # Left and right
    if abs(ball.x) > 0.4:
        dx = -dx

    # Top and Bottom
    if ball.z > 0.25:
        score_B += 1
        print_on_screen(f"Player A : Player B = {score_A} : {score_B}", position=(-0.85, .45), scale=2, duration=0.5)
        reset()

    if ball.z < -0.65:
        score_A += 1
        print_on_screen(f"Player A : Player B = {score_A} : {score_B}", position=(-0.85, .45), scale=2, duration=0.5)
        reset()

    # Collisions
    hit_info = ball.intersects()
    if hit_info.hit:
        if hit_info.entity == paddle_A or hit_info.entity == paddle_B:
            dz = -dz * ball_speed_inc
            dx *= ball_speed_inc


def reset():
    ball.x = 0
    ball.z = 0

    global dx, dz
    dx = DX
    dz = DZ


if __name__ == "__main__":
    app = Ursina(window_type='offscreen')
    # app = Ursina()
    window.color = color.orange

    table = Entity(model='cube', color=color.green, scale=(10, 0.5, 14),
                   position=(0, 0, 0), texture="white_cube")

    paddle_A = Entity(parent=table, color=color.black, model='cube', scale=(0.15, 0.03, 0.05),
                      position=(0, 3.7, 0.22), collider='box')
    paddle_B = duplicate(paddle_A, z=-0.62)

    Text(text="Player A", scale=2, position=(-0.1, 0.32))
    Text(text="Player B", scale=2, position=(-0.1, -0.4))

    line = Entity(parent=table, model='quad', scale=(0.88, 0.2, 0.1), position=(0, 3.5, -0.20))
    ball = Entity(parent=table, model='sphere', color=color.red, scale=0.05,
                  position=(0, 3.71, -0.20), collider='box')

    dx = DX
    dz = DZ
    score_A = 0
    score_B = 0

    camera.position = (0, 15, -26)
    camera.rotation_x = 30

    # PPO policy parameters
    policy = PolicyNetwork().to(DEVICE)

    ppo_trainer = PPOTrainer(policy)

    step_size = 8
    alpha = 0.999
    scheduler = StepLR(ppo_trainer.policy_optim, step_size=step_size, gamma=alpha)

    # training hyperparameters
    tmax = 2000
    discount_rate = .99
    epsilon = .1
    beta = .01

    # all episode rewards
    rewards = []
    # clear previous contents
    with open(reward_file, 'w'):
        pass

    rollout_iter = 0
    # training loop
    for episode_idx in range(EPISODES):
        # episode results
        ep_rewards = []
        ep_states = []
        ep_actions = []
        ep_old_probs = []
        score_A = 0
        score_B = 0

        rollout_iter = 0
        # perform rollout
        for _ in range(tmax):
            prev_score_A = score_A
            prev_score_B = score_B
            # print(f"Score: {score_A}: {score_B}")
            app.step()

            if score_A != prev_score_A:
                ep_rewards.append(1)
            elif score_B != prev_score_B:
                ep_rewards.append(-1)
            else:
                ep_rewards.append(0)

            rollout_iter += 1
            if score_A == 11 or score_B == 11 or rollout_iter == tmax:
                # final score
                rewards.append(score_A - score_B)
                break

        # discard rewards from random actions
        ep_rewards = ep_rewards[NRAND:]

        ppo_trainer.train_policy(ep_old_probs, ep_states, ep_actions, ep_rewards, epsilon=epsilon, beta=beta)

        # the clipping parameter reduces as time goes on
        if (episode_idx + 1) % step_size == 0:
            epsilon *= alpha

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= 0.995

        scheduler.step()

        # display some progress
        if (episode_idx + 1) % LOGGING_FREQ == 0:
            print("Episode: {0:d}, score: {1:f}".format(episode_idx + 1, np.mean(rewards[-LOGGING_FREQ:])))

        # append mean reward to text file
        if len(rewards) > 0:
            with open(reward_file, 'a') as file:
                file.write(str(rewards[-1]) + '\n')

        # save policy
        if (episode_idx + 1) % 500 == 0:
            torch.save(policy, 'PPO_3d.policy')

    torch.save(policy, 'PPO_3d_final.policy')


