import sys
sys.path.append(".")

import gym
import matplotlib.pyplot as plt

from util_helper.utils import device,get_screen



if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    print(obs)
    print(env.action_space)
    print(env.observation_space)
    print(env.reward_range)
    # plt.figure()

    # last_screen = get_screen(env)
    # plt.imshow(last_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #         interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    env.render()
    env.close()

    # plt.ioff()
    # plt.show()

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
