import gymnasium as gym
env = gym.make("CartPole-v1", render_mode=None)
obs, info = env.reset()
print("Initial observation:", obs)
print("Action space:", env.action_space)
env.close()