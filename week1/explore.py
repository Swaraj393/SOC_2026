import gymnasium as gym

def my_policy(observation):
    """
    Return 0 (push left) or 1 (push right) based on the observation.
    observation = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    """
    
    return env.action_space.sample()
    

env = gym.make("LunarLander-v3", render_mode="human")

scores = []
for episode in range(5):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = my_policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    scores.append(total_reward)
    print(f"Episode {episode+1}: {total_reward}")

print(f"\nAverage over 5 episodes: {sum(scores)/len(scores):.1f}")
env.close()