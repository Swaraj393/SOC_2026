# Assignment 1: Explore CartPole

**Estimated time:** 2–3 hours
**Code you'll write:** ~30 lines
**Algorithms you'll implement:** Zero

The goal is to **see an RL environment in action** and build intuition for what an agent actually does. You will *not* train a learning agent this week. You'll just look, poke, and reflect.

## Setup Check

Before starting, make sure this runs without errors:

```bash
python setup/verify_setup.py
```

You should see ✅ for `gymnasium`. If not, fix that first.

## Part 1 — Watch a Random Agent (30 min)

Create a file `week1/explore.py` and paste this in:

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

print(f"Initial observation: {obs}")
print(f"Observation = [cart position, cart velocity, pole angle, pole angular velocity]")
print(f"Action space: {env.action_space}  (0 = push left, 1 = push right)")

total_reward = 0
steps = 0

done = False
while not done:
    action = env.action_space.sample()  # pick a random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"\nEpisode ended after {steps} steps with total reward {total_reward}")
env.close()
```

Run it:

```bash
python week1/explore.py
```

A window should pop up showing a cart with a pole. The pole will fall almost immediately because the agent is making random choices.

**Run it 5 times.** Write down how many steps it lasted each time. You'll probably see numbers between 10 and 40.

### Reflect (write in a file `week1/notes.md`)

1. What does each of the 4 numbers in the observation mean? (Hint: it's printed above.)
2. Why does the pole fall so quickly?
3. The reward is `+1` per step the pole stays up. What's the maximum total reward possible? (Look up `CartPole-v1` in the Gymnasium docs.)

## Part 2 — Write a Hand-Coded Rule (1 hour)

Random actions are bad. Let's see if a *very simple* rule can do better. No learning yet — just a hand-written `if` statement.

Modify your script:

```python
import gymnasium as gym

def my_policy(observation):
    """
    Return 0 (push left) or 1 (push right) based on the observation.
    observation = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    """
    
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    if pole_angle > 0:
        return 1  # push right
    else:
        return 0  # push left
    

env = gym.make("CartPole-v1", render_mode="human")

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
```

### Your task

Write a rule inside `my_policy` that does **better than random** (random averages ~20). Aim for an average of **at least 50** over 5 episodes.

**You don't need a clever rule.** A one-line `if` statement based on `pole_angle` will already get you there. Try it.

### Reflect (in `notes.md`)

4. Write down the rule you used.
5. What was your average score?
6. Did the rule work every time, or did it sometimes fail? Why?
7. Can you think of a smarter rule? (You don't have to implement it — just describe it.)

## Part 3 — Try a Different Environment (30 min)

CartPole isn't the only environment. Pick **one** of these and run it with random actions (no hand-coded rule needed):

- `MountainCar-v0` — a car has to climb a hill but can't accelerate fast enough on its own
- `Acrobot-v1` — two linked joints try to swing up
- `LunarLander-v2` — land a spaceship on a pad (this one's fun)

Same template, just change the environment name:

```python
env = gym.make("MountainCar-v0", render_mode="human")
# ... rest same as Part 1
```

### Reflect (in `notes.md`)

8. Which environment did you try?
9. What is the agent trying to do? (Describe in plain English, no math.)
10. What does a state look like? What are the possible actions?
11. Does random play ever solve it? Why or why not?

## Part 4 — Big-Picture Reflection (30 min)

Now you've seen RL environments up close. In `notes.md`, answer in 2–4 sentences each:

12. **In your own words**, what is the agent's goal in any RL problem?
13. Why is "random" not a good strategy — and what would the agent need to do instead?
14. What does it mean for an agent to "learn"? (Hint: what should change between episode 1 and episode 100?)
15. Write down **3 questions** you have about RL that you'd like to ask your mentor.

## What to Submit / Show Your Mentor

By end of week, you should have:

- `week1/explore.py` — your modified script with the hand-coded rule
- `week1/notes.md` — your answers to all 15 reflection questions

That's it. No report, no plots, no algorithm implementation.

## Common Issues

**"The window doesn't open"**
On some systems (especially over SSH or in containers) `render_mode="human"` won't work. Use `render_mode="rgb_array"` instead and skip watching live — you can still see the reward.

**"My rule scores worse than random"**
That's fine and informative — write it down in your notes. Check the sign of `pole_angle`: positive means leaning one way, negative the other. Try flipping your `if`.

**"I want to do more"**
Great. Don't. Save your energy for Week 2 when we start actually training agents. If you're really itching, read [HuggingFace Deep RL — Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction).

## Grading

There is no grade. The point is:

- ✅ You set up a working environment
- ✅ You saw a random agent fail
- ✅ You built a tiny hand-coded controller
- ✅ You wrote down what you observed in your own words

If you did those four things, you're ready for Week 2.

---

← Back to [Week 1 README](../README.md)
