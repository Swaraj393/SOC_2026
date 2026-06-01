PART 1

observations :


Initial observation: [-0.02128793  0.00055258  0.02770845 -0.04134442]
Observation = [cart position, cart velocity, pole angle, pole angular velocity]
Action space: Discrete(2)  (0 = push left, 1 = push right)

Episode ended after 13 steps with total reward 13.0


Initial observation: [ 0.01960635  0.00515443  0.04065011 -0.00445052]
Observation = [cart position, cart velocity, pole angle, pole angular velocity]
Action space: Discrete(2)  (0 = push left, 1 = push right)

Episode ended after 17 steps with total reward 17.0



Initial observation: [ 0.00256987  0.03953074 -0.02007557  0.03310142]
Observation = [cart position, cart velocity, pole angle, pole angular velocity]
Action space: Discrete(2)  (0 = push left, 1 = push right)

Episode ended after 15 steps with total reward 15.0


Initial observation: [ 0.00416838  0.04615549 -0.01341645  0.03718137]
Observation = [cart position, cart velocity, pole angle, pole angular velocity]
Action space: Discrete(2)  (0 = push left, 1 = push right)

Episode ended after 13 steps with total reward 13.0


Initial observation: [-0.02628783 -0.01775549 -0.01477702 -0.0378525 ]
Observation = [cart position, cart velocity, pole angle, pole angular velocity]
Action space: Discrete(2)  (0 = push left, 1 = push right)

Episode ended after 22 steps with total reward 22.0



1. What does each of the 4 numbers in the observation mean? (Hint: it's printed above.)
The four numbers represent:

Cart Position: The horizontal position of the cart on the track (0.0 is the center).

Cart Velocity: How fast the cart is moving left (negative) or right (positive).

Pole Angle: The angle of the pole from the vertical upright position (0.0 is perfectly straight up, measured in radians).

Pole Angular Velocity: How fast the pole is rotating/falling.

2. Why does the pole fall so quickly?
The pole falls almost instantly because the agent is currently using a random policy.

3. The reward is `+1` per step the pole stays up. What's the maximum total reward possible? (Look up `CartPole-v1` in the Gymnasium docs.)
For CartPole-v1, the maximum total reward possible is 500.


PART 2

Rule : if pole_angle > 0:
        return 1  # push right
    else:
        return 0  # push left

Average score :
Episode 1: 41.0
Episode 2: 39.0
Episode 3: 39.0
Episode 4: 40.0
Episode 5: 35.0

Average over 5 episodes: 38.8


Did the rule work every time, or did it sometimes fail? Why?
No, the rule does not work every time; it will eventually fail. While looking only at the pole angle (pushing right when it leans right, left when it leans left) is enough to keep it upright for a short while, it fails because it ignores the other variables in the observation array.

7. Can you think of a smarter rule? (You don't have to implement it — just describe it.)
A smarter rule would be to use a weighted combination of all the variables in the observation vector rather than just reacting to the angle.
Instead of a basic if statement, a smarter rule would calculate a "score" to decide whether to push left or right:

Incorporate Velocity: Look at pole_ang_vel. If the pole is leaning right (pole_angle > 0) but already moving quickly back toward the left (pole_ang_vel < 0), we should stop pushing right and instead push left early to prevent it from swinging too far the other way.

Incorporate Cart Position: Look at cart_pos. If the cart is getting dangerously close to the right edge, the rule should apply a "pull-back" force (pushing left) to center the cart, even if the pole is slightly leaning right.


PART 3

What is the agent trying to do?
Answer: The agent is piloting a spaceship and trying to safely land it on a designated landing pad between two yellow flags. It needs to control its descent speed and orientation so it doesn't crash, tip over, or fly off the screen, all while consuming as little fuel as possible.

What does a state look like? What are the possible actions?
Answer:  State (Observation space): An 8-dimensional vector containing the lander's coordinates, its velocities, its angle and angular velocity, and two boolean flags indicating whether the left and right legs are touching the ground.Actions: A discrete action space with 4 choices: 0 (do nothing), 1 (fire left orientation engine), 2 (fire main engine), and 3 (fire right orientation engine).

Does random play ever solve it? Why or why not?

Answer: No, random play will almost never solve it. LunarLander requires precise, coordinated sequences of actions to counteract gravity, stabilize tilt, and slow down right before touching the ground. A random agent fires its engines chaotically, which usually causes the lander to spin out of control, fly completely away from the landing pad, or rocket downward and smash violently into the surface.



PART 4

In your own words, what is the agent's goal in any RL problem?
The agent's primary goal is to learn a strategy, called a policy, that maximizes the cumulative reward it receives over time from the environment. It isn't just looking for immediate gratification or a high reward on the very next step; instead, it aims to make a sequence of decisions that leads to the best possible long-term outcome, such as keeping a pole balanced or safely landing a spaceship.

Why is "random" not a good strategy — and what would the agent need to do instead?
A random strategy is ineffective because it lacks intent, consistency, and a feedback loop, meaning it cannot deliberately replicate success or avoid catastrophic failure. To be successful, the agent needs to analyze its current state, predict the future consequences of its potential actions, and systematically choose actions that have a higher probability of yielding positive rewards based on past experiences.


What does it mean for an agent to "learn"? (Hint: what should change between episode 1 and episode 100?)
For an agent to "learn" means it is progressively updating its internal decision-making rules based on the trial-and-error feedback (rewards and penalties) it receives from the environment. In episode 1, the agent behaves chaotically and makes frequent mistakes because it knows nothing about the world; by episode 100, its behavior should look coordinated and deliberate, showing fewer critical errors and achieving significantly higher total scores as it repeats successful strategies.

Write down 3 questions you have about RL that you'd like to ask your mentor.
Question 1: In complex environments with massive state spaces (like thousands of pixels on a screen), how does an RL agent efficiently figure out which specific variables matter without getting overwhelmed by noise?

Question 2: How do engineers design a good reward function for real-world tasks where success isn't simple? If the reward function is slightly flawed, how do you prevent the agent from "cheating" or exploiting loopholes to get rewards without actually solving the task?

Question 3: Since we saw that hand-coded rules (like a PID controller) can solve CartPole incredibly well, how do you decide when a problem is complex enough to justify using Reinforcement Learning over traditional engineering algorithms?