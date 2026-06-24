
## 1. Which hyperparameter change hurt learning the most? What does this tell you about gradient-based optimisation?
The learning rate of `0.1` hurt learning the most. The agent became much less stable, which shows that gradient-based optimisation can fail when the step size is too large. Even if the direction of the gradient is correct, overly aggressive updates can overshoot good policies and make training worse.

## 2. REINFORCE updates after every single episode. Why does using one trajectory’s data make the gradient estimate noisy?
One episode only gives the agent a small and highly variable sample of experience. Because the return from a single trajectory depends a lot on randomness, the gradient estimate can point in a different direction from episode to episode. This makes REINFORCE updates noisy and unstable, especially early in training.

## 3. PPO uses a clipped probability ratio. In your own words, what problem does the clip prevent?
The clip prevents the policy from changing too much in one update. Without clipping, PPO could make a very large update that accidentally destroys a good policy. The clipping keeps training more conservative and helps avoid unstable jumps.

## 4. PPO trains on mini-batches from a large replay buffer. Why might this produce a smoother reward curve than REINFORCE?
Mini-batches let PPO reuse collected data more efficiently and average out some of the randomness in the gradients. This reduces variance compared to updating from a single episode at a time. As a result, PPO usually learns more steadily and produces a smoother reward curve than REINFORCE.

## 5. Describe the two reward curves in 3–4 sentences. Which learned faster? Which was more consistent? Were there any surprises?
PPO learned much faster than REINFORCE and quickly reached very high rewards on CartPole. Its reward curve was also much more consistent, with fewer extreme drops and much smoother improvement over time. REINFORCE improved only slowly and remained noisy, with many ups and downs. The main surprise was how quickly PPO solved the task compared to the instability of REINFORCE.

## 6. List 5 features you would add to the state to make the environment more realistic. For each, explain why it would help the agent.
1. Recent price history: This would help the agent detect short-term trends instead of reacting only to one return.
2. Trading volume: Volume can show how strong a move is and whether a price change is likely to continue.
3. Moving averages: These give the agent a smoother view of the market direction and can help with trend-following.
4. Volatility estimate: Knowing whether the market is calm or unstable would help the agent adjust risk.
5. Cash balance or portfolio value: This would let the agent make decisions based on real trading constraints, not just price movement.

## 7. The current reward is position * price_return * 100. What happens if the agent always stays long? Is that a good strategy on a random walk?
If the agent always stays long, it will earn positive reward only when the price goes up and lose reward when the price goes down. On a random walk, upward and downward moves are roughly balanced, so this strategy is not reliably profitable. It may look good in some short episodes, but over time it should not beat a better policy.

## 8. What are three limitations of this toy environment compared to a real financial market? How might each limitation affect what the agent learns?
First, the price process is just a random walk, so there is no real structure for the agent to discover. Second, there are no transaction costs or slippage, which makes trading look easier than it really is. Third, the environment has only a tiny state space and no real market features like news, liquidity, or multi-asset interactions, so the agent learns a very simplified strategy that would not transfer well to real trading.

## 9. What was the mean reward of the random agent over 10 episodes? Was it positive, negative, or near zero? Does that match your expectation?
The mean reward of the random agent was **1.38** over 10 episodes. That is very close to zero, which is what I expected for a random policy in a random-walk market. Small positive or negative values can happen because of randomness, but there is no real edge.

## 10. Write down 3 questions about environment design or RL-for-trading that you want to discuss with your mentor.
1. How do we choose a reward function that encourages profit without making the agent take unrealistic risks?
2. What state features are most important in a trading environment if we want the agent to learn something useful?
3. How can we make a toy trading environment simple enough for learning, but still realistic enough to transfer to real markets?