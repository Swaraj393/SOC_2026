1. Describe the trend in episode returns. Does the moving average increase? Does it plateau or
oscillate?

The episode returns show a noisy but slightly increasing trend over time. The moving average initially improves, indicating some learning, but it does not steadily increase and instead oscillates. Towards later episodes, it fluctuates between moderate values rather than converging, suggesting unstable learning.

2. Compare the trained policy to a random policy when you run a few episodes with rendering.
What qualitative differences do you see?

A random policy behaves erratically and fails quickly, as actions are chosen without any learning. In contrast, the trained policy is more stable and is able to balance the pole for longer durations. Although not optimal, it shows more controlled and purposeful behavior compared to random actions.

3. Try at least one hyperparameter change (learning rate, number of episodes, hidden size, or
discount factor). What changed in the learning behavior?

When experimenting with hyperparameters (e.g., learning rate or number of episodes), the learning behavior changed noticeably. A higher learning rate caused more instability and fluctuations, while increasing the number of episodes slightly improved performance but did not fully stabilize learning. This shows that REINFORCE is sensitive to hyperparameter choices.

4. In your own words, what does the REINFORCE loss −log π(at|st)Gt try to do?

The REINFORCE loss −log π(at|st)Gt try encourages the model to increase the probability of actions that lead to higher returns. The negative sign is used because we perform gradient descent, so minimizing this loss effectively maximizes expected reward. In simple terms, the model learns to repeat actions that worked well in the past.

Additional observations

The learning process is quite unstable due to high variance in returns, which is a known limitation of the REINFORCE algorithm. Even though occasional high rewards are achieved, the agent does not consistently maintain good performance. This highlights the need for techniques like baselines or variance reduction for more stable training.