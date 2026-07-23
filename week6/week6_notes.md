Number of days: 2514
Mean daily return: +0.00049
Std of daily returns: 0.01114
Lag-1 autocorrelation: -0.1163

TASK 2:

The three random-agent episodes produced finite rewards of 1.97, 31.16, and −0.92. Although the rewards vary because each episode uses a different historical window and random actions, they remain within a reasonable range comparable to the synthetic environments from previous weeks. This confirms that the environment is functioning correctly.

At the beginning of each episode, the history buffer is initialized with zeros because there are no previous returns available from the current episode yet. As a result, during the first few steps the agent cannot observe the true recent market history and instead receives partially empty observations. Once enough steps have passed, the zero values are gradually replaced with real historical returns, allowing the agent to make decisions using actual market information.

TASK 3:
run 1:
PPO (TRAIN): +4.02
PPO (TEST): +10.63
Random:      -6.28
Buy & Hold:  +7.93

run 2:
 PPO on TRAIN data:    +0.00
 PPO on TEST data:     +0.00
 Random (test):        -6.41
 Buy & hold (test):    +9.47

run 3:
PPO on TRAIN data:    +3.60
 PPO on TEST data:     +8.57
 Random (test):        -9.50
 Buy & hold (test):    +9.11

The test performance varied considerably across the three runs, ranging from 0.00 to 10.63. This variation occurs because PPO training is stochastic, with different random initializations and exploration leading to different learned policies. If only Run 2 had been observed, one might conclude that the agent learned nothing useful, while Runs 1 and 3 suggest the agent can outperform the random baseline. Therefore, conclusions should be based on multiple runs rather than a single experiment.

## 1. What was the lag-1 autocorrelation of your real returns? How does it compare to the Week 5 markets, and what does it imply about how much edge is available to your agent?

The measured lag-1 autocorrelation of the SPY daily log returns was **-0.1163**. This is much closer to zero than the synthetic momentum market used in Week 5 (ρ = 0.5), indicating that real daily returns contain far weaker linear structure. This suggests that there is very little predictable edge available from only the previous few daily returns, making the learning problem much harder.

---

## 2. How do the mean and std of real daily returns compare to the synthetic environment's daily_vol = 0.01? Is the mean exactly zero? What does a small positive mean represent?

The real data had a mean daily return of **+0.00049** and a standard deviation of **0.01114**. The standard deviation is very close to the synthetic environment's daily volatility of **0.01**, so both are in the same ballpark. The mean is not exactly zero because stock indices generally exhibit a small long-term upward drift, representing the average long-term growth of the market.

---

## 3. Why must the train/test split be chronological rather than a random shuffle? Name the error that shuffling would introduce, and give one concrete example of how it could fake a profitable backtest.

The train/test split must be chronological because an agent should only learn from past data and be evaluated on future data. Randomly shuffling the returns would introduce **lookahead bias**, allowing the training data to contain information from the future. For example, if data from the 2020 market crash appeared in both training and testing after shuffling, the agent could indirectly learn patterns from the future, producing unrealistically good backtest results.

---

## 4. Real returns are one fixed sequence—unlike Week 5, you cannot generate more. How does the random-window trick partially compensate, and what can it not fix?

Randomly selecting different windows exposes the agent to many different portions of the historical data and provides more varied training episodes. It also allows evaluation over multiple market periods instead of relying on a single window. However, all windows come from the same historical dataset and often overlap, so no new information is created and the amount of available data remains fundamentally limited.

---

## 5. Report your four P&L numbers. Did the agent beat the random baseline on the test set? Did it beat buy-and-hold? Given Weeks 4–5, is your result surprising?

For the first run, the PPO achieved **+4.02** on the training data and **+10.63** on the test data. The random agent achieved **-6.28**, while the buy-and-hold strategy achieved **+7.93** on the test data. The PPO outperformed both the random baseline and buy-and-hold in this run. This result is somewhat surprising because Week 5 suggested that real markets contain much weaker signals than the synthetic momentum environment.

---

## 6. How large was the gap between train and test performance, and what does it tell you? If the gap is near zero, what does that tell you?

The first run showed a difference of approximately **6.61** between the training and test performance, with the test score being higher than the training score. Normally, overfitting causes training performance to exceed testing performance, so this result is likely due to randomness rather than genuine generalization. It highlights why conclusions should not be based on a single run.

---

## 7. Across your three runs, how much did the test P&L vary? What does this say about trusting any single training run?

The PPO test P&L values were **+10.63**, **+0.00**, and **+8.57** across the three runs. This large variation demonstrates that reinforcement learning training is stochastic and individual runs can produce noticeably different outcomes. Therefore, reliable conclusions should be based on multiple runs or averages over several random seeds rather than a single experiment.

---

## 8. On many multi-year periods, buy-and-hold on an index ends clearly positive. Why is beating buy-and-hold a much higher bar than beating the random agent?

Buy-and-hold benefits from the long-term positive growth of the stock market, so it naturally earns positive returns over long periods. In contrast, the random agent frequently changes positions, pays transaction costs, and has no strategy, making it an easy baseline to outperform. Therefore, consistently beating buy-and-hold requires discovering genuinely useful market structure rather than simply avoiding random decisions.

---

## 9. In Week 5 the agent won because ρ ≠ 0. Real daily returns have ρ near zero—yet real quantitative funds exist. Name 2–3 places genuine structure might hide that your current state representation cannot see.

Real market structure may exist in much richer information than just the previous five daily returns. Examples include longer historical trends, trading volume and volatility patterns, company fundamentals, macroeconomic indicators, earnings announcements, news sentiment, and relationships between multiple assets. Our current state representation is too limited to capture these more complex signals.

---

## 10. Write down 3 questions about backtesting, overfitting, or RL on real data that you want to discuss with your mentor.

1. How do professional quantitative funds reduce overfitting when developing trading strategies?
2. What state features would most improve this RL trading environment beyond the last five returns?
3. How many independent training runs or random seeds are generally considered sufficient before reporting RL trading results?