Introduction

This project asks a simple but important question: when an RL trading agent is evaluated honestly on real market data, can modest environment improvements make it learn something useful, and do those gains survive out of sample? The goal was not to search for a magical profitable strategy, but to build a disciplined backtest: choose two real markets, train on past data only, evaluate on future data only, compare against baselines, and report the results without hiding the negative ones. That is the standard the course built toward over Weeks 5 and 6.

The project used two markets, SPY and NIFTY 50, and compared a baseline PPO agent against an improved PPO agent. The baseline environment replayed historical returns with a 5-return observation window and three actions. The improved environment added richer state features and finer position sizing. The main question was whether these controlled changes helped the agent generalize better on unseen market history.

Data

Two markets were used: SPY and NIFTY 50 (^NSEI). Both had roughly eight years of daily data and were downloaded with adjusted prices so that splits and dividends would not create artificial jumps in returns. The log-return series were then computed and summarized before any RL training.

Market	Days	Mean daily return	Std. dev.	Lag-1 autocorrelation
SPY	2514	+0.000488	0.011137	-0.1163
NIFTY 50	2457	+0.000421	0.010545	-0.0284

Both markets had small positive drift and volatility around 1% per day, which is close to the synthetic volatility level used earlier in the course. The lag-1 autocorrelations were weak and negative in both cases, especially for NIFTY, which suggests there is very little simple one-step momentum signal available to exploit. That is an important result by itself: the markets do not obviously resemble the strong synthetic momentum market from Week 5.

A small positive mean return is not exactly zero because equity indices tend to drift upward over long horizons. That drift is part of why buy-and-hold is a meaningful baseline on real data, unlike the random-walk settings from Weeks 4 and 5.

Environment design

The baseline environment was the historical trading environment from Week 6. Each episode replays a random contiguous 100-day window from the training slice of historical returns. The observation contains the last five returns plus the current position, the actions are discrete, and the reward is step P&L minus transaction costs.

The improved environment kept the same basic structure but added two controlled upgrades.

First, it expanded the observation with richer state features: rolling volatility and rolling momentum computed only from past returns in the current episode. This was intended to give the agent a slightly more informative view of local market regime without introducing lookahead bias.

Second, it changed the action space from three positions to five: full long, half long, flat, half short, and full short. This allowed the agent to scale risk more gradually rather than using only all-or-nothing positions. The idea was that if the edge in real data is weak, intermediate positions might help the agent trade more cautiously around noisy signals.

The expectation before training was that the improved environment might help on both markets, but that any advantage would probably be modest. The data showed very weak autocorrelation, so a large improvement would have been suspicious. The aim was controlled improvement, not a rewrite of the problem into something easier.

Experimental setup

For each market, the data was split chronologically into 80% training and 20% testing. No shuffling was used, because shuffling would mix future information into training and create lookahead bias. All design decisions were made using the training slice only, and the test slice was used only for final evaluation.

Three random seeds were used for each PPO configuration. The project compared four strategies on the test set:

Improved PPO
Baseline PPO
Random agent
Buy-and-hold

Training used 100,000 PPO timesteps per run. Evaluation reported final cumulative P&L, annualized Sharpe ratio, and maximum drawdown. The report below uses the mean across the three seeds for the PPO-based strategies, together with the spread across runs.

Results
SPY




Strategy	Final mean cumulative P&L	Sharpe ratio	Max drawdown
Improved PPO	+8.83 ± 2.50	+2.24 ± 0.00	+5.44 ± 1.54
Baseline PPO	+7.16 ± 5.06	+1.49 ± 1.05	+4.41 ± 3.12
Random	-8.79 ± 7.35	-1.75 ± 1.49	+13.59 ± 5.12
Buy & Hold	+10.73 ± 3.69	+2.23 ± 0.89	+6.61 ± 2.25

For SPY, the improved PPO beat the baseline PPO on average and reduced drawdown relative to the random strategy. However, buy-and-hold remained the strongest benchmark on final P&L. That matters: the improved agent was better than the baseline learning setup, but not clearly better than the passive market exposure of the index itself.

The train-vs-test comparison for the improved SPY agent was also informative:




Split	Final mean cumulative P&L	Sharpe ratio	Max drawdown
Train	+1.66 ± 0.47	+0.61 ± 0.00	+10.82 ± 3.06
Test	+8.83 ± 2.50	+2.24 ± 0.00	+5.44 ± 1.54

The test result was higher than the training result, so there was no classic overfitting gap here. That does not mean the agent discovered a strong general edge; it more likely means the training windows were harder or the result is partly due to random variation across seeds.

NIFTY 50




Strategy	Final mean cumulative P&L	Sharpe ratio	Max drawdown
Improved PPO	+4.47 ± 0.00	+1.95 ± 0.00	+3.04 ± 0.00
Baseline PPO	+2.94 ± 4.16	+0.64 ± 0.90	+2.06 ± 2.92
Random	-7.73 ± 7.15	-1.64 ± 1.68	+12.09 ± 4.98
Buy & Hold	+8.82 ± 4.34	+1.91 ± 1.10	+6.19 ± 2.14

On NIFTY, the improved PPO again beat the baseline PPO and the random agent, but buy-and-hold stayed ahead on raw P&L. This is a realistic outcome for daily equity data: the index drift is strong enough that passive exposure often remains difficult to beat once transaction costs are included.

The train-vs-test comparison was:




Split	Final mean cumulative P&L	Sharpe ratio	Max drawdown
Train	+0.92 ± 0.00	+0.64 ± 0.00	+6.44 ± 0.00
Test	+4.47 ± 0.00	+1.95 ± 0.00	+3.04 ± 0.00

Again, the test result exceeded the train result. That means the project did not produce a conventional overfitting signature. Instead, the model appears to have learned a modest, somewhat robust behavior that transferred to held-out data, but the edge is still not strong enough to beat buy-and-hold.

Discussion

The most defensible conclusion is that the environment improvements helped, but only modestly. The improved PPO consistently outperformed the baseline PPO on both markets, which suggests that richer state features and finer position sizes gave the agent something useful to work with. That is a meaningful result because it shows the environment design mattered more than the algorithm itself.

At the same time, neither improved model beat buy-and-hold on final P&L. That is not a failure of the project; it is a realistic finding on real daily data. The markets had weak short-horizon autocorrelation, so the agent was trying to extract signal from a very noisy setting. In that context, outperforming a passive long-only index strategy is a very high bar.

The evaluation also showed that a single training run would have been misleading. Across the three seeds, the PPO results moved enough to matter, and the baseline PPO was noticeably noisier than the improved PPO. That is exactly why the project required multiple seeds and a chronological split. The clean methodology was more informative than any single number.

The strongest conclusion is that modest environment design improvements can make an RL trader more stable and slightly more profitable on real data, but they did not produce a reliable edge strong enough to beat buy-and-hold on either market.

Limitations and next steps

This backtest still ignores several real-world issues. It does not model slippage beyond a simple transaction cost, it does not include order execution delays, it does not account for regime changes explicitly, and it only uses a small local state window. It also evaluates one asset at a time, so it does not exploit cross-asset relationships, macro signals, or news-based features.

If there were two more weeks, the first thing to try would be a richer feature set with volume, volatility regime, and longer-horizon trend features, while keeping the same strict chronological evaluation discipline. A second natural extension would be a multi-asset environment, because genuine trading edge on real data often comes from relationships between instruments rather than from a single return series alone.

Conclusion

This project built an honest RL backtest on two real markets using a chronological train/test split, multiple seeds, and multiple baselines. The improved environment consistently helped relative to the baseline PPO, but the final results still fell short of buy-and-hold. That is a useful and defensible outcome: it shows the pipeline can detect modest improvements, while also showing how hard it is to beat a strong passive benchmark on real daily data.