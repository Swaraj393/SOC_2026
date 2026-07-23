The signal check confirms that the environment is behaving as intended: the measured lag-1 autocorrelation is very close to the target rho for all tested values. This means the AR(1) process is correctly introducing momentum when rho > 0 and mean reversion when rho < 0.
The random agent’s average reward remains negative across all cases because it does not use the signal and keeps incurring transaction costs. This shows that the environment now contains structure, but only a learning or rule-based policy can exploit it.


Task 5 questions:

1) Measured autocorrelations and why they are not exact

My three measured lag-1 autocorrelations were approximately -0.000 for ρ=0.0, +0.501 for ρ=0.5, and -0.501 for ρ=−0.5. They do not match ρ exactly because the sample is finite, and autocorrelation estimated from data always has a small amount of sampling noise. The values are still very close, which confirms that the AR(1) process was implemented correctly.

2) Why the random agent has negative mean reward

The random agent changes position very often, and every position change pays a transaction cost. Even when the market has momentum or mean reversion, a random agent ignores the state, so it does not use the information contained in the returns. That means it keeps paying costs without systematically exploiting the signal, which creates a slight negative drift in reward.

3) Week 5 cumulative P&L vs Week 4

Compared with Week 4, the key change is that the price returns are no longer an independent random walk; they now follow an AR(1) process with ρ=0.5, which creates positive autocorrelation. In Week 4, no strategy had a real edge, so the curves stayed near zero, while in Week 5 the market has memory and a momentum-based policy can, in principle, exploit it. The single property that explains the difference is the introduction of autocorrelated returns.

4) Policy observed in Task 3

In my run, the learned PPO policy mostly stayed at position 0 and did not visibly switch long or short, so it was not close to the hand-coded momentum rule. The hand-coded rule would match the sign of the last return, but the observed policy was much more conservative. This likely means the agent learned that switching was not worthwhile under the current reward/cost setup.

5) Why buy-and-hold cannot exploit autocorrelation

Buy-and-hold keeps the same position regardless of what the market is doing, so it does not react to the sign of the last return. Autocorrelation creates a predictable pattern only if the agent changes its position based on recent returns. A fixed long position may gain when the trend is up, but it cannot adapt when the sign flips, so it does not fully exploit the structure.

6) How final P&L changed with rho

Final P&L should improve as ρ increases because larger positive autocorrelation means a stronger momentum signal. In my sweep, the numbers were noisy and did not show the ideal monotonic pattern, but the intended relationship is still that stronger signal strength should make trading more profitable. So the broad expectation is: larger ρ → more edge → higher P&L.

7) Where transaction cost eats the edge

At ρ=0.25, the edge becomes much less useful once the transaction cost is high enough that switching positions costs more than the expected gain from the signal. In simple terms, trading only makes sense when the signal is stronger than the fee paid to act on it. If the cost is too large, the best decision is often to trade less or not trade at all.

For mean reversion with ρ=−0.5, the optimal rule is to bet against the last move: go short after a positive return and go long after a negative return. In other words, the sign flips compared with the momentum case. That is because the next return is expected to have the opposite sign from the last one.

9) What to expect next week with real stock returns

I would expect the lag-1 autocorrelation of real daily stock returns to be much closer to 0.0 than to 0.5. That means the next week will probably be harder than this one, because the signal will be weaker and may be close to noise. An RL agent will need much better data handling and evaluation discipline to show a real edge.

10) Three questions to discuss with my mentor
How do we decide whether a weak market signal is strong enough to be worth trading after transaction costs?
What is the best way to avoid overfitting when training RL agents on financial time series?
If the market signal is very small, should we prefer a simpler rule-based strategy over an RL policy?