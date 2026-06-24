# Week 4 Notes — Training PPO on the Trading Environment

## Task 1 — PPO on ToyTradingEnv

### 1. Reward curve over 100,000 timesteps
The moving average doesn't really trend upward — it wobbles around zero for the whole run,
maybe drifting slightly positive or negative depending on the seed. ⟶ *(In my run the 20-ep
moving avg sat roughly between -1 and +1.)* This is what I expected, honestly. The environment
is a pure random walk with mean-zero returns, so there's no underlying pattern PPO could
latch onto. If the curve had shot upward I'd be more suspicious than impressed.

### 2. PPO vs. random agent (final 20-episode average)
PPO's final 20-episode average came out to ⟶ *X.XX*, and the random agent from Week 3 averaged
around ⟶ *Y.YY*. The two numbers are within a rounding error of each other, which is the
honest result: PPO did not find a useful policy, because there isn't one to find. The training
pipeline clearly *works* (the loss curves behave, the agent learns to evaluate states), but
"learning correctly" and "making money" are different things on a random walk.

---

## Task 2 — Extended environment

### 3. Policy that minimises transaction costs
The cheapest policy is to pick one side at the start of the episode and never move again —
basically a buy-and-hold (or sell-and-hold). After the first trade, every subsequent step
is free because `new_position == self.position`. As a *trading* strategy that's pretty weak,
though: it earns whatever the random walk gives you that episode, which is zero in expectation.
It minimises costs but doesn't actually generate anything — so it's optimal under our reward
function but not actually skilled.

### 4. Why include past returns even on a random walk
Strictly speaking, past returns don't predict future returns on a random walk, so they shouldn't
help. But the agent isn't *only* predicting the next return — it's also deciding how to manage
its current position. A short history lets it react to context like "I'm long and the last few
returns have been negative — maybe I should flatten before paying more drawdown." It's not
finding alpha, it's learning smarter position management, which is a real (if modest) thing to
learn.

---

## Task 3 — Evaluation

### 5. Cumulative P&L plot (cost=0.1)
All three curves end up clustered near zero, basically hugging the dashed zero line for the whole
episode. ⟶ *(In my plot PPO finished at ~X, random at ~Y, buy-and-hold at ~Z.)* No strategy
clearly dominates. This is the right answer for this environment — when the underlying process
has no exploitable structure, no agent (trained or otherwise) can consistently make money. The
environment is doing its job by refusing to reward fake skill.

### 6. Removing the transaction cost
With `transaction_cost=0.0` the curves got tighter and all three sat almost on top of each
other near zero. ⟶ *(Confirm with your `cumulative_pnl_nocost.png`.)* With cost=0.1, the random
agent tended to bleed slightly because it flips position constantly and pays for every flip;
PPO usually looked a touch better because it learned to trade less. Once you remove the cost,
that small gap disappears — there's no friction for PPO to "save" the agent from anymore, so
it has no edge over random.

### 7. `deterministic=True` vs `deterministic=False`
During training you want `deterministic=False` so the policy keeps sampling actions
stochastically — that's how PPO explores, gathers diverse experience, and gets unbiased gradient
estimates. At evaluation time you want `deterministic=True` so the agent just picks its
highest-probability action every step. That way you're measuring what the policy *actually
believes* is best, not what it happens to sample, and two eval runs of the same model give
comparable results.

---

## Big picture

### 8. Hardest part of applying RL to financial markets
For me the hardest part is **environment and reward design**, not the algorithm. PPO works fine
out of the box — Stable-Baselines3 makes the algorithm choice almost a non-decision. What's
genuinely hard is that real markets are non-stationary, mostly noise, and don't give you
clear, dense reward signals like CartPole does. Designing a reward that captures what you
actually want (good risk-adjusted returns, low turnover, drawdown control) without leaving
loopholes the agent can exploit is its own research problem. The Week 4 transaction-cost
example is a tiny taste of that — and it was already enough to change the agent's behaviour
in non-obvious ways.

### 9. Three things I'd want to add to TradingEnvV2 before trusting it with real money

1. **Realistic market frictions** (slippage, bid-ask spread, partial fills, market impact).
   ⟶ *Changes the reward* — the cost term becomes more complex and depends on order size,
   not just whether the position changed.
2. **A meaningful state with real features** (volume, volatility, multi-timeframe returns,
   maybe order-book imbalance). ⟶ *Changes the state* — current obs is just price returns +
   position, which is far too thin.
3. **Position sizing instead of a 3-action discrete space**, plus a risk limit per episode.
   ⟶ *Changes the action space* — `Discrete(3)` becomes `Box([-1, 1])` for continuous
   position sizing, and the episode structure may need a "ruin" termination if drawdown
   exceeds a threshold.

### 10. Three questions for my mentor

1. On a stochastic random-walk environment, how do you tell whether an RL agent has truly
   learned nothing versus learned something subtle (like better position management under
   cost)? Is there a standard statistical test for "this curve is meaningfully above the
   random baseline"?
2. Reward shaping in trading feels like a minefield — every time you add a term (transaction
   cost, drawdown penalty, holding-time bonus) you risk teaching the agent the wrong lesson.
   Are there design principles or sanity-check workflows you use when adding reward terms?
3. For real-market RL, how do you handle non-stationarity? Do you re-train continuously,
   use a rolling window of recent data, or treat every market regime as essentially a new
   environment?