1. Does the moving average trend upward, stay flat, or oscillate around zero?
On a true random walk, expect: oscillating around zero, possibly with a small drift that's just noise.

2. Is this what you expected?
Yes — because tomorrow's return is independent of today's. There is no pattern to learn.

3. How does PPO compare to the random agent?
They should be roughly comparable. If PPO is dramatically higher, be suspicious (overfitting / seed leak). If PPO is dramatically lower, something is wrong with training.

What changed in the observation, and why?
v1 gave the agent only the most recent return + position (2 dims). v2 gives a rolling window of the last 5 returns + position (6 dims). The idea is that the agent can now see short-term momentum and decide based on a small price history rather than a single snapshot. Whether it actually helps on a random walk is another question — past returns carry no information about future returns there — but the agent might still learn smarter position management (e.g. "I'm long and prices kept dropping, time to cut").

Question 1 — Where do the three curves land?
Are they all hugging zero? (Expected on a random walk.)
Is any one of them noticeably off? By how much?

Question 2 — Cost=0.1 vs Cost=0.0
Compare plots/cumulative_pnl.png (cost=0.1) with plots/cumulative_pnl_nocost.png (cost=0.0):

Observation	and Likely reason

With cost=0.1, PPO ends up lower than random	
Random agent flips constantly and pays heavy costs; PPO learned to hold more, which reduces 
bleed but doesn't generate alpha. So PPO often beats Random under cost.

With cost=0.0, all three converge tighter to zero	
No friction means no bleed — every strategy is a fair sample of the random walk and the only difference is variance.

Buy-and-hold sits dead on zero	
It pays zero transaction cost (only one position change at step 0) and the drift is zero. Pure noise around 0.

The expected story: with cost > 0, the gap between PPO and Random tends to widen because the random agent gets punished for churning, while PPO learns to trade less. With cost = 0, the gap shrinks — there's nothing left for PPO to "save" the agent from.

Question 3 — What did the agent actually learn?
On a true random walk with no exploitable structure, the most a smart agent can learn is:

Don't trade unnecessarily (when there's a cost).
Pick a side and stick with it (since flipping costs money but doesn't help).
That's it. There's no profitable strategy to find. If your PPO curve sits slightly above the random baseline under cost=0.1 and roughly matches it under cost=0.0, that's exactly the right result — and a sign your training pipeline is working correctly.