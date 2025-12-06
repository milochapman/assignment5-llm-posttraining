# Assignment 5 – Theory Answers (Final Version)

## Question 1 — Reinforcement Learning and Value Functions

We consider a simple reinforcement learning environment where each episode produces a three‑word sentence. The vocabulary consists of the words “I”, “like”, and “pizza”. The only terminal sentence that receives a reward is **“I like pizza”**, which yields a reward of +10. All other terminal sentences give a reward of 0. The policy always selects words uniformly at random, and the discount factor is γ = 1.

A full three‑word sentence is a terminal state. Since each of the three positions can be filled with any of the three vocabulary words, there are 27 possible terminal states. Only one of these produces a reward.

Because the reward occurs only at the final step, the value of any state is equal to the probability that the agent will eventually generate “I like pizza”, multiplied by the terminal reward of 10.

For the initial state, the probability of selecting the correct three‑word sequence is:

- probability of choosing “I” first: 1/3  
- probability of choosing “like” second: 1/3  
- probability of choosing “pizza” third: 1/3  

Thus,

V(s₀) = 10 × (1/3 × 1/3 × 1/3) = 10/27 ≈ 0.3704.

For the state [I], the first word has already been chosen correctly. Only two steps remain, so:

V([I]) = 10 × (1/3 × 1/3) = 10/9 ≈ 1.1111.

For the state [I, like], the agent must only choose “pizza” next:

V([I, like]) = 10 × (1/3) = 10/3 ≈ 3.3333.

For the state [I, pizza], the second word is incorrect, so the target sentence can no longer be formed, giving:

V([I, pizza]) = 0.

---

## Question 2 — Q‑Learning Update

The agent is in the state [I], and the current Q‑values for the possible next‑word actions are:

Q([I], I) = 1.0  
Q([I], like) = 1.0  
Q([I], pizza) = 0.5  

The agent chooses the action “like” and transitions to the next state [I, like]. The Q‑values for that state are:

Q([I, like], I) = 1.0  
Q([I, like], like) = 0.5  
Q([I, like], pizza) = 2.0  

The learning rate is α = 0.5, the discount factor is γ = 1, and the intermediate reward is r = 0.

Using the Q‑learning update equation:

Q(s,a) ← Q(s,a) + α × (r + γ·maxₐ′ Q(s′, a′) − Q(s,a))

The maximum Q‑value in the next state is maxₐ′ Q([I, like], a′) = 2.0. Substituting into the update:

Q([I], like) ← 1.0 + 0.5 × (0 + 2.0 − 1.0)

The term inside the parentheses is 1.0, so:

Q([I], like) = 1.5.

This means that, after observing the transition to the next state, the estimated value of choosing “like” increases from 1.0 to 1.5, reflecting a better estimate of future rewards based on the best available action in the next state.
