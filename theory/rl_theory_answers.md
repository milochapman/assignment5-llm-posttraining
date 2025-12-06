# Assignment 5 — Theory: Reinforcement Learning

This file contains written answers **with full working** for the
Reinforcement Learning questions in Assignment 5.

---

## Question 1

We consider a simple text generation task:

- Each episode is exactly 3 words long.
- Vocabulary: `{I, like, pizza}`.
- The only sentence that receives a reward is:

> I like pizza

with reward **+10**.  
All other 3‑word sentences receive reward **0**.

The agent:

- Starts from the empty state \(s_0 = [\ ]\).
- Chooses one word at a time.
- Follows a **uniform random policy**: each word is equally likely at each step.
- Uses discount factor \(\gamma = 1\).
- Uses value function

\[
V(s) = \mathbb{E}[R_t \mid s_t = s].
\]

### 1. All possible 1‑word and 2‑word states

At each position we can choose any of the 3 words, so:

- **1‑word states (length 1):**

  - `[I]`
  - `[like]`
  - `[pizza]`

- **2‑word states (length 2):**

  All ordered pairs of the 3 words (3 × 3 = 9 total):

  - `[I, I]`
  - `[I, like]`
  - `[I, pizza]`
  - `[like, I]`
  - `[like, like]`
  - `[like, pizza]`
  - `[pizza, I]`
  - `[pizza, like]`
  - `[pizza, pizza]`

### 2. Number of terminal states and non‑zero‑reward states

A terminal state is any full 3‑word sentence.

- Number of possible 3‑word sentences:
  - 3 choices for each position → \(3^3 = 27\) terminal states.

- Only one of these gives **non‑zero reward**:

  - `I like pizza` → reward **+10**.

So:

- **Number of terminal states:** 27  
- **Number with non‑zero reward:** 1

### 3. Value function \(V(s)\) for specific states

We use the fact that reward is only given at the end.  
So for any state \(s\):

\[
V(s) = P(\text{eventually generate "I like pizza"} \mid s) \times 10.
\]

We compute this probability under the **uniform random policy**.

#### (a) \(s_0 = [\ ]\) (empty state)

From the empty state we generate 3 words, each chosen uniformly from 3 options.

- Total possible 3‑word sentences: 27.
- Exactly one is `I like pizza`.

So:

\[
P(\text{I like pizza} \mid s_0) = \frac{1}{27}.
\]

Therefore:

\[
V(s_0) = 10 \times \frac{1}{27} = \frac{10}{27} \approx 0.3704.
\]

#### (b) \(s_1 = [I]\)

Now we already have the first word fixed as `I`. Only 2 steps remain.

To get `I like pizza`, the next words must be:

- Second word: `like` (probability \(1/3\))
- Third word: `pizza` (probability \(1/3\))

Under the random policy, these are independent choices:

\[
P(\text{I like pizza} \mid [I]) = \frac{1}{3} \times \frac{1}{3} = \frac{1}{9}.
\]

So:

\[
V(s_1) = 10 \times \frac{1}{9} = \frac{10}{9} \approx 1.1111.
\]

#### (c) \(s_2 = [I, like]\)

Now we already have `I like`. Only one step remains.

To get `I like pizza`, the last word must be `pizza`:

- Probability of choosing `pizza` at the last step: \(1/3\).

Thus:

\[
P(\text{I like pizza} \mid [I, like]) = \frac{1}{3}.
\]

So:

\[
V(s_2) = 10 \times \frac{1}{3} = \frac{10}{3} \approx 3.3333.
\]

#### (d) \(s_3 = [I, pizza]\)

The current partial sentence is `I pizza`. With only one word left, there is **no way** to turn this into `I like pizza`, since the second word is already incorrect.

Therefore:

\[
P(\text{I like pizza} \mid [I, pizza]) = 0,
\]

and

\[
V(s_3) = 10 \times 0 = 0.
\]

---

## Question 2

We revisit the same environment:

- Vocabulary: `{I, like, pizza}`
- Sentence length: 3 words
- Reward: `I like pizza` → +10; all others → 0
- Discount factor: \(\gamma = 1\)

We are now using **Q‑learning**.

The agent is in state:

\[
s = [I]
\]

and must choose the next word.  
Current Q‑values for this state:

| Action (next word) | Q([I], action) |
|--------------------|----------------|
| I                  | 1.0            |
| like               | 1.0            |
| pizza              | 0.5            |

Additional information:

- Learning rate: \(\alpha = 0.5\)
- Discount factor: \(\gamma = 1\)
- Next state after taking action `like`:

  \[
  s' = [I, like]
  \]

- Q‑values at \(s' = [I, like]\):

  - \(Q([I, like], I) = 1.0\)
  - \(Q([I, like], like) = 0.5\)
  - \(Q([I, like], pizza) = 2.0\)

We assume that the reward at this step (after moving from `[I]` to `[I, like]`) is **0**, since reward is only given when the full 3‑word sentence is completed.

### 1. Q‑learning update rule

The standard one‑step Q‑learning update for a transition
\((s, a, r, s')\) is:

\[
Q(s, a) \leftarrow Q(s, a)
  + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right].
\]

### 2. Updated value of \(Q([I], \text{like})\)

We plug in the values for:

- \(s = [I]\)
- \(a = \text{like}\)
- \(Q(s, a) = Q([I], \text{like}) = 1.0\)
- Immediate reward: \(r = 0\)
- \(\gamma = 1\)
- Next state: \(s' = [I, like]\)
- \(\max_{a'} Q(s', a') = \max\{1.0, 0.5, 2.0\} = 2.0\)
- Learning rate: \(\alpha = 0.5\)

Compute the temporal‑difference (TD) error:

\[
\delta
= r + \gamma \max_{a'} Q(s', a') - Q(s, a)
= 0 + 1 \times 2.0 - 1.0
= 1.0.
\]

Now update:

\[
Q_{\text{new}}(s, a)
= Q(s, a) + \alpha \delta
= 1.0 + 0.5 \times 1.0
= 1.5.
\]

So the **updated Q‑value** is:

\[
\boxed{Q([I], \text{like}) = 1.5.}
\]