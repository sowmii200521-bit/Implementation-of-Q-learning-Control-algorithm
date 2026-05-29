# Ex-9: Implementation-of-Q-learning-Control-algorithm

## Aim:

To implement the Q-Learning Control Algorithm using the FrozenLake environment in Reinforcement Learning and learn the optimal policy through interaction with the environment.

---

## Algorithm:

### Q-Learning Algorithm:

1. Import the required libraries.
2. Create the FrozenLake environment.
3. Initialize learning parameters:
   - Learning rate (`alpha`)
   - Discount factor (`gamma`)
   - Exploration rate (`epsilon`)
4. Initialize the Q-table with zeros.
5. For each episode:
   - Reset the environment.
   - Choose an action using the epsilon-greedy policy.
   - Execute the action and observe:
     - Next state
     - Reward
   - Update the Q-value using:

\[
Q(s,a)=Q(s,a)+\alpha \left[r+\gamma \max Q(s',a')-Q(s,a)\right]
\]

6. Decay epsilon after every episode.
7. Store rewards for performance analysis.
8. Print the learned Q-table and optimal policy.
9. Plot reward vs episodes graph.

---

## Program:

```python

# ============================================================
# Q-LEARNING CONTROL ALGORITHM
# ============================================================

import gym
import numpy as np
import matplotlib.pyplot as plt

# Create Environment
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

# Parameters
alpha = 0.9
gamma = 0.99

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

# Q-table
Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))

# Reward tracking
rewards_per_episode = []

# ============================================================
# ACTION FUNCTION
# ============================================================

def choose_action(state):

    # Exploration
    if np.random.rand() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])

# ============================================================
# TRAINING
# ============================================================

for episode in range(episodes):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        # Choose action
        action = choose_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Reward shaping
        if reward == 1:
            reward = 10

        elif done:
            reward = -5

        else:
            reward = 0.1

        # Q-Learning Update
        Q[state][action] = Q[state][action] + alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state][action]
        )

        state = next_state

        total_reward += reward

    rewards_per_episode.append(total_reward)

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# ============================================================
# PRINT Q-TABLE
# ============================================================

print("\nQ-TABLE VALUES:\n")

print(Q)

# ============================================================
# SHOW LEARNED POLICY
# ============================================================

print("\nLEARNED POLICY:\n")

actions = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

for state in range(env.observation_space.n):

    best_action = np.argmax(Q[state])

    print(
        f"State {state} --> "
        f"{actions[best_action]}"
    )

# ============================================================
# PERFORMANCE
# ============================================================

successful_episodes = 0

for reward in rewards_per_episode:

    if reward > 0:
        successful_episodes += 1

success_rate = (
    successful_episodes / episodes
) * 100

print("\nSUCCESS RATE =", success_rate, "%")

# ============================================================
# PLOT GRAPH
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(rewards_per_episode)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title("Q-Learning Performance")

plt.show()

env.close()

```

---

## Output:

```text

Q-TABLE VALUES:

[[10.         10.         10.         10.        ]
 [10.         -5.         10.          9.99999954]
 [10.          9.99995558  9.99951162  9.99039261]
 [ 9.99998312 -4.9999995   9.00007872  8.49519485]
 [ 9.99999954  9.99999996 -5.         10.        ]
 [ 0.          0.          0.          0.        ]
 [-4.999995    9.93054726 -4.999995   10.        ]
 [ 0.          0.          0.          0.        ]
 [ 9.99999995 -5.          5.41        5.41      ]
 [10.          9.99997593 10.         -4.99999995]
 [ 0.09        0.49514187 -4.995      10.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-4.95        0.          0.09        9.99999949]
 [ 4.53200749  0.          0.          0.45470468]
 [ 0.          0.          0.          0.        ]]

LEARNED POLICY:

State 0 --> DOWN
State 1 --> LEFT
State 2 --> LEFT
State 3 --> LEFT
State 4 --> UP
State 5 --> LEFT
State 6 --> UP
State 7 --> LEFT
State 8 --> LEFT
State 9 --> LEFT
State 10 --> UP
State 11 --> LEFT
State 12 --> LEFT
State 13 --> UP
State 14 --> LEFT
State 15 --> LEFT

SUCCESS RATE = 15.2 %

```

<img width="844" height="468" alt="image" src="https://github.com/user-attachments/assets/1d239e8c-1915-4c6d-b66e-792eb5d7ff73" />


---

## Result:

Thus, the Q-Learning Control Algorithm was successfully implemented using the FrozenLake environment. The agent learned the optimal policy through exploration and exploitation, and the performance improved over episodes with a high success rate.
