# MDP Solver: Value Iteration and Policy Iteration

## Overview

This project implements two classic algorithms for solving Markov Decision Processes (MDPs): **Value Iteration** and **Policy Iteration**. These algorithms are used to find the optimal policy (best action to take from each state) in a grid-based environment where the agent can move in four directions (Up, Down, Left, Right) with some randomness in movement.

## Algorithms

### 1. Value Iteration

#### Purpose

Finds the optimal value function (how good it is to be in each state) and the optimal policy (best action from each state). This algorithm is particularly useful when you want to find the best possible strategy for an agent in a stochastic environment.

#### Mathematical Foundation

The algorithm is based on the Bellman Optimality Equation:

```
V*(s) = max_a ∑ P(s'|s,a)[R(s) + γV*(s')]
```

Where:

- V\*(s) is the optimal value of state s
- max_a represents taking the maximum over all possible actions
- P(s'|s,a) is the probability of transitioning to state s' from state s by taking action a
- R(s) is the immediate reward for being in state s
- γ (gamma) is the discount factor (0 ≤ γ < 1)
- V\*(s') is the optimal value of the next state

#### How it works

1. **Initialization**:

   - Start with an initial guess for the value of each state (typically zeros)
   - Create an empty policy table

2. **Iteration**:

   - For each state s:
     - For each possible action a:
       - Calculate the expected value of taking action a in state s
       - Consider all possible next states and their probabilities
       - Include both immediate reward and discounted future value
     - Update V(s) to the maximum value found
     - Update the policy to choose the action that gave the maximum value

3. **Convergence**:
   - Continue until the maximum change in value function is below a threshold
   - The resulting value function and policy are optimal

#### Example

Consider a 3x3 grid where:

- The agent starts at (1,1)
- There's a reward of 100 at (0,0)
- There's a reward of 10 at (0,2)
- All other states have a reward of -1
- γ = 0.99

The algorithm will:

1. Initialize all values to 0
2. Repeatedly update values considering:
   - 80% chance of intended movement
   - 10% chance of each perpendicular movement
   - Wall collisions (staying in place)
3. Eventually converge to optimal values and policy

#### Key function

```python
value_iteration(reward, threshold=1e-4)
```

### 2. Policy Iteration

#### Purpose

Alternates between evaluating a policy (how good it is) and improving it (finding a better policy). This algorithm is often more efficient than value iteration when the policy converges quickly.

#### Mathematical Foundation

The algorithm uses two main equations:

1. **Policy Evaluation**:

```
Vπ(s) = ∑ P(s'|s,π(s))[R(s) + γVπ(s')]
```

Where:

- Vπ(s) is the value of state s under policy π
- π(s) is the action chosen by policy π in state s

2. **Policy Improvement**:

```
π'(s) = argmax_a ∑ P(s'|s,a)[R(s) + γVπ(s')]
```

Where:

- π'(s) is the improved policy
- argmax_a finds the action that maximizes the expected value

#### How it works

1. **Initialization**:

   - Start with a random policy
   - Initialize value function to zeros

2. **Policy Evaluation**:

   - For each state s:
     - Calculate the value of being in state s under current policy
     - Consider the action specified by the policy
     - Include both immediate reward and discounted future value
   - Continue until values converge

3. **Policy Improvement**:
   - For each state s:
     - For each possible action a:
       - Calculate the expected value of taking action a
     - Update policy to choose the best action
   - If policy changed, go back to evaluation
   - If policy unchanged, algorithm has converged

#### Example

Using the same 3x3 grid as before:

1. Start with random policy (e.g., all states choose "Up")
2. Evaluate this policy:
   - Calculate values for each state assuming the agent always moves up
   - Consider the 80-10-10 probability distribution
3. Improve the policy:
   - For each state, find the action that gives highest expected value
   - Update policy accordingly
4. Repeat until policy stabilizes

#### Key function

```python
policy_iteration(grid, rewards, discount=0.99, max_iter=1000, tol=1e-4)
```

#### Comparison with Value Iteration

- **Value Iteration**:

  - Updates both values and policy in each iteration
  - May take more iterations to converge
  - Simpler to implement
  - More computationally intensive per iteration

- **Policy Iteration**:
  - Separates evaluation and improvement steps
  - Often converges in fewer iterations
  - More complex to implement
  - More efficient when policy converges quickly

## Functions

### 1. `get_transition_probabilities(state, action)`

#### Purpose

Calculates the possible next states and their probabilities if the agent takes a given action from a given state.

#### How it works

- 80% chance to go in the intended direction
- 10% chance to go in each of the two perpendicular directions
- If the move would go off the grid, the agent stays in place

#### Example

If the agent tries to move "Up" from (1,1):

- 80% chance: moves to (0,1)
- 10% chance: moves to (1,0) (left)
- 10% chance: moves to (1,2) (right)

If the agent is at (0,1) and tries to move "Up":

- 80% chance: would go to (-1,1) (off the grid), so stays at (0,1)
- 10% chance: moves to (0,0)
- 10% chance: moves to (0,2)

### 2. `value_iteration(reward, threshold=1e-4)`

#### Purpose

Performs value iteration to find the optimal value function and policy.

#### How it works

1. Initialize values and policy
2. Loop until convergence:
   - For each state, calculate the value of each action
   - Update the state's value to the maximum action value
   - Update the policy to the best action
3. Return the final values and policy

#### Example

If the agent is at (1,1) and the reward for (0,0) is 100:

- The value of (1,1) will be updated based on the expected reward of moving to (0,0) and the value of (0,0)

### 3. `policy_iteration(grid, rewards, discount=0.99, max_iter=1000, tol=1e-4)`

#### Purpose

Performs policy iteration to find the optimal policy.

#### How it works

1. Initialize a random policy
2. Loop until the policy is stable:
   - Policy Evaluation: Calculate the value of each state under the current policy
   - Policy Improvement: Update the policy to the best action for each state
3. Return the final values and policy

#### Example

If the current policy says to move "Up" from (1,1), but moving "Right" gives a higher expected value, the policy will be updated to move "Right"

### 4. `display_policy(policy)`

#### Purpose

Displays the policy in a grid format.

#### Example

If the policy for (1,1) is "Up", the output will show "Up" in the corresponding cell.

## Usage

### 1. Define the reward structure

Set the reward for each state (e.g., terminal states, obstacles).

### 2. Run the algorithms

Call `value_iteration(reward)` or `policy_iteration(grid, rewards)`.

### 3. Display the results

Use `display_policy(policy)` to see the optimal policy.

## Example Code

```python
rewards = [100, 3, 0, -3]
for r in rewards:
    reward = np.full((grid_size, grid_size), -1)
    reward[0, 0] = r  # Top left square
    reward[0, 2] = 10  # Top right square

    print(f"\nResults for reward = {r} using Value Iteration")
    values, policy = value_iteration(reward=reward)
    print("State Values:")
    print(values)
    print("Policy:")
    display_policy(policy)
```

## Dependencies

- **NumPy**: For numerical operations and array handling

## Requirements

- Python 3.x
- NumPy

## Installation

```bash
pip install numpy
```

## License

This project is open source and available under the MIT License.
# Markov-Decission-Process
