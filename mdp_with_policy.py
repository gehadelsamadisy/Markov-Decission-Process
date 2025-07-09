import numpy as np

# Define constants
gamma = 0.99  
actions = ["Up", "Down", "Left", "Right"]  # Actions
grid_size = 3  # 3x3 grid

def get_transition_probabilities(state, action):
    # Returns the transition probabilities for a given state and action.
    # 80% chance to go in the intended direction, 10% for each perpendicular direction.
    # Collision with walls results in no movement.
    transitions = []  # List of (probability, resulting state)
    intended_delta = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
    perpendiculars = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
                      "Left": ["Up", "Down"], "Right": ["Up", "Down"]}

    for move, prob in zip([action] + perpendiculars[action], [0.8, 0.1, 0.1]):
        delta = intended_delta.get(move, (0, 0))
        next_state = (state[0] + delta[0], state[1] + delta[1])
        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
            transitions.append((prob, next_state))
        else:
            transitions.append((prob, state))  # Collision with wall

    return transitions

def value_iteration(reward, threshold=1e-4):
    # Performs value iteration for the given reward structure.
    values = np.zeros((grid_size, grid_size))  # Initialize state values
    policy = np.full((grid_size, grid_size), "", dtype=object)  # Initialize policy

    terminal_states = [(0, 0), (0, 2)]  # Define terminal states

    while True:
        delta = 0  # Track changes for convergence
        new_values = np.copy(values)

        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                if state in terminal_states: 
                    new_values[state] = reward[state]
                    continue

                action_values = []

                for action in actions:
                    action_value = 0
                    for prob, next_state in get_transition_probabilities(state, action):
                        action_value += prob * (reward[state] + gamma * values[next_state])
                    action_values.append(action_value)

                new_values[state] = max(action_values)
                policy[state] = actions[np.argmax(action_values)]

                delta = max(delta, abs(new_values[state] - values[state]))

        values = new_values

        if delta < threshold:
            break

    return values, policy

def policy_iteration(grid, rewards, discount=0.99, max_iter=1000, tol=1e-4):
    # Performs policy iteration for the given reward structure.
    rows, cols = grid.shape
    policy = np.random.choice(actions, size=(rows, cols))
    value = np.zeros((rows, cols))
    terminal_states = [(0, 0), (0, 2)]  # Define terminal states
    
    print("\nInitial Random Policy:")
    display_policy(policy)
    print()
    
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_transition_probs(action, state):
        r, c = state
        if state in terminal_states:
            return [(1.0, state)]  # Terminal states stay in place
            
        outcomes = []
        
        # Define movement directions based on action
        intended_delta = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
        perpendiculars = {"Up": [(0, -1), (0, 1)], "Down": [(0, -1), (0, 1)],
                         "Left": [(-1, 0), (1, 0)], "Right": [(-1, 0), (1, 0)]}
        
        # Add intended direction with 0.8 probability
        dr, dc = intended_delta[action]
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            outcomes.append((0.8, (nr, nc)))
        else:
            outcomes.append((0.8, (r, c)))
            
        # Add perpendicular directions with 0.1 probability each
        for dr, dc in perpendiculars[action]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                outcomes.append((0.1, (nr, nc)))
            else:
                outcomes.append((0.1, (r, c)))
                
        return outcomes

    def policy_evaluation(policy):
        nonlocal value
        for _ in range(max_iter):
            new_value = np.copy(value)
            for r in range(rows):
                for c in range(cols):
                    state = (r, c)
                    if state in terminal_states:
                        new_value[r, c] = rewards[r, c]  # Use the actual reward for terminal states
                        continue
                        
                    action = policy[r, c]
                    expected_value = 0
                    for prob, next_state in get_transition_probs(action, state):
                        nr, nc = next_state
                        expected_value += prob * (rewards[r, c] + discount * value[nr, nc])
                    new_value[r, c] = expected_value
            if np.max(np.abs(new_value - value)) < tol:
                break
            value = new_value

    def policy_improvement():
        stable = True
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                if state in terminal_states:
                    continue
                    
                best_action = None
                best_value = float('-inf')
                for action in actions:
                    expected_value = 0
                    for prob, next_state in get_transition_probs(action, state):
                        nr, nc = next_state
                        expected_value += prob * (rewards[r, c] + discount * value[nr, nc])
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action
                if policy[r, c] != best_action:
                    stable = False
                policy[r, c] = best_action
        return stable

    while True:
        policy_evaluation(policy)
        if policy_improvement():
            break
    return value, policy

def display_policy(policy):
    """Displays the policy in a grid format."""
    for row in policy:
        print(" ".join([f"{cell:^5}" for cell in row]))

if __name__ == "__main__":
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

    print("-"*50)

    for r in rewards:
        reward = np.full((grid_size, grid_size), -1)
        reward[0, 0] = r  # Top left square
        reward[0, 2] = 10  # Top right square

        print(f"\nResults for reward = {r} using Policy Iteration")
        values, policy = policy_iteration(np.ones((grid_size, grid_size)), reward)
        print("State Values:")
        print(values)
        print("Policy:")
        display_policy(policy)
