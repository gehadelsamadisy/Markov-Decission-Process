import numpy as np

# Define constants
gamma = 0.99  # Discount factor
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

        print(f"\nResults for reward = {r}")
        values, policy = value_iteration(reward=reward)
        print("State Values:")
        print(values)
        print("Policy:")
        display_policy(policy)
