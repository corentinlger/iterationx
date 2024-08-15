import jax.numpy as jnp
import matplotlib.pyplot as plt

from flax import struct

NUM_STATES = 9
NUM_ACTIONS = 4

# Define the transition probabilities and rewards : P[s][a] = [p(s_p|s, a), s_p, r, done]
P = {
    0: {0: [(1.0, 0, -1, False)], 1: [(1.0, 3, -1, False)], 2: [(1.0, 0, -1, False)], 3: [(1.0, 1, -1, False)]},
    1: {0: [(1.0, 1, -1, False)], 1: [(1.0, 4, -1, False)], 2: [(1.0, 0, -1, False)], 3: [(1.0, 2, -1, False)]},
    2: {0: [(1.0, 2, -1, False)], 1: [(1.0, 5, -1, False)], 2: [(1.0, 1, -1, False)], 3: [(1.0, 2, -1, False)]},
    3: {0: [(1.0, 0, -1, False)], 1: [(1.0, 6, -1, False)], 2: [(1.0, 3, -1, False)], 3: [(1.0, 4, -1, False)]},
    4: {0: [(1.0, 1, -1, False)], 1: [(1.0, 7, -1, False)], 2: [(1.0, 3, -1, False)], 3: [(1.0, 5, -1, False)]},
    5: {0: [(1.0, 2, -1, False)], 1: [(1.0, 8, 10, False)], 2: [(1.0, 4, -1, False)], 3: [(1.0, 5, -1, False)]},
    6: {0: [(1.0, 3, -1, False)], 1: [(1.0, 6, -1, False)], 2: [(1.0, 6, -1, False)], 3: [(1.0, 7, -1, False)]},
    7: {0: [(1.0, 4, -1, False)], 1: [(1.0, 7, -1, False)], 2: [(1.0, 6, -1, False)], 3: [(1.0, 8, 10, False)]},
    8: {0: [(1.0, 5, -1, True)], 1: [(1.0, 8, 10, False)], 2: [(1.0, 7, -1, False)], 3: [(1.0, 8, 10, False)]}
}

@struct.dataclass
class MDP:
    """MDP class containing states, actions, transition probabilities and rewards

    :return: mdp object
    """
    states: jnp.array
    actions: jnp.array
    transition_probabilities: jnp.ndarray
    rewards: jnp.ndarray

    @classmethod
    def create(cls, num_states=NUM_STATES, num_actions=NUM_ACTIONS, P=P):
        transition_probabilities = jnp.zeros((num_states, num_actions, num_states))
        rewards = jnp.zeros((num_states, num_actions, num_states))
        states = jnp.arange(num_states)
        actions = jnp.arange(num_actions)

        for s in states:
            for a in actions:
                for prob, next_state, reward, done in P[int(s)][int(a)]:
                    transition_probabilities = transition_probabilities.at[s, a, next_state].set(prob)
                    rewards = rewards.at[s, a, next_state].set(reward)

        return cls(states, actions, transition_probabilities, rewards)
    
    def infos(self):
        """Print the details of the MDP."""
        print("MDP infos:")
        print(f"S: {self.states}")
        print(f"A: {self.actions}")
        print(f"P(s'|s,a).shape: {self.transition_probabilities.shape}")
        print(f"R(s', a, s).shape: {self.rewards.shape}")
    

def print_policy_and_value(policy, values):
    """Print current policy and values for all states

    :param policy: policy
    :param value: values
    """
    for s in range(len(policy)):
        print(f"Policy({s}) = {policy[s]}  ; Value({s}) = {values[s]:.2f}")

def visualize_policy_and_values(policy, values, vmin=0, vmax=100):
    """Plot a heatmap with current policy and values for all states

    :param policy: policy
    :param values: values
    :param vmin: min plotted value, defaults to 0
    :param vmax: max plotted value, defaults to 100
    """
    value_grid = values.reshape((3, 3))
    plt.figure(figsize=(6, 6))
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(value_grid, cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(label='Value')

    # Mapping of policy actions to symbols
    policy_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for i in range(3):
        for j in range(3):
            state_index = i * 3 + j
            plt.text(j, i - 0.1, f'S{state_index}\n{values[state_index]:.2f}', ha='center', va='center', color='white')
            plt.text(j, i + 0.2, policy_symbols[int(policy[state_index])], ha='center', va='center', color='white', fontsize=20)

    plt.title('Value Function and Policy Heatmap')
    plt.xticks([])
    plt.yticks([])
    plt.show()
