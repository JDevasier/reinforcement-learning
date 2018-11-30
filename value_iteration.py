import sys
import math

# one spot in the world (1,1), (1,2), etc.
class state:
    def __init__(self, x, y, terminal = False, reward = None, accessible = True, utility = 0):
        self.x = x
        self.y = y
        self.terminal = terminal
        self.reward = reward
        self.accessible = accessible
        self.utility = utility

class environment:
    def __init__(self, states, rows, cols):
        self.states = states
        self.rows = rows
        self.cols = cols
    
    def get_state(self, row, col):
        if row < 1:
            row = 1
        if col < 1:
            col = 1
        if row > self.rows:
            row = self.rows
        if col > self.cols:
            col = self.cols
        return self.states[self.rows - row][col - 1]
        
            
    def actions(self, state):
        if state.terminal or not state.accessible:
            return []
        acts = {}
        acts["down"] = self.get_state(state.x-1, state.y)
        acts["up"] = self.get_state(state.x+1, state.y)
        acts["left"] = self.get_state(state.x, state.y-1)
        acts["right"] = self.get_state(state.x, state.y+1)
        return acts

def get_file(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines

def get_environment(env_string):
    arr = [line.replace("\n", "").split(",") for line in env_string]
    env = []
    rows = len(arr)
    for x, row in enumerate(arr):
        env_row = []
        for y, val in enumerate(row):
            if val == ".":
                env_row.append(state(rows - x, y+1))
            elif not val == "X":
                env_row.append(state(rows - x, y+1, terminal=True, reward=float(val), utility=float(val)))
            else:
                env_row.append(state(rows - x, y+1, accessible=False))
        env.append(env_row)
    return environment(env, len(env), len(env[0]))

def print_environment(env):
    print("size:", env.rows, env.cols)
    for row in env.states:
        r_arr = []
        r_arr.append(row[0].x)
        for state in row:
            if state.terminal:
                r_arr.append(state.reward)
            elif not state.accessible:
                r_arr.append("X")
            else:
                r_arr.append(".")
        print(r_arr)
    cols = [state.y for state in env.states[0]]
    cols.insert(0, "")
    print(cols)

def remove_backwards(A, action):
    Actions = A
    if action == "up":
        Actions.remove("down")
    elif action == "down":
        Actions.remove("up")
    elif action == "left":
        Actions.remove("right")
    elif action == "right":
        Actions.remove("left")
    return Actions

def value_iteration(environment_file, non_terminal_reward, gamma, k):
    env_str = get_file(environment_file)
    S = get_environment(env_str)
    N = len(S.states)
    U = [0] * N
    A = ["up", "down", "left", "right"]
    error = 0.01
    
    while True:
        delta = 0
        U_new = [st.utility for row in S.states for st in row]

        for row in S.states:
            for s in row:
                if s.terminal:
                    s.utility = s.reward
                elif s.accessible:
                    new_states = S.actions(s)

                    actions = {}
                    _A = A.copy()
                    for a in _A:
                        possible_actions = remove_backwards(_A, a)
                        possible_actions.remove(a)
                        
                        s1 = new_states[a] # P(s1 | s, a) = 0.8
                        s2 = new_states[possible_actions.pop()] # P(s2 | s, a) = 0.1
                        s3 = new_states[possible_actions.pop()] # P(s3 | s, a) = 0.1
                        
                        actions[a] = 0.8 * s1.utility + 0.1 * s2.utility + 0.1 * s3.utility
                    
                    s.utility = non_terminal_reward + gamma * max(actions.values())
                    print(s.utility)

            #if delta < error * (1 - gamma) / gamma:
            #   break            


def main():
    value_iteration(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))

main()