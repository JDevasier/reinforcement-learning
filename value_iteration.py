import sys
import math


class state:
    def __init__(self, x, y, terminal=False, reward=None, accessible=True, utility=0):
        self.x = x
        self.y = y
        self.terminal = terminal
        self.reward = reward
        self.accessible = accessible
        self.utility = utility
        self._utility = utility
        self.action = ""


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
                env_row.append(state(rows - x, y+1, terminal=True,
                                     reward=float(val), utility=float(val)))
            else:
                env_row.append(
                    state(rows - x, y+1, accessible=False, utility=0))
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
    Actions = A.copy()
    if action == "up":
        Actions.remove("down")
    elif action == "down":
        Actions.remove("up")
    elif action == "left":
        Actions.remove("right")
    elif action == "right":
        Actions.remove("left")
    return Actions


def argmax(a):
    highest = (-1, -1)
    for key, val in a.items():
        if val > highest[1]:
            highest = (key, val)
    return highest


def get_action_utilities(S, A, s):
    new_states = S.actions(s)
    s._utility = s.utility
    actions = {}
    _A = A.copy()
    for a in _A:
        possible_actions = remove_backwards(_A, a)
        possible_actions.remove(a)

        s1 = new_states[a]  # P(s1 | s, a) = 0.8
        if not s1.accessible:
            s1 = s
        # P(s2 | s, a) = 0.1
        s2 = new_states[possible_actions.pop()]
        if not s2.accessible:
            s2 = s
        # P(s3 | s, a) = 0.1
        s3 = new_states[possible_actions.pop()]
        if not s3.accessible:
            s3 = s

        actions[a] = 0.8 * s1._utility + 0.1 * \
            s2._utility + 0.1 * s3._utility

    return actions


def value_iteration(environment_file, non_terminal_reward, gamma, k):
    env_str = get_file(environment_file)
    S = get_environment(env_str)
    N = len(S.states)
    A = ["up", "down", "left", "right"]
    error = 0.001

    for row in S.states:
        for s in row:
            print((s.x, s.y), s.utility)

    for _ in range(k):
        delta = 0

        s = S.states[0][0]

        for row in S.states:
            for s in row:
                if s.terminal:
                    s.utility = s.reward
                elif s.accessible:
                    actions = get_action_utilities(S, A, s)
                    best = argmax(actions)
                    s.utility = non_terminal_reward + gamma * best[1]
                    s.action = best[0]

                    if abs(s._utility - s.utility) > delta:
                        delta = abs(s._utility - s.utility)

        #print(delta, error * (1 - gamma) / gamma)

    for row in S.states:
        print(["{:6.3f}".format(_st.utility) for _st in row])
        #print("{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(row[0].utility, row[1].utility, row[2].utility, row[3].utility))


def main():
    value_iteration(sys.argv[1], float(sys.argv[2]),
                    float(sys.argv[3]), int(sys.argv[4]))


main()
