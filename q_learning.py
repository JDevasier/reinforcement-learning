import sys
import math
import random


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

        if self.terminal:
            self.Q = {"none":0}
            self.N_sa = {"none":0}
        else:
            self.Q = {"up": 0, "down": 0, "left": 0, "right": 0}
            self.N_sa = {"up": 0, "down": 0, "left": 0, "right": 0}


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
        if state.terminal:
            return {"none":state}
        
        if not state.accessible:
            return {}
            
        down = self.get_state(state.x-1, state.y)
        if not down.accessible:
            down = state

        up = self.get_state(state.x+1, state.y)
        if not up.accessible:
            up = state

        left = self.get_state(state.x, state.y-1)
        if not left.accessible:
            left = state

        right = self.get_state(state.x, state.y+1)
        if not right.accessible:
            right = state

        return {"down":down, "up":up, "left":left, "right":right}


def get_file(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines


def get_environment(env_string, non_terminal_reward):
    arr = [line.replace("\n", "").split(",") for line in env_string]
    env = []
    rows = len(arr)
    for x, row in enumerate(arr):
        env_row = []
        for y, val in enumerate(row):
            if val == ".":
                env_row.append(state(rows - x, y+1, reward=float(non_terminal_reward)))
            elif not val == "X":
                env_row.append(state(rows - x, y+1, terminal=True, reward=float(val), utility=float(val)))
            else:
                env_row.append(state(rows - x, y+1, accessible=False, utility=0))
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
    highest = (-9999999999999, -9999999999999)
    ties = []
    for key, val in a.items():
        if val > highest[1]:
            highest = (key, val)
        elif val == highest[1]:
            if not key in ties:
                ties.append((key, val))
            if not highest in ties:
                ties.append(highest)

    if len(ties) == 0:
        return highest
    else:
        return random.choice(ties)


def eta_function(N):
    return 1 / N


def f_function(u, n, N_e):
    if n < N_e:
        return 1
    else:
        return u


def choose_state(env):
    a = [s for row in env.states for s in row if s.accessible and not s.terminal]
    return random.choice(a)


def get_action_utilities(S, s):
    if s.terminal:
        return {"none":s.reward}

    new_states = S.actions(s)
    s._utility = s.utility
    actions = {}
    _A = list(S.actions(s).keys())
    for a in _A:
        possible_actions = remove_backwards(_A, a)
        possible_actions.remove(a)
        
        s1 = new_states[a]  # P(s1 | s, a) = 0.8
        # P(s2 | s, a) = 0.1
        s2 = new_states[possible_actions.pop()]
        # P(s3 | s, a) = 0.1
        s3 = new_states[possible_actions.pop()]

        actions[a] = 0.8 * s1._utility + 0.1 *  s2._utility + 0.1 * s3._utility

    return actions


def q_learning_update(S, s, r, _s, a, gamma, eta, non_terminal_reward):
    if _s.terminal:
        _s.Q["none"] = _s.reward

    if not s == None:
        if a in s.N_sa:
            s.N_sa[a] += 1
        else:
            s.N_sa[a] = 1
        
        c = eta(s.N_sa[a])

        #best_util = argmax(get_action_utilities(S, _s))
        best_util = argmax(_s.Q)
        # print(_s.Q)
        
        s.Q[a] = (1 - c) * s.Q[a] + c * (r + gamma * best_util[1])
        # print(s.Q, a, c, r, best_util)

_s, s, r, a = None, None, None, None

def AgentMode_Q_Learning(environment_file, non_terminal_reward, gamma, k, N_e):
    env_str = get_file(environment_file)
    S = get_environment(env_str, non_terminal_reward)
    count = 0

    while(True):
        global _s, s, r, a
        _s = choose_state(S)
        a, s, r = None, None, None

        # print("Starting at ", (_s.x, _s.y))

        if count >= k:
            break
        while(True):
            if count >= k:
                break
            if not s == None:
                _s = S.actions(s)[a]
                #r = s.reward
            q_learning_update(S, s, r, _s, a, gamma, eta_function, non_terminal_reward)
            count += 1

            # print("Reached", (_s.x, _s.y), "by going", a)
            
            if _s.terminal:
                # print("Reached Terminal State, restarting", (_s.x, _s.y))
                break

            f_values = {_a: f_function(_s.Q[_a], _s.N_sa[_a], N_e) for _a in _s.Q}
            # print(f_values)
            a = argmax(f_values)[0]
            # print("Taking action", a)
            
            if random.random() <= 0.8:
                s = _s
                r = _s.reward

                _s = S.actions(_s)[a]
            else:
                _actions = remove_backwards(list(S.actions(_s).keys()).copy(), a)

                _actions.remove(a)
                _action = random.choice(_actions)

                s = _s
                r = s.reward

                _s = S.actions(_s)[_action]

    # for row in S.states:
    #     for st in row:
    #         if st.accessible and not st.terminal:
    #             print((st.x, st.y), argmax(st.Q))

    for row in S.states:
        #print("{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(argmax(row[0].Q)[1], argmax(row[1].Q)[1], argmax(row[2].Q)[1], argmax(row[3].Q)[1]))
        print(["{:6.3f}".format(argmax(st.Q)[1]) for st in row])

 
def main():
    AgentMode_Q_Learning(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))


main()
