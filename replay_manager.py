import itertools

import numpy as np

from main import BlackJackUltimate


class ReplayManager:
    def __init__(self):
        self.filename = 'train_states.txt'

    def save_to_file(self, training_states):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        with open('./%s' % self.filename, 'w') as f:
            cur_episode = -1

            for i in training_states:
                if cur_episode < i[0]:
                    cur_episode += 1
                    f.write('Episode: %s \n' % cur_episode)

                f.write(np.array2string(np.array(list([int(x) for x in i[1][0]])), separator=', '))
                f.write('\n')
                f.write(str(i[2]))
                f.write('\n')
                f.write(str(i[3]))
                f.write('\n')

    def load_from_file(self):
        result = []
        actions = []
        rewards = []
        with open('./%s' % self.filename, 'r') as f:
            cur_episode = -1
            counter = 0

            for line in f:
                if 'Episode' in line:
                    cur_episode += 1
                    counter = 0
                else:
                    if counter == 0:
                        result.append((cur_episode, [int(i) for i in
                                                     line.replace('\n', '').replace('[', '').replace(']', '').split(', ')]))
                    elif counter == 1:
                        actions.append((cur_episode, int(line.replace('\n', ''))))
                    elif counter == 2:
                        rewards.append((cur_episode, int(line.replace('\n', ''))))

                    counter += 1
                    if counter > 2:
                        counter = 0

        return ([list(g) for k, g in itertools.groupby(result, lambda x: x[0])],
                [list(g) for k, g in itertools.groupby(actions, lambda x: x[0])],
                [list(g) for k, g in itertools.groupby(rewards, lambda x: x[0])])

    def get_top_n(self, num, after_elem=0):
        result = self.load_from_file()[0]
        return sorted(filter(lambda x: len(x) > 3 and x[0][0] > after_elem, result), key=lambda x: len(x))[:num]

    def get_all(self):
        return self.load_from_file()


def play_replay(episode, env, root):
    for step in episode:
        grid_field = np.array(step[1]).reshape((3, 3))
        env.set_grid_field(grid_field)
        env._board.draw_replay_field(grid_field)
        root.after(750)
        root.update()


if __name__ == '__main__':
    rm = ReplayManager()
    best_episodes = rm.get_top_n(3, 1500)

    for episode in best_episodes:
        env = BlackJackUltimate()
        env.create_replay_env()

        play_replay(episode, env, env._board.get_root())

        env.run_mainloop()

