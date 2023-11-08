import random
import time
from copy import deepcopy

import torch

from board_gui import BoardGUI


class BlackJackUltimate:
    def __init__(self):
        self._max_hand_size = 8

        self.dealer_score = 0
        self.player_score = 0

        self.dealer_cards = [('', 0)] * self._max_hand_size
        self.player_cards = [('', 0)] * self._max_hand_size

        self.was_player_stopped = False

        self._board = BoardGUI()
        self._board.on("step", self.step_event_handler)
        self._board.on("agent_step", self.agent_step_event_handler)

        self.is_against_agent = False

        self.deck = {
            '2_of_hearts': 2, '2_of_diamonds': 2, '2_of_clubs': 2, '2_of_spades': 2,
            '3_of_hearts': 3, '3_of_diamonds': 3, '3_of_clubs': 3, '3_of_spades': 3,
            '4_of_hearts': 4, '4_of_diamonds': 4, '4_of_clubs': 4, '4_of_spades': 4,
            '5_of_hearts': 5, '5_of_diamonds': 5, '5_of_clubs': 5, '5_of_spades': 5,
            '6_of_hearts': 6, '6_of_diamonds': 6, '6_of_clubs': 6, '6_of_spades': 6,
            '7_of_hearts': 7, '7_of_diamonds': 7, '7_of_clubs': 7, '7_of_spades': 7,
            '8_of_hearts': 8, '8_of_diamonds': 8, '8_of_clubs': 8, '8_of_spades': 8,
            '9_of_hearts': 9, '9_of_diamonds': 9, '9_of_clubs': 9, '9_of_spades': 9,
            '10_of_hearts': 10, '10_of_diamonds': 10, '10_of_clubs': 10, '10_of_spades': 10,
            'jack_of_hearts': 10, 'jack_of_diamonds': 10, 'jack_of_clubs': 10, 'jack_of_spades': 10,
            'queen_of_hearts': 10, 'queen_of_diamonds': 10, 'queen_of_clubs': 10, 'queen_of_spades': 10,
            'king_of_hearts': 10, 'king_of_diamonds': 10, 'king_of_clubs': 10, 'king_of_spades': 10,
            'ace_of_hearts': 11, 'ace_of_diamonds': 11, 'ace_of_clubs': 11, 'ace_of_spades': 11
        }

        self.deck_count_init = {
            '2_of_hearts': 4, '2_of_diamonds': 4, '2_of_clubs': 4, '2_of_spades': 4,
            '3_of_hearts': 4, '3_of_diamonds': 4, '3_of_clubs': 4, '3_of_spades': 4,
            '4_of_hearts': 4, '4_of_diamonds': 4, '4_of_clubs': 4, '4_of_spades': 4,
            '5_of_hearts': 4, '5_of_diamonds': 4, '5_of_clubs': 4, '5_of_spades': 4,
            '6_of_hearts': 4, '6_of_diamonds': 4, '6_of_clubs': 4, '6_of_spades': 4,
            '7_of_hearts': 4, '7_of_diamonds': 4, '7_of_clubs': 4, '7_of_spades': 4,
            '8_of_hearts': 4, '8_of_diamonds': 4, '8_of_clubs': 4, '8_of_spades': 4,
            '9_of_hearts': 4, '9_of_diamonds': 4, '9_of_clubs': 4, '9_of_spades': 4,
            '10_of_hearts': 4, '10_of_diamonds': 4, '10_of_clubs': 4, '10_of_spades': 4,
            'jack_of_hearts': 4, 'jack_of_diamonds': 4, 'jack_of_clubs': 4, 'jack_of_spades': 4,
            'queen_of_hearts': 4, 'queen_of_diamonds': 4, 'queen_of_clubs': 4, 'queen_of_spades': 4,
            'king_of_hearts': 4, 'king_of_diamonds': 4, 'king_of_clubs': 4, 'king_of_spades': 4,
            'ace_of_hearts': 4, 'ace_of_diamonds': 4, 'ace_of_clubs': 4, 'ace_of_spades': 4
        }

        self.deck_count = deepcopy(self.deck_count_init)

        # standard value for BlackJack
        self.deck_num = 6
        self.deck_count.update((x, y * self.deck_num) for x, y in self.deck_count.items())

    def get_state(self):
        state = [self.dealer_score] + [x[1] for x in self.dealer_cards] + [self.player_score] + [x[1] for x in
                                                                                                 self.player_cards]
        return torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)

    def calculate_score(self, player_type):
        if player_type == 0:
            self.dealer_score = sum([x[1] for x in self.dealer_cards])
        else:
            self.player_score = sum([x[1] for x in self.player_cards])

    def draw_card(self, is_player=True, is_external=False):
        card_name = random.choice([x for x, y in self.deck_count.items() if y > 0])
        self.deck_count[card_name] -= 1

        if is_player:
            first_empty_idx = [y[1] for y in self.player_cards].index(0)
            self.player_cards[first_empty_idx] = (card_name, self.deck[card_name])
            self.calculate_score(1)

            if not is_external:
                self._board.update_field(self.player_score, [x[0] for x in self.player_cards if x[0] != ''], 1)
        else:
            first_empty_idx = [y[1] for y in self.dealer_cards].index(0)
            self.dealer_cards[first_empty_idx] = (card_name, self.deck[card_name])
            self.calculate_score(0)

            if not is_external:
                self._board.update_field(self.dealer_score, [x[0] for x in self.dealer_cards if x[0] != ''], 0)

    def reset(self, is_external=False):
        self.dealer_score = 0
        self.player_score = 0
        self.dealer_cards = [('', 0)] * self._max_hand_size
        self.player_cards = [('', 0)] * self._max_hand_size
        self.deck_count = deepcopy(self.deck_count_init)
        self._board.reset()

        if not is_external:
            self._board.draw_field()

        # first dealer's move
        self.draw_card(False, is_external)
        self.draw_card(False, is_external)

        if not is_external:
            self._board.run_mainloop()

        return self.get_state()

    @staticmethod
    def get_action_num():
        return 2

    def get_action_sample(self):
        return random.choice(range(0, self.get_action_num()))

    '''
    0 - hit
    1 - stand
    '''

    def step(self, action):
        if action == 0:
            self.draw_card(True)
            if self.player_score > 21:
                return self.get_state(), -5, True
        else:
            while self.dealer_score < 21 or self.dealer_score < self.player_score:
                self.draw_card(False)
                if self.dealer_score > 21:
                    return self.get_state(), 5, True
                if self.dealer_score >= self.player_score:
                    current_state = self.get_state()
                    return self.get_state(), -5, True

        return self.get_state(), 0, False

    def external_step(self, action):
        state, reward, done = self.step(action, True)
        if done:
            self.reset(True)
        return state, reward, done

    def set_play_agent_mode(self):
        self.is_against_agent = True

    def agent_step(self):
        while True:
            # action = self._x_agent.select_action(self.get_state(), self.get_action_sample())
            # copy_game = TicTacToeGame()
            # copy_game.set_grid_field(self._grid_field)
            # action = self._x_agent.select_action(copy_game, self.get_action_sample())
            action = action.item()
            state, reward, done = self.step(action)
            if done:
                self.reset()
                break

    def agent_step_event_handler(self, data):
        if self.is_against_agent:
            self.agent_step()

    def step_event_handler(self, action):
        state, reward, done = self.step(action)
        if done:
            self.reset()

    def create_env(self):
        self._board.create_env()

    def create_replay_env(self):
        self._board.create_replay_env()

    def run_mainloop(self):
        self._board.run_mainloop()


if __name__ == '__main__':
    tg = BlackJackUltimate()
    tg.create_env()
    tg.reset()
