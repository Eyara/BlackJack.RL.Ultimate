import random
from copy import deepcopy

import torch

from board_gui import BoardGUI


class BlackJackUltimate:
    def __init__(self, is_external=False):
        self._max_hand_size = 8

        self.dealer_score = 0
        self.player_score = 0

        self.dealer_cards = [('', 0)] * self._max_hand_size
        self.player_cards = [('', 0)] * self._max_hand_size

        self.was_player_stopped = False

        self._board = BoardGUI()
        self._board.on("step", self.step_event_handler)

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

        self.is_external = is_external

    def get_state(self):
        state = [self.dealer_score] + [x[1] for x in self.dealer_cards] + [self.player_score] + [x[1] for x in
                                                                                                 self.player_cards]

        return torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)

    def calculate_score(self, player_type):
        if player_type == 0:
            self.dealer_score = sum([x[1] for x in self.dealer_cards])
        else:
            self.player_score = sum([x[1] for x in self.player_cards])

    def draw_card(self, is_player=True):
        card_name = random.choice([x for x, y in self.deck_count.items() if y > 0])
        self.deck_count[card_name] -= 1

        if is_player:
            first_empty_idx = [y[1] for y in self.player_cards].index(0)
            self.player_cards[first_empty_idx] = (card_name, self.deck[card_name])
            self.calculate_score(1)

            if not self.is_external:
                self._board.update_field(self.player_score, [x[0] for x in self.player_cards if x[0] != ''], 1)
        else:
            first_empty_idx = [y[1] for y in self.dealer_cards].index(0)
            self.dealer_cards[first_empty_idx] = (card_name, self.deck[card_name])
            self.calculate_score(0)

            if not self.is_external:
                self._board.update_field(self.dealer_score, [x[0] for x in self.dealer_cards if x[0] != ''], 0)

    def change_ace_value(self, is_player=False):
        deck = self.player_cards if is_player else self.dealer_cards

        for i, card in enumerate(deck):
            if card[0].startswith('ace') and card[1] == 11:
                deck[i] = (card[0], 1)
                break

        if is_player:
            self.player_cards = deck
        else:
            self.dealer_cards = deck

        self.calculate_score(1 if is_player else 0)

        if not self.is_external:
            self._board.update_score(self.player_score, 1)
            self._board.update_score(self.dealer_score, 0)

    def reset(self):
        self.dealer_score = 0
        self.player_score = 0
        self.dealer_cards = [('', 0)] * self._max_hand_size
        self.player_cards = [('', 0)] * self._max_hand_size
        self.deck_count = deepcopy(self.deck_count_init)

        if not self.is_external:
            self._board.reset()
            self._board.draw_field()

        # first dealer's move
        self.draw_card(False)
        self.draw_card(False)

        if not self.is_external:
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
                self.change_ace_value(True)
                if self.player_score > 21:
                    return self.get_state(), -5, True
        else:
            while self.dealer_score < 21 or self.dealer_score < self.player_score:
                self.draw_card(False)
                if self.dealer_score > 21:
                    self.change_ace_value(False)
                    if self.dealer_score > 21:
                        return self.get_state(), 5, True
                if self.dealer_score >= self.player_score:
                    return self.get_state(), -5, True

        return self.get_state(), 0, False

    def external_step(self, action):
        state, reward, done = self.step(action)
        if done:
            self.reset()
        return state, reward, done

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
    blackjack_game = BlackJackUltimate()
    blackjack_game.create_env()
    blackjack_game.reset()
