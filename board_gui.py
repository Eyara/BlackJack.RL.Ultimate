import tkinter as tk
from tkinter import font

from PIL import Image, ImageTk


class ActionButton(tk.Button):
    def __init__(self, master=None, **kwargs, ):
        tk.Button.__init__(self, master, compound='center', **kwargs)
        self.config(width=2, height=1, font=font.Font(size=12), background="white", fg="black",
                    activebackground="white", activeforeground="black", )


class BoardGUI:
    def __init__(self):
        self._root = None
        self._dealer_card_frame = None
        self._player_card_frame = None

        self._dealer_deck = []
        self._player_deck = []

        self._dealer_score = 0
        self._player_score = 0

        self._card_images = []

        self.listeners = {}
        self.step_event_name = "step"
        self.step_agent_event_name = "agent_step"

    def get_root(self):
        return self._root

    def create_replay_env(self):
        self._root = tk.Tk()
        self._root.title("BlackJack's replay")
        self._root.geometry("1200x600")

    def run_mainloop(self):
        self._root.mainloop()

    def create_env(self):
        self._root = tk.Tk()
        self._root.title("BlackJack")
        self._root.geometry("1000x800")
        self.draw_field()

    def draw_replay_field(self, grid):
        pass

    def draw_field(self):
        self._root.configure(bg='green')
        for i in range(12):
            self._root.columnconfigure(i, weight=2)
        for i in range(12):
            self._root.rowconfigure(i, weight=2)

        self._dealer_card_frame = tk.Frame(self._root, relief="sunken", borderwidth=1, bg="black")
        self._dealer_card_frame.grid(row=0, column=2, sticky='ew', columnspan=8)

        dealer_score = tk.Label(self._root, text="Dealer: %s" % self._dealer_score, background="green",
                                fg="white", font=font.Font(size=18))
        dealer_score.grid(row=0, column=0, sticky='ew')
        player_score = tk.Label(self._root, text="Player: %s" % self._player_score, background="green",
                                fg="white", font=font.Font(size=18))
        player_score.grid(row=10, column=0, sticky='ew')

        self._player_card_frame = tk.Frame(self._root, relief="sunken", borderwidth=1, bg="black")
        self._player_card_frame.grid(row=10, column=2, sticky='ew', columnspan=8)

        btn_hit = ActionButton(text="Hit", command=self.hit_click_handle)

        btn_hit.grid(row=6, column=3, sticky='ew')

        btn_stand = ActionButton(text="Stand", command=self.stand_click_handle)

        btn_stand.grid(row=6, column=6, sticky='ew')

    '''
    Type.
    0 - Dealer
    1 - Player
    '''

    def update_field(self, score, desk, type):
        self.update_score(score, type)
        self.update_deck(desk, type)

    def update_score(self, score, player_type):
        label_text = "Dealer: %s" if player_type == 0 else "Player: %s"

        if player_type == 0:
            self._dealer_score = score
        else:
            self._player_score = score

        score_label = tk.Label(self._root, text=label_text % score, background="green",
                               fg="white", font=font.Font(size=18)
                               )

        row = 0 if player_type == 0 else 10
        score_label.grid(row=row, column=0, sticky='ew')

    def update_deck(self, deck, player_type):
        card_frame = self._dealer_card_frame if player_type == 0 else self._player_card_frame
        player_deck = self._dealer_deck if player_type == 0 else self._player_deck

        new_cards = deck[len(player_deck):]
        for card in new_cards:
            self.draw_card(card_frame, card)

        if player_type == 0:
            self._dealer_deck = deck
        else:
            self._player_deck = deck

    def draw_card(self, frame, card_name):
        self._card_images.append(ImageTk.PhotoImage(Image.open('./cards/%s.png' % card_name).resize((150, 250))))
        tk.Label(frame, image=self._card_images[len(self._card_images) - 1], relief="raised").pack(side="left")

    def reset(self):
        self._dealer_deck = []
        self._player_deck = []

        self._dealer_score = 0
        self._player_score = 0

        self._card_images = []

        self.draw_field()

    def hit_click_handle(self):
        self.emit(self.step_event_name, 0)

    def stand_click_handle(self):
        self.emit(self.step_event_name, 1)

    def on(self, event_name, callback):
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(callback)

    def emit(self, event_name, data=None):
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                callback(data)
