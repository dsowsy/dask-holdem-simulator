from config import SCHEDULER_ADDRESS
from utils import *
from HandCalculations import *
from poker import Card
from distributed import Client, Pub, Sub
import random
import asyncio
import sys
import time

MIN_PLAYERS = 2
MAX_PLAYERS = 22

class GameBoard:
    def __init__(self, buy_in=20):
        self.buy_in = buy_in
        self.joined_players = []
        self.small_blind_position = 0
        self.deck = list(Card)
        random.shuffle(self.deck)
        self.subs = {}
        self.pubs = {}
        self.small_blind = 5
        self.big_blind = 10
        self.pot = 0
        self.active_players = []
        self.hands = {}
        self.turn_card = None
        self.river_card = None

    async def initialize_connections(self):
        self.client = await Client(address=SCHEDULER_ADDRESS, set_as_default=True, asynchronous=True)
        self.join_sub = Sub("PlayerJoin", client=self.client)

        while self.client.status != "running":
            print("Waiting for client to connect to scheduler...")
            await asyncio.sleep(0.1)

    async def deal_cards_to_players(self):
        for _ in range(2):  # Deal two cards to each player
            for player in self.joined_players:
                card = self.deck.pop()
                if player not in self.hands:
                    self.hands[player] = []
                self.hands[player].append(card)
                self.pubs[player].put({'topic': player, 'cards': cards_to_ints([card])})


    async def set_blinds_and_dealer(self):
        # Check for heads-up play (2 players)
        if len(self.joined_players) == 2:
            dealer_player = self.joined_players[self.small_blind_position]
            small_blind_player = dealer_player
            big_blind_player = self.joined_players[(self.small_blind_position + 1) % 2]
        else:
            # Assuming a cyclic order for dealer, small blind, and big blind
            dealer_position = (self.small_blind_position - 1) % len(self.joined_players)
            dealer_player = self.joined_players[dealer_position]
            small_blind_player = self.joined_players[self.small_blind_position]
            big_blind_player = self.joined_players[(self.small_blind_position + 1) % len(self.joined_players)]

        print(f"Dealer: {dealer_player}")
        print(f"Small Blind: {small_blind_player}")
        print(f"Big Blind: {big_blind_player}")

        self.small_blind_position = (self.small_blind_position + 1) % len(self.joined_players)

        blinds_and_dealer_pub = Pub("BlindsAndDealer", client=self.client)
        blinds_and_dealer_pub.put({
            'dealer': dealer_player,
            'small_blind': small_blind_player,
            'big_blind': big_blind_player
        })

    async def set_blinds(self):
        small_blind_player = self.joined_players[self.small_blind_position]
        big_blind_player = self.joined_players[(self.small_blind_position + 1) % len(self.joined_players)]
        print(f"Small Blind: {small_blind_player}")
        print(f"Big Blind: {big_blind_player}")
        self.small_blind_position = (self.small_blind_position + 1) % len(self.joined_players)

    async def deal_flop(self):
        self.flop_cards = [self.deck.pop() for _ in range(3)]
        print(f"Flop: [ {pretty_print_cards(self.flop_cards)} ]")
        game_state_pub = Pub("GameState", client=self.client)
        game_state_pub.put({'topic': 'Flop', 'cards': cards_to_ints(self.flop_cards)})

    async def deal_turn(self):
        self.turn_card = self.deck.pop()
        print(f"Turn: [ {pretty_print_card(self.turn_card)} ]")
        game_state_pub = Pub("GameState", client=self.client)
        game_state_pub.put({'topic': 'Turn', 'cards': cards_to_ints([self.turn_card])})

    async def deal_river(self):
        self.river_card = self.deck.pop()
        print(f"River:[ {pretty_print_card(self.river_card)} ]")
        game_state_pub = Pub("GameState", client=self.client)
        game_state_pub.put({'topic': 'River', 'cards': cards_to_ints([self.river_card])})

    async def process_move(self, player, game_round):
        print(f"Waiting for move from {player}...")
        try:
            move = await asyncio.wait_for(self.subs[player].get(), timeout=10)  # give player 30 seconds to make a move
        except asyncio.TimeoutError:
            print(f"{player} did not make a move in time. Automatically folding.")
            move = {'action': PlayerAction.FOLD.value}

        action = move.get('action')

        if action == PlayerAction.RAISE.value:
            amount = move.get('amount')
            self.pot += amount
            print(f"{player} raised by {amount}. Total pot: {self.pot}")

        elif action == PlayerAction.CALL.value:
            call_amount = 10
            self.pot += call_amount
            print(f"{player} called. Total pot: {self.pot}")

        elif action == PlayerAction.FOLD.value:
            self.active_players.remove(player)
            print(f"{player} folded.")

        elif action == PlayerAction.CHECK.value:
            print(f"{player} checked.")

        # Handle game end scenarios
        if len(self.active_players) == 1:
            winner = self.active_players[0]
            print(f"{winner} wins the pot of {self.pot}!")
            self.pubs[winner].put({'winner': winner, 'amount_won': self.pot})
            self.pot = 0  # Reset pot
            self.end_game()
            return False

        # If you've reached the river, calculate the winner
        elif game_round == 3:  # River betting round
            board_hand = self.flop_cards
            if self.turn_card is not None:
                board_hand.append(self.turn_card)
            if self.river_card is not None:
                board_hand.append(self.river_card)

            winners = self.determine_winner(board_hand)

            if not winners:
                # Logic for handling the case where there's no clear winner, or splitting the pot among active players.
                split_amount = self.pot // len(self.active_players)
                for player in self.active_players:
                    self.pubs[player].put({'winner': player, 'amount_won': split_amount})

            for winner in winners:
                self.pubs[winner].put({'winner': winner, 'amount_won': self.pot})
                print(f"{winner} splits the pot and receives {self.pot / len(winners)}!")

            self.end_game()
            time.sleep(2)  # Sleep for 2 seconds

            return False

        return True

    async def wait_for_players(self):
        joined_players = set()

        while True:
            print(f"Waiting {30} seconds for players to join...", end='', flush=True)
            time_to_wait = 30  # seconds

            while time_to_wait > 0:
                try:
                    message = await asyncio.wait_for(self.join_sub.get(), timeout=1)
                    print(f"Received message: {message}")  # Debug print
                    player, status = message.split(' ')  # Assuming "PlayerName joined" format
                    joined_players.add(player)

                    if player not in self.pubs:
                        self.pubs[player] = Pub(player, client=self.client)
                        self.subs[player] = Sub(player, client=self.client)

                except asyncio.TimeoutError:
                    pass

                time_to_wait -= 1
                print(f"\rWaiting {time_to_wait} seconds for players to join...", end='', flush=True)

            if MIN_PLAYERS <= len(joined_players) <= MAX_PLAYERS:
                self.joined_players = list(joined_players)
                break
            else:
                print(f"\nNot enough players joined. We have {len(joined_players)} players. Retrying.")
                joined_players.clear()

    def end_game(self):
        self.pot = 0
        self.active_players = self.joined_players.copy()
        self.deck = list(Card)
        random.shuffle(self.deck)
        game_state_pub = Pub("GameState", client=self.client)
        game_state_pub.put({'topic': 'GameEnded'})
        print("Game has ended. Notifying players.")



    def determine_winner(self, board_hand):
        player_hand_strengths = {}  # player: hand_strength
        for player in self.active_players:
            player_hand_strengths[player] = evaluate_best_hand(self.hands[player], board_hand)

        # Find the highest hand strength
        max_strength = max(player_hand_strengths.values())

        # Find all players with the winning hand strength
        winners = [player for player, strength in player_hand_strengths.items() if strength == max_strength]

        if self.pot == 0:
            print("Error: Attempt to distribute empty pot. Debug necessary.")

        # If there's only one winner
        if len(winners) == 1:
            winner = winners[0]
            self.pubs[winner].put({'winner': winner, 'amount_won': self.pot})
            print(f"{winner} wins the pot of {self.pot}!")
        else:
            # Multiple winners - split the pot
            split_amount = self.pot // len(winners)
            for winner in winners:
                self.pubs[winner].put({'winner': winner, 'amount_won': split_amount})
                print(f"{winner} splits the pot and receives {split_amount}!")
        return winners

    async def main(self):
        await self.initialize_connections()
        await self.wait_for_players()
        self.active_players = self.joined_players.copy()

        # Proceed with the game
        while True:
            print("\nStarting a new game...\n")
            self.deck = list(Card)
            random.shuffle(self.deck)

            await self.deal_cards_to_players()
            await self.set_blinds_and_dealer()

            # Process moves for each betting round
            for game_round in range(4):  # Pre-flop, flop, turn, and river betting rounds
                for player in self.active_players:
                    await self.process_move(player, game_round)

                # Deal community cards after the pre-flop betting round and after every subsequent round
                if game_round == 1:
                    await self.deal_flop()
                elif game_round == 2:
                    await self.deal_turn()
                elif game_round == 3:
                    await self.deal_river()

if __name__ == "__main__":
    buy_in = 20  # Default value
    if len(sys.argv) > 1:
        try:
            new_buy_in = float(sys.argv[1])
            if new_buy_in > 0:
                buy_in = new_buy_in
            else:
                print("Invalid buy-in provided. Defaulting to 20.")
        except ValueError:
            print("Invalid buy-in provided. Defaulting to 20.")
    gameboard = GameBoard(buy_in)
    asyncio.run(gameboard.main())

