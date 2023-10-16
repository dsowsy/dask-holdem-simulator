from config import SCHEDULER_ADDRESS
from utils import *
from Decision import *
from HandCalculations import *
from distributed import Client, Worker, Pub, Sub
import asyncio
import random
import sys
from poker import Card

class Player:
    def __init__(self, player_id):
        self.play_id = player_id
        self.worker_name = f"Player{player_id}"
        self.topic = f"Player{player_id}"
        self.client = None
        self.pub = None
        self.sub = None
        self.money = 500.00
        self.cards = []
        self.board_cards = []

    def register_subscribers(self):
        self.game_state_sub = Sub("GameState", client=self.client)
        self.player_sub = Sub(self.topic, client=self.client)  # This is for player-specific messages
        self.blinds_and_dealer_sub = Sub("BlindsAndDealer", client=self.client)

    def register_publishers(self):
        self.pub = Pub(self.topic, client=self.client)
        self.join_pub = Pub("PlayerJoin", client=self.client)
        self.join_pub.put(f"{self.topic} joined")

    async def join_game(self):
        # Initialize the connection and join the game
        self.client = await Client(address=SCHEDULER_ADDRESS, set_as_default=True, asynchronous=True)
        self.register_subscribers()
        self.register_publishers()
        print(f"{self.topic} sent a join message to PlayerJoin topic!")

    async def listen_to_game_state(self):
        while True:
            game_state_message = await self.game_state_sub.get()
            topic = game_state_message.get('topic')

            # Process the game state message as needed
            if topic == 'GameEnded':
                print(f"Received game end signal. Clearing cards.")
                self.cards.clear()
                self.board_cards.clear()
            else:
                if topic == 'Flop':
                    flop_cards = ints_to_cards(game_state_message.get('cards', []))
                    self.board_cards.extend(flop_cards)
                    print(f"Received Flop cards: [ {pretty_print_cards(flop_cards)} ]")

                elif topic == 'Turn':
                    turn_card = ints_to_cards(game_state_message.get('cards', []))
                    self.board_cards.append(turn_card)

                    print(f"Received Turn card: [ {pretty_print_cards(turn_card)} ]")
                elif topic == 'River':
                    river_card = ints_to_cards(game_state_message.get('cards', []))
                    self.board_cards.append(river_card)
                    print(f"Received River card: [ {pretty_print_cards(river_card)} ]")
                # ... add more cases as needed for other game state topics ...

    async def listen_to_blinds_and_dealer(self):
        while True:
            blinds_and_dealer_message = await self.blinds_and_dealer_sub.get()
            dealer = blinds_and_dealer_message.get('dealer')
            small_blind = blinds_and_dealer_message.get('small_blind')
            big_blind = blinds_and_dealer_message.get('big_blind')

            # Check player's position
            if self.worker_name == dealer and self.worker_name == small_blind:
                self.player_position = PlayerPosition.DEALER_SMALL_BLIND
            elif self.worker_name == dealer:
                self.player_position = PlayerPosition.DEALER
            elif self.worker_name == small_blind:
                self.player_position = PlayerPosition.SMALL_BLIND
            elif self.worker_name == big_blind:
                self.player_position = PlayerPosition.BIG_BLIND
            else:
                self.player_position = PlayerPosition.NONE

            print(f"{self.worker_name} received Dealer: {dealer}, Small Blind: {small_blind}, Big Blind: {big_blind}")
            print(f"My position: {self.player_position.name}")


    async def listen_to_player_specific_messages(self):
        while True:
            try:
                player_specific_messages = await self.player_sub.get(timeout=0.1)

                topic = player_specific_messages.get('topic')

                if topic == 'GameEnded':
                    self.cards.clear()
                    print(f"{self.worker_name} received game end signal. Clearing cards.")
                    continue

                cards_received = player_specific_messages.get('cards')
                if cards_received:
                    cards_received = ints_to_cards(cards_received)
                    self.cards.extend(cards_received)
                    print(f"Cards received: {pretty_print_cards(cards_received)}")
                    print(f"My hand: {pretty_print_cards(self.cards)}")

                game_winner_message = await self.player_sub.get(timeout=0.1)
                if game_winner_message:
                    winner = game_winner_message.get('winner')
                    amount_won = game_winner_message.get('amount_won')
                    if winner and amount_won:
                        if winner == self.worker_name:
                            self.money += amount_won
                            print(f"{self.worker_name} won the pot! New balance: {self.money}")
                        else:
                            print(f"{winner} won the pot of {amount_won}!")
                    else:
                        print("Error: Invalid winner or pot amount.")
            except asyncio.TimeoutError:
                continue  # Just continue if a timeout occurs


    def deduct_money(self, amount):
        if self.money - amount < 0:
            self.money = 0
        else:
            self.money -= amount

    def add_money(self, amount):
        self.money += amount

    async def bet(self, amount):
        if amount > self.money:
            print(f"{self.worker_name} doesn't have enough money to bet {amount}. Current balance: {self.money}")
            return False
        self.deduct_money(amount)
        await self.send_move(PlayerAction.BET, amount)
        print(f"{self.worker_name} has bet {amount}. Remaining balance: {self.money}")
        return True

    async def process_data(self):
        # Start multiple tasks to listen to different topics
        asyncio.create_task(self.listen_to_game_state())
        asyncio.create_task(self.listen_to_blinds_and_dealer())
        asyncio.create_task(self.listen_to_player_specific_messages())

        while True:
            # await self.make_decision()
            await asyncio.sleep(1)

    async def send_move(self, action, amount=None):
        if action == PlayerAction.RAISE:
            self.pub.put({'action': action.value, 'amount': amount})
        else:
            self.pub.put({'action': action.value})

    async def make_decision(self):
        if not len(self.cards) == 0:
            # The player shouldn't be able to make a decision yet
            return
        if self.money <= 0:
            print(f"{self.worker_name} has no money left. Folding.")
            await self.send_move(action=PlayerAction.FOLD)
            # Exit the player process here to leave the game.
            sys.exit(0)
        else:
            decision = random.choice(list(PlayerAction))
            if decision == PlayerAction.BET:
                bet_amount = random.randint(10, 50)
                await self.bet(bet_amount)
            elif decision == PlayerAction.RAISE:
                raise_amount = random.randint(10, 50)  # Random raise amount for demonstration purposes
                await self.send_move(decision, raise_amount)
            else:
                await self.send_move(decision)

    async def main(self):
        async with Worker(SCHEDULER_ADDRESS, name=self.worker_name) as worker:
            await self.join_game()
            await self.process_data()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an integer as an argument.")
        sys.exit(1)

    if not sys.argv[1].isdigit():
        print("Please provide a valid integer as an argument.")
        sys.exit(1)

    player = Player(sys.argv[1])
    asyncio.run(player.main())
