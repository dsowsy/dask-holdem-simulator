from poker import Rank, Suit, Card
from itertools import combinations
from colorama import Fore, init, Style
import torch
from enum import Enum, auto

init(autoreset=True)

# Conveniences for working with the poker lib
# - enums
# - color pretty printing functions
# - conversion of cards from poker <> treys
# - card utils

RANKS = {
    Rank.DEUCE: 0,
    Rank.THREE: 1,
    Rank.FOUR: 2,
    Rank.FIVE: 3,
    Rank.SIX: 4,
    Rank.SEVEN: 5,
    Rank.EIGHT: 6,
    Rank.NINE: 7,
    Rank.TEN: 8,
    Rank.JACK: 9,
    Rank.QUEEN: 10,
    Rank.KING: 11,
    Rank.ACE: 12
}

SUITS = {
    Suit.CLUBS: 0,
    Suit.DIAMONDS: 1,
    Suit.HEARTS: 2,
    Suit.SPADES: 3
}

class PlayerAction(Enum):
    CALL = "CALL"
    RAISE = "RAISE"
    FOLD = "FOLD"
    CHECK = "CHECK"
    BET = "BET" 

class PlayerPosition(Enum):
    NONE = auto()
    SMALL_BLIND = auto()
    BIG_BLIND = auto()
    DEALER = auto()
    DEALER_SMALL_BLIND = auto()  # for the special case when there are only 2 players

def pretty_print_card(card):
    card_str = str(card)
    color = Fore.RED if card.suit in [Suit.HEARTS, Suit.DIAMONDS] else Fore.GREEN
    return f"{color}{card_str}{Style.RESET_ALL}"

def pretty_print_cards(cards_list):
    return ', '.join([pretty_print_card(card) for card in cards_list])

def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

def int_to_card(val):
    rank_value, suit_value = divmod(val, 4)
    rank = [k for k, v in RANKS.items() if v == rank_value][0]
    suit = [k for k, v in SUITS.items() if v == suit_value][0]
    return (rank, suit)

def is_straight(cards):
    values = sorted([card[0] for card in cards])
    if set([Rank.DEUCE, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.ACE]).issubset(values):
        return True
    for i in range(len(values) - 4):
        if RANKS[values[i+4]] - RANKS[values[i]] == 4:
            return True
    return False

def is_flush(cards):
    suits = [card[1] for card in cards]
    return any(suits.count(suit) >= 5 for suit in SUITS)

def card_to_int(card):
    rank_value = RANKS[card.rank]
    suit_value = SUITS[card.suit]
    return rank_value * 4 + suit_value

def cards_to_ints(cards):
    return [card_to_int(card) for card in cards]


def ints_to_cards(ints):
    return [Card(int_to_card(val)) for val in ints]

def to_treys(cards):
    treys_cards = []
    for card in cards:
        rank_value = (card % 13) + 2
        suit_value = card // 13
        rank_str = str(rank_value) if rank_value < 10 else chr(ord('A') + rank_value - 10)
        suit_str = 'cdhs'[suit_value]
        treys_card_str = f'{rank_str}{suit_str}'
        treys_cards.append(treys_card_str)
    return [treys.Card.new(card_str) for card_str in treys_cards]

def from_treys(cards):
    return [card_to_int(Card(rank=Rank(cards[i].get_rank()), suit=Suit(cards[i].get_suit()))) for i in range(len(cards))]
