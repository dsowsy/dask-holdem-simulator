import torch
import random

from utils import * # type: ignore
from treys import Evaluator, Card
from pokerlib_to_treys import convert_to_treys

from collections import Counter
from pokerlib_to_treys import *


#
# The following code is implemented from the Effective Hand Strength
# algorithm: https://en.wikipedia.org/wiki/Effective_hand_strength_algorithm
#
# strength = HandStrength([your_cards], [board_cards])
# hand_potential = HandPotential([your_cards], [board_cards]):
#
#

# Pre-flight check to determine if CUDA is available,and CPU if not
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def is_straight(values):
    # Ensure there are no duplicates
    if len(values) != len(set(values)):
        return False

    sorted_values = sorted(values)
    # Extend the list for the wrap-around case
    extended_values = sorted_values + [v + 13 for v in sorted_values]

    for i in range(len(extended_values) - 4):
        if extended_values[i:i+5] == list(range(extended_values[i], extended_values[i] + 5)):
            return True

    return False

def determine_flush(suits, suit_counts):
    for suit, count in suit_counts.items():
        if count >= 5:
            return True, suit
    return False, None

def CalculateHandRankValue(ourcards, boardcards):
    if isinstance(ourcards, torch.Tensor):
        ourcards = ourcards.tolist()
    if isinstance(boardcards, torch.Tensor):
        boardcards = boardcards.tolist()

    cards = ourcards + boardcards

    values = [card % 13 for card in cards]
    suits = [card // 13 for card in cards]

    suit_counts = Counter(suits)
    value_counts = Counter(values)

    max_value_count = max(value_counts.values())

    flush, most_common_suit = determine_flush(suits, suit_counts)
    if flush:
        flush_cards = [val for idx, val in enumerate(values) if suits[idx] == most_common_suit]
        if is_straight(flush_cards):
            return 8  # Straight flush

    # Other hand checks
    if max_value_count == 4:
        return 7  # Four of a kind
    if max_value_count == 3 and 2 in value_counts.values():
        return 6  # Full house
    if is_straight(values):
        return 4  # Straight
    if flush:
        return 5  # Flush
    if max_value_count == 3:
        return 3  # Three of a kind
    if max_value_count == 2 and list(value_counts.values()).count(2) == 2:
        return 2  # Two pair
    if max_value_count == 2:
        return 1  # One pair
    return 0  # High card



def CalculateHandRank(ourcards, boardcards):
    if is_tensor(ourcards):
        ourcards = ourcards.tolist()
    if is_tensor(boardcards):
        boardcards = boardcards.tolist()

    all_cards = ourcards + boardcards
    all_cards = [int_to_card(val) for val in all_cards]
    all_hands = list(combinations(all_cards, 5))
    best_rank = (0, )

    for hand in all_hands:
        hand_ranks = [RANKS[card[0]] for card in hand]
        unique_ranks = set(hand_ranks)
        rank_counts = [hand_ranks.count(rank) for rank in unique_ranks]

        if is_straight(hand) and is_flush(hand):
            if set([Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]).issubset({card[0] for card in hand}):
                return (10,)
            best_rank = max(best_rank, (9, max(hand_ranks)))
        elif 4 in rank_counts:
            quad_rank = [rank for rank in unique_ranks if hand_ranks.count(rank) == 4][0]
            best_rank = max(best_rank, (8, quad_rank))
        elif 3 in rank_counts and 2 in rank_counts:
            three_rank = [rank for rank in unique_ranks if hand_ranks.count(rank) == 3][0]
            pair_rank = [rank for rank in unique_ranks if hand_ranks.count(rank) == 2][0]
            best_rank = max(best_rank, (7, three_rank, pair_rank))
        elif is_flush(hand):
            best_rank = max(best_rank, (6, max(hand_ranks)))
        elif is_straight(hand):
            best_rank = max(best_rank, (5, max(hand_ranks)))
        elif 3 in rank_counts:
            three_rank = [rank for rank in unique_ranks if hand_ranks.count(rank) == 3][0]
            best_rank = max(best_rank, (4, three_rank))
        elif rank_counts.count(2) == 2:
            pairs = sorted([rank for rank in unique_ranks if hand_ranks.count(rank) == 2], reverse=True)
            best_rank = max(best_rank, (3, pairs[0], pairs[1]))
        elif 2 in rank_counts:
            pair_rank = [rank for rank in unique_ranks if hand_ranks.count(rank) == 2][0]
            best_rank = max(best_rank, (2, pair_rank))
        else:
            best_rank = max(best_rank, (1, max(hand_ranks)))

    return best_rank

def remove_cards_from_deck(cards_to_remove):
    """Helper function to remove given cards from a full deck."""
    full_deck = [i for i in range(52)]
    for card in cards_to_remove:
        if card not in full_deck:
            print(f"Trying to remove card: {card} which is not in full_deck")
        else:
            full_deck.remove(card)
    return full_deck

def generate_combinations(cards):
    """Generate all 2-card combinations from the given cards."""
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            yield torch.tensor([cards[i], cards[j]], device=device)

# Yields hands in form of tensors
def generate_opponent_hands(ourcards, boardcards):
    cards_left = remove_cards_from_deck(torch.cat((ourcards, boardcards)))
    return generate_combinations(cards_left)

def generate_possible_turns_and_rivers(boardcards):
    cards_left = remove_cards_from_deck(boardcards)
    return generate_combinations(cards_left)


def HandStrength(ourcards, boardcards):
    # Ensure inputs are torch tensors on the appropriate device
    ourcards = ourcards.clone().detach().to(device)
    boardcards = boardcards.clone().detach().to(device)

    ahead, tied, behind = 0, 0, 0
    ourrank = CalculateHandRankValue(ourcards, boardcards)

    # Consider all two-card combinations of the remaining cards for the opponent
    for oppcards in generate_opponent_hands(ourcards, boardcards):
        opprank = CalculateHandRankValue(oppcards, boardcards)
        if ourrank > opprank:
            ahead += 1
        elif ourrank == opprank:
            tied += 1
        else:
            behind += 1

    handstrength = (ahead + tied / 2) / (ahead + tied + behind)
    return handstrength

def HandPotential(ourcards, boardcards):
    # Ensure inputs are torch tensors on GPU
    ourcards = ourcards.clone().detach().to(device)
    boardcards = boardcards.clone().detach().to(device)

    # Define the state space
    ahead, tied, behind = 0, 1, 2

    # Initialization
    HP = torch.zeros((3, 3), device=device)
    HPTotal = torch.zeros(3, device=device)

    ourrank = CalculateHandRankValue(ourcards, boardcards)

    # Consider all two card combinations of the remaining cards for the opponent
    for oppcards in generate_opponent_hands(ourcards, boardcards):  # assuming
        opprank = CalculateHandRankValue(oppcards, boardcards)

        if ourrank > opprank:
            index = ahead
        elif ourrank == opprank:
            index = tied
        else:
            index = behind

        HPTotal[index] += 1

        # All possible board cards to come

        for combination in generate_possible_turns_and_rivers(boardcards):
            turn, river = combination[0], combination[1]
            updated_board = torch.cat((boardcards, torch.tensor([turn, river], device=device)))
            ourbest = CalculateHandRankValue(ourcards, updated_board)
            oppbest = CalculateHandRankValue(oppcards, updated_board)

            if ourbest > oppbest:
                HP[index, ahead] += 1
            elif ourbest == oppbest:
                HP[index, tied] += 1
            else:
                HP[index, behind] += 1

    # Ppot and Npot computations
    Ppot = (HP[behind, ahead] + HP[behind, tied] / 2 +
            HP[tied, ahead] / 2) / (HPTotal[behind] + HPTotal[tied])
    Npot = (HP[ahead, behind] + HP[tied, behind] / 2 +
            HP[ahead, tied] / 2) / (HPTotal[ahead] + HPTotal[tied])

    return [Ppot, Npot]


def evaluate_best_hand(player_cards, board_hand):
    # Use the actual Treys evaluator instead of random values
    evaluator = Evaluator()
    
    # Convert cards to treys format
    treys_player_cards = convert_to_treys(player_cards)
    treys_board_hand = convert_to_treys(board_hand)
    
    # Evaluate the hand strength
    hand_strength = evaluator.evaluate(treys_board_hand, treys_player_cards)
    # Return the hand strength (lower is better in Treys, so we return as-is)
    return hand_strength

# Example usage:
# ourcards = [0, 1]
# boardcards = [10, 11, 12, 13, 14]

# rank = CalculateHandRankValue(ourcards, boardcards)
# print(rank)  # This will print the hand rank


#ourcards = torch.tensor([0, 1])
#boardcards = torch.tensor([10, 11, 12, 13, 14])
#rank = CalculateHandRankValue(ourcards, boardcards)
#print(rank)  # This should print 8


#ourcards = torch.tensor([0, 13, 26, 39, 12])
#boardcards = torch.tensor( [26, 39, 12, 13, 14])
#rank = CalculateHandRankValue(ourcards, boardcards)
#print(rank)  # This should print 7

#ourcards = torch.tensor([0, 13, 26, 12, 14])
#boardcards = torch.tensor( [26, 39, 12, 14])
#rank = CalculateHandRankValue(ourcards, boardcards)
#print(rank)  # This should print 6



evaluator = Evaluator()

# def evaluate_best_hand(player_cards, board_hand):
#     evaluator = Evaluator()

#     # Convert cards to treys format
#     treys_player_cards = convert_to_treys(player_cards)
#     treys_board_hand = convert_to_treys(board_hand)

#     # Evaluate the hand strength
#     hand_strength = evaluator.evaluate(treys_board_hand, treys_player_cards)
#     # Return the hand strength (you can adjust this to be normalized if desired)
#     return hand_strength



