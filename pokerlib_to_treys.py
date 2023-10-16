
import torch
from poker import Card
from treys import Card as TCard
from treys import Evaluator

# Returns a treys evaluation of the pokerlib board and hand cards
# as a normalized score from [0,1]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def convert_to_treys(cards):
    return [TCard.new(card.rank.val + card.suit.value[1]) for card in cards]

def CalculateHandValue(pokerlib_board, pokerlib_hand):
    # Convert to treys cards
    converted_board = convert_to_treys(pokerlib_board)
    converted_hand = convert_to_treys(pokerlib_hand)
    evaluator = Evaluator()
    hand_strength = evaluator.evaluate(converted_board, converted_hand)
    return  1 - hand_strength / 7462

# Evaluate treys example
# board = [TCard.new('Ah'), TCard.new('Kd'), TCard.new('Jc')]
# hand = [ TCard.new('Qs'), TCard.new('Th')]

# evaluator = Evaluator()
# print(evaluator.evaluate(board, hand))

# pokerlib_board = [Card('As'), Card('Kd'), Card('Jc')]
# pokerlib_hand = [Card('Qs'), Card('Th')]

# hand_strength = CalculateHandValue(pokerlib_board, pokerlib_hand)
# pclass = evaluator.get_rank_class(hand_strength)
# print(f"Score: {hand_strength}")

# Hand strength is valued on a scale of 1 to 7462,
# where 1 is a Royal Flush and 7462 is unsuited 7-5-4-3-2,
# as there are only 7642 distinctly ranked hands in poker.

# Example usage
if __name__ == "__main__":
    # Create example input data (lists of pokerlib cards)
    pokerlib_board = [Card('As'), Card('Kd'), Card('Jc')]
    pokerlib_hand = [Card('Qs'), Card('Th')]

    # Specify the device (GPU) || CPU
    device = torch.device(device)

    # Calculate the hand value
    #score = CalculateHandValue(pokerlib_board, pokerlib_hand, device)
    score = CalculateHandValue(pokerlib_board, pokerlib_hand)
    print(score)


