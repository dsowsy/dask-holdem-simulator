import unittest
import torch
from timeout_decorator import timeout

# Assuming device to be used is the default one, you can update it as needed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing functions from HandCalculations.py
from HandCalculations import *

class TestHandCalculations(unittest.TestCase):

    @timeout(2)  # 2 seconds timeout
    def test_CalculateHandRank(self):

         # Straight flush
        self.assertEqual(CalculateHandRankValue(torch.tensor([0, 1]), torch.tensor([10, 11, 12, 13, 14])), 8)
        self.assertEqual(CalculateHandRankValue([0, 1],[10, 11, 12, 13, 14]), 8)

        # Four of a kind
        # self.assertEqual(CalculateHandRankValue([0, 13, 26, 39, 12], [26, 39, 12, 13, 14]), 7)

        # Full house
        # self.assertEqual(CalculateHandRankValue([0, 13, 26, 12, 14], [26, 39, 12, 14]), 6)

    @timeout(2)  # 2 seconds timeout
    def test_remove_cards_from_deck(self):
        self.assertEqual(len(remove_cards_from_deck(torch.tensor([0, 1, 2], device=device))), 49)
        self.assertEqual(len(remove_cards_from_deck(torch.tensor([52, 53], device=device))), 52)  # Non-existent cards should be ignored

    @timeout(2)  # 2 seconds timeout
    def test_generate_combinations(self):
        combinations = list(generate_combinations(torch.tensor([0, 1, 2], device=device)))
        self.assertEqual(len(combinations), 3)
        # self.assertIn(torch.tensor([0, 1], device=device), combinations)

    @timeout(2)  # 2 seconds timeout
    def test_generate_opponent_hands(self):
        hands = list(generate_opponent_hands(torch.tensor([0, 1], device=device), torch.tensor([2, 3], device=device)))
        self.assertEqual(len(hands), 1128)  # Combinatorial math: 46 choose 2

    @timeout(2)  # 2 seconds timeout
    def test_generate_possible_turns_and_rivers(self):
        combinations = list(generate_possible_turns_and_rivers(torch.tensor([0, 1, 2], device=device)))
        self.assertEqual(len(combinations), 1176)  # 46 choose 2

    @timeout(2)  # 2 seconds timeout
    def test_HandStrength(self):
        strength = HandStrength(torch.tensor([0, 1], device=device), torch.tensor([2, 3, 4], device=device))
        # Validate the strength is between 0 and 1 (inclusive)
        self.assertGreaterEqual(strength, 0)
        self.assertLessEqual(strength, 1)

    # @timeout(2)  # 2 seconds timeout
    # def test_HandPotential(self):
    #     Ppot, Npot = HandPotential(torch.tensor([0, 1], device=device).clone().detach(), torch.tensor([2, 3, 4], device=device))
    #     # Validate potentials are between 0 and 1 (inclusive)
    #     self.assertGreaterEqual(Ppot.item(), 0)
    #     self.assertLessEqual(Ppot.item(), 1)
    #     self.assertGreaterEqual(Npot.item(), 0)
    #     self.assertLessEqual(Npot.item(), 1)

if __name__ == "__main__":
    unittest.main()
