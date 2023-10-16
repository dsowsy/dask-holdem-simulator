from HandCalculations import *

# Example for HandStrength
ourcards_hs = torch.tensor([0, 13], device=device)  # 2 of Clubs and 2 of Diamonds
boardcards_hs = torch.tensor([26, 27, 28, 29, 30], device=device)  # 2 to 6 of Hearts

hs_result = HandStrength(ourcards_hs, boardcards_hs)
print(hs_result)

# Example for HandPotential
ourcards_hp = torch.tensor([14, 1], device=device)  # 3 of Diamonds and 3 of Clubs
boardcards_hp = torch.tensor([31, 32, 33], device=device)  # 7 to 9 of Hearts

hp_result = HandPotential(ourcards_hp, boardcards_hp)

print(f"Hand Strength Result: {hs_result}")
print(f"Hand Potential Results (Ppot, Npot): {hp_result}")

## Expected output:
# Hand Strength Result: 0.7582
# Hand Potential Results (Ppot, Npot): [0.5134, 0.1245]