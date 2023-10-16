
# Prototyping idea -- Not used

def decision_based_on_blind_position(hand_strength, ppot, npot, position):
    # Position can be: 'SB' (small blind), 'BB' (big blind), or 'NONE'

    # Ensure valid position input
    if position not in ['SB', 'BB', 'NONE']:
        raise ValueError("Invalid position. Expected 'SB', 'BB', or 'NONE'.")

    # High hand strength
    if hand_strength > 0.8:
        if npot < 0.3:
            if position == 'SB':
                return "check"
            elif position == 'BB':
                return "call"
            else:  # 'NONE'
                return "raise"

    # Moderate hand strength
    elif hand_strength > 0.5:
        if ppot > 0.6:
            if position == 'SB':
                return "check"
            elif position == 'BB':
                return "call"
            else:  # 'NONE'
                return "call"
        else:
            if position == 'SB':
                return "fold"
            elif position == 'BB':
                return "check"
            else:  # 'NONE'
                return "fold"

    # Low hand strength
    else:
        if position == 'SB':
            return "fold"
        elif position == 'BB':
            if ppot > 0.8:
                return "call"
            else:
                return "check"
        else:  # 'NONE'
            return "fold"
