"""
Advanced AI Decision Engine for Texas Hold'em Poker
Implements position-aware play, pot odds calculation, range-based strategies, and bluff detection.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math
import random
from collections import defaultdict

from utils import PlayerAction, PlayerPosition
from HandCalculations import HandStrength, HandPotential, CalculateHandRankValue
from poker import Card, Rank, Suit

class Position(Enum):
    """Player positions in poker"""
    UTG = "UTG"  # Under the Gun
    MP = "MP"    # Middle Position
    CO = "CO"    # Cutoff
    BTN = "BTN"  # Button
    SB = "SB"    # Small Blind
    BB = "BB"    # Big Blind

class BettingAction(Enum):
    """Betting actions with amounts"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

@dataclass
class PotOdds:
    """Pot odds calculation"""
    pot_size: int
    call_amount: int
    odds_to_call: float
    implied_odds: float
    break_even_percentage: float

@dataclass
class HandRange:
    """Hand range for position-based play"""
    position: Position
    hands: List[str]  # List of hand combinations like "AKs", "TT+", etc.
    frequency: float   # How often to play this range (0.0 to 1.0)

@dataclass
class OpponentProfile:
    """Opponent behavior profile"""
    player_id: str
    vpip: float  # Voluntarily Put Money In Pot percentage
    pfr: float   # Pre-Flop Raise percentage
    af: float    # Aggression Factor
    hands_played: int
    avg_bet_size: float
    fold_to_cbet: float
    three_bet_frequency: float
    bluff_frequency: float

@dataclass
class GameContext:
    """Current game context for decision making"""
    position: Position
    stack_size: int
    pot_size: int
    current_bet: int
    min_raise: int
    num_players: int
    active_players: int
    street: str  # preflop, flop, turn, river
    board_cards: List[Card]
    player_cards: List[Card]
    opponent_profiles: Dict[str, OpponentProfile]
    action_history: List[Dict]

class AdvancedDecisionEngine:
    """
    Advanced AI decision engine with position-aware play, pot odds calculation,
    range-based strategies, and bluff detection.
    """
    
    def __init__(self):
        self.position_ranges = self._initialize_position_ranges()
        self.opponent_profiles = {}
        self.hand_history = []
        self.session_stats = defaultdict(int)
        
    def _initialize_position_ranges(self) -> Dict[Position, HandRange]:
        """Initialize hand ranges for different positions"""
        ranges = {}
        
        # UTG - Tight range
        ranges[Position.UTG] = HandRange(
            position=Position.UTG,
            hands=["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs"],
            frequency=0.12
        )
        
        # MP - Slightly wider
        ranges[Position.MP] = HandRange(
            position=Position.MP,
            hands=["AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "KQs"],
            frequency=0.18
        )
        
        # CO - Even wider
        ranges[Position.CO] = HandRange(
            position=Position.CO,
            hands=["AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "A9s", "KQs", "KJs", "QJs"],
            frequency=0.25
        )
        
        # BTN - Wide range
        ranges[Position.BTN] = HandRange(
            position=Position.BTN,
            hands=["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "A9s", "A8s", "A7s", "KQs", "KJs", "KTs", "QJs", "QTs", "JTs"],
            frequency=0.35
        )
        
        # SB - Very wide
        ranges[Position.SB] = HandRange(
            position=Position.SB,
            hands=["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s"],
            frequency=0.45
        )
        
        # BB - Defending wide
        ranges[Position.BB] = HandRange(
            position=Position.BB,
            hands=["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s", "KQs", "KJs", "KTs", "K9s", "K8s", "QJs", "QTs", "Q9s", "Q8s", "JTs", "J9s", "J8s", "T9s", "T8s", "98s"],
            frequency=0.60
        )
        
        return ranges
    
    def calculate_pot_odds(self, pot_size: int, call_amount: int, 
                          implied_odds_multiplier: float = 1.0) -> PotOdds:
        """
        Calculate pot odds and implied odds
        
        Args:
            pot_size: Current pot size
            call_amount: Amount needed to call
            implied_odds_multiplier: Multiplier for implied odds calculation
            
        Returns:
            PotOdds object with calculated values
        """
        if call_amount == 0:
            return PotOdds(pot_size, 0, float('inf'), float('inf'), 0.0)
        
        odds_to_call = pot_size / call_amount
        implied_odds = (pot_size * implied_odds_multiplier) / call_amount
        break_even_percentage = call_amount / (pot_size + call_amount)
        
        return PotOdds(
            pot_size=pot_size,
            call_amount=call_amount,
            odds_to_call=odds_to_call,
            implied_odds=implied_odds,
            break_even_percentage=break_even_percentage
        )
    
    def calculate_hand_strength(self, player_cards: List[Card], 
                               board_cards: List[Card]) -> Tuple[float, float]:
        """
        Calculate hand strength and potential
        
        Args:
            player_cards: Player's hole cards
            board_cards: Community cards
            
        Returns:
            Tuple of (hand_strength, hand_potential)
        """
        if len(player_cards) < 2:
            return 0.0, 0.0
        
        # Convert cards to integer representation for HandCalculations
        player_cards_int = [self._card_to_int(card) for card in player_cards]
        board_cards_int = [self._card_to_int(card) for card in board_cards]
        
        # Convert to tensors for GPU calculation
        player_tensor = torch.tensor(player_cards_int, dtype=torch.long)
        board_tensor = torch.tensor(board_cards_int, dtype=torch.long)
        
        try:
            hand_strength = HandStrength(player_tensor, board_tensor)
            hand_potential = HandPotential(player_tensor, board_tensor)
            return hand_strength, hand_potential
        except Exception as e:
            # Fallback to simpler calculation
            return self._simple_hand_strength(player_cards, board_cards)
    
    def _card_to_int(self, card: Card) -> int:
        """Convert poker Card to integer representation"""
        rank_value = card.rank.value - 2  # 2=0, 3=1, ..., A=12
        suit_value = card.suit.value
        return rank_value + (suit_value * 13)
    
    def _simple_hand_strength(self, player_cards: List[Card], 
                              board_cards: List[Card]) -> Tuple[float, float]:
        """Simple hand strength calculation as fallback"""
        if len(player_cards) < 2:
            return 0.0, 0.0
        
        # Basic hand strength based on card ranks
        ranks = [card.rank.value for card in player_cards]
        suits = [card.suit.value for card in player_cards]
        
        # High card strength
        high_card = max(ranks) / 14.0
        
        # Paired strength
        paired = len(set(ranks)) == 1
        pair_strength = 0.5 if paired else 0.0
        
        # Suited strength
        suited = len(set(suits)) == 1
        suited_strength = 0.2 if suited else 0.0
        
        # Connected strength
        connected = abs(ranks[0] - ranks[1]) <= 2
        connected_strength = 0.1 if connected else 0.0
        
        total_strength = high_card + pair_strength + suited_strength + connected_strength
        return min(total_strength, 1.0), total_strength * 0.8  # Potential slightly lower
    
    def is_hand_in_range(self, player_cards: List[Card], position: Position) -> bool:
        """
        Check if current hand is in the range for the given position
        
        Args:
            player_cards: Player's hole cards
            position: Current position
            
        Returns:
            True if hand is in range, False otherwise
        """
        if len(player_cards) < 2:
            return False
        
        hand_str = self._cards_to_hand_string(player_cards)
        range_obj = self.position_ranges.get(position)
        
        if not range_obj:
            return False
        
        # Check if hand matches any pattern in the range
        for pattern in range_obj.hands:
            if self._hand_matches_pattern(hand_str, pattern):
                return True
        
        return False
    
    def _cards_to_hand_string(self, cards: List[Card]) -> str:
        """Convert cards to hand string like 'AKs' or 'TT'"""
        if len(cards) < 2:
            return ""
        
        rank1, rank2 = cards[0].rank.value, cards[1].rank.value
        suit1, suit2 = cards[0].suit.value, cards[1].suit.value
        
        # Convert ranks to string representation
        rank_chars = "23456789TJQKA"
        rank1_char = rank_chars[rank1 - 2]  # 2=0, 3=1, ..., A=12
        rank2_char = rank_chars[rank2 - 2]
        
        # Determine if suited
        suited = suit1 == suit2
        
        # Order ranks (higher first)
        if rank1 > rank2:
            high_rank, low_rank = rank1_char, rank2_char
        else:
            high_rank, low_rank = rank2_char, rank1_char
        
        if high_rank == low_rank:
            return f"{high_rank}{low_rank}"  # e.g., "TT"
        else:
            suffix = "s" if suited else "o"
            return f"{high_rank}{low_rank}{suffix}"  # e.g., "AKs" or "AKo"
    
    def _hand_matches_pattern(self, hand_str: str, pattern: str) -> bool:
        """Check if hand string matches a pattern like 'AKs' or 'TT+'"""
        if not hand_str or not pattern:
            return False
        
        # Handle specific patterns
        if pattern.endswith("+"):
            # Range like "TT+" means TT or higher pairs
            base_rank = pattern[0]
            return hand_str.startswith(base_rank) and len(hand_str) == 2
        
        # Handle specific hand patterns
        if pattern.endswith("s") or pattern.endswith("o"):
            # Specific hand like "AKs" or "AKo"
            return hand_str == pattern
        
        # Handle pairs
        if len(pattern) == 2 and pattern[0] == pattern[1]:
            return hand_str == pattern
        
        return hand_str == pattern
    
    def detect_bluff_indicators(self, opponent_id: str, 
                               action_history: List[Dict]) -> Dict[str, float]:
        """
        Detect potential bluff indicators from opponent behavior
        
        Args:
            opponent_id: ID of the opponent to analyze
            action_history: History of actions in the current hand
            
        Returns:
            Dictionary of bluff indicators and their confidence scores
        """
        indicators = {
            'timing_tell': 0.0,
            'bet_sizing_tell': 0.0,
            'action_pattern': 0.0,
            'over_betting': 0.0,
            'inconsistent_play': 0.0
        }
        
        if not action_history:
            return indicators
        
        # Analyze timing patterns
        timing_scores = []
        for i, action in enumerate(action_history):
            if action.get('player') == opponent_id and 'timing' in action:
                timing_scores.append(action['timing'])
        
        if timing_scores:
            avg_timing = np.mean(timing_scores)
            # Very fast or very slow actions might indicate bluffing
            if avg_timing < 1.0 or avg_timing > 10.0:
                indicators['timing_tell'] = 0.7
        
        # Analyze bet sizing
        bet_sizes = []
        for action in action_history:
            if action.get('player') == opponent_id and 'amount' in action:
                bet_sizes.append(action['amount'])
        
        if bet_sizes:
            avg_bet = np.mean(bet_sizes)
            pot_size = action_history[-1].get('pot_size', 100)
            bet_pot_ratio = avg_bet / pot_size if pot_size > 0 else 0
            
            # Over-betting might indicate bluffing
            if bet_pot_ratio > 0.8:
                indicators['over_betting'] = 0.8
            elif bet_pot_ratio < 0.2:
                indicators['bet_sizing_tell'] = 0.6
        
        # Analyze action patterns
        opponent_actions = [a for a in action_history if a.get('player') == opponent_id]
        if len(opponent_actions) >= 2:
            # Check for inconsistent play
            action_types = [a.get('action') for a in opponent_actions]
            if len(set(action_types)) > 2:  # Many different action types
                indicators['inconsistent_play'] = 0.6
        
        return indicators
    
    def make_decision(self, context: GameContext) -> BettingAction:
        """
        Make a decision based on current game context
        
        Args:
            context: Current game context
            
        Returns:
            BettingAction with recommended action
        """
        # Calculate hand strength
        hand_strength, hand_potential = self.calculate_hand_strength(
            context.player_cards, context.board_cards
        )
        
        # Calculate pot odds
        pot_odds = self.calculate_pot_odds(
            context.pot_size, context.current_bet
        )
        
        # Check if hand is in range for position
        in_range = self.is_hand_in_range(context.player_cards, context.position)
        
        # Detect bluff indicators
        bluff_indicators = self.detect_bluff_indicators(
            "opponent", context.action_history
        )
        
        # Make decision based on all factors
        decision = self._evaluate_decision_factors(
            hand_strength, hand_potential, pot_odds, in_range, 
            bluff_indicators, context
        )
        
        return decision
    
    def _evaluate_decision_factors(self, hand_strength: float, hand_potential: float,
                                  pot_odds: PotOdds, in_range: bool,
                                  bluff_indicators: Dict[str, float],
                                  context: GameContext) -> BettingAction:
        """
        Evaluate all factors to make a decision
        
        Args:
            hand_strength: Current hand strength (0.0 to 1.0)
            hand_potential: Hand potential (0.0 to 1.0)
            pot_odds: Calculated pot odds
            in_range: Whether hand is in position range
            bluff_indicators: Bluff detection indicators
            context: Game context
            
        Returns:
            Recommended betting action
        """
        # Base decision on hand strength
        if hand_strength < 0.2:
            return BettingAction.FOLD
        
        # Consider pot odds
        if pot_odds.break_even_percentage > 0.5 and hand_strength < 0.4:
            return BettingAction.FOLD
        
        # Consider position and range
        if not in_range and context.position in [Position.UTG, Position.MP]:
            if hand_strength < 0.6:
                return BettingAction.FOLD
        
        # Consider bluff indicators
        bluff_confidence = sum(bluff_indicators.values()) / len(bluff_indicators)
        if bluff_confidence > 0.5 and hand_strength > 0.3:
            # Opponent might be bluffing, call with mediocre hands
            return BettingAction.CALL
        
        # Strong hand - bet or raise
        if hand_strength > 0.7:
            if context.current_bet == 0:
                return BettingAction.BET
            else:
                return BettingAction.RAISE
        
        # Medium hand - call or check
        if hand_strength > 0.4:
            if context.current_bet == 0:
                return BettingAction.CHECK
            else:
                return BettingAction.CALL
        
        # Weak hand - fold
        return BettingAction.FOLD
    
    def update_opponent_profile(self, opponent_id: str, action: Dict):
        """
        Update opponent profile based on their action
        
        Args:
            opponent_id: ID of the opponent
            action: Action taken by opponent
        """
        if opponent_id not in self.opponent_profiles:
            self.opponent_profiles[opponent_id] = OpponentProfile(
                player_id=opponent_id,
                vpip=0.0,
                pfr=0.0,
                af=0.0,
                hands_played=0,
                avg_bet_size=0.0,
                fold_to_cbet=0.0,
                three_bet_frequency=0.0,
                bluff_frequency=0.0
            )
        
        profile = self.opponent_profiles[opponent_id]
        
        # Update basic stats
        if action.get('action') in ['bet', 'raise', 'call']:
            profile.vpip += 1
        
        if action.get('action') == 'raise':
            profile.pfr += 1
        
        # Update aggression factor
        aggressive_actions = ['bet', 'raise']
        passive_actions = ['call', 'check']
        
        if action.get('action') in aggressive_actions:
            profile.af += 1
        elif action.get('action') in passive_actions:
            profile.af = max(0, profile.af - 0.5)
        
        profile.hands_played += 1
        
        # Normalize percentages
        if profile.hands_played > 0:
            profile.vpip = profile.vpip / profile.hands_played
            profile.pfr = profile.pfr / profile.hands_played
            profile.af = profile.af / profile.hands_played
    
    def get_session_stats(self) -> Dict[str, int]:
        """Get current session statistics"""
        return dict(self.session_stats)
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.session_stats.clear()

# Factory functions for different bot personalities
def create_tight_aggressive_bot() -> AdvancedDecisionEngine:
    """Create a tight-aggressive bot"""
    bot = AdvancedDecisionEngine()
    # Modify ranges to be tighter
    for position in bot.position_ranges:
        bot.position_ranges[position].frequency *= 0.7
    return bot

def create_loose_aggressive_bot() -> AdvancedDecisionEngine:
    """Create a loose-aggressive bot"""
    bot = AdvancedDecisionEngine()
    # Modify ranges to be wider
    for position in bot.position_ranges:
        bot.position_ranges[position].frequency *= 1.3
    return bot

def create_tight_passive_bot() -> AdvancedDecisionEngine:
    """Create a tight-passive bot"""
    bot = AdvancedDecisionEngine()
    # Modify ranges to be tighter and more passive
    for position in bot.position_ranges:
        bot.position_ranges[position].frequency *= 0.6
    return bot