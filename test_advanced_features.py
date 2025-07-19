"""
Comprehensive test suite for advanced poker simulator features
Tests AdvancedDecisionEngine, StatisticsAnalytics, and MultiTableManager
"""

import pytest
import asyncio
import tempfile
import os
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from AdvancedDecisionEngine import (
    AdvancedDecisionEngine, Position, BettingAction, PotOdds, 
    HandRange, OpponentProfile, GameContext, create_tight_aggressive_bot,
    create_loose_aggressive_bot, create_tight_passive_bot
)

from StatisticsAnalytics import (
    StatisticsAnalytics, HandResult, SessionStats, PlayerProfile
)

from MultiTableManager import (
    MultiTableManager, TableState, TableType, TableInfo, 
    PlayerTableAssignment, TableSelectionCriteria, CrossTableLearning,
    create_cash_game_table, create_tournament_table
)

from poker import Card, Rank, Suit
from utils import PlayerAction, PlayerPosition

class TestAdvancedDecisionEngine:
    """Test the Advanced AI Decision Engine"""
    
    def setup_method(self):
        """Set up test environment"""
        self.engine = AdvancedDecisionEngine()
        
        # Create test cards
        self.ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        self.king_spades = Card(Rank.KING, Suit.SPADES)
        self.queen_diamonds = Card(Rank.QUEEN, Suit.DIAMONDS)
        self.jack_clubs = Card(Rank.JACK, Suit.CLUBS)
        self.ten_hearts = Card(Rank.TEN, Suit.HEARTS)
        
    def test_engine_initialization(self):
        """Test that the decision engine initializes correctly"""
        assert self.engine is not None
        assert hasattr(self.engine, 'position_ranges')
        assert hasattr(self.engine, 'opponent_profiles')
        assert hasattr(self.engine, 'hand_history')
        
        # Check that position ranges are initialized
        assert len(self.engine.position_ranges) == 6  # 6 positions
        assert Position.UTG in self.engine.position_ranges
        assert Position.BB in self.engine.position_ranges
    
    def test_pot_odds_calculation(self):
        """Test pot odds calculation"""
        # Test basic pot odds
        pot_odds = self.engine.calculate_pot_odds(100, 20)
        assert pot_odds.pot_size == 100
        assert pot_odds.call_amount == 20
        assert pot_odds.odds_to_call == 5.0
        assert pot_odds.break_even_percentage == 0.167  # 20/(100+20)
        
        # Test zero call amount
        pot_odds = self.engine.calculate_pot_odds(100, 0)
        assert pot_odds.odds_to_call == float('inf')
        assert pot_odds.break_even_percentage == 0.0
    
    def test_hand_strength_calculation(self):
        """Test hand strength calculation"""
        player_cards = [self.ace_hearts, self.king_spades]
        board_cards = [self.queen_diamonds, self.jack_clubs, self.ten_hearts]
        
        strength, potential = self.engine.calculate_hand_strength(player_cards, board_cards)
        
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= potential <= 1.0
        assert strength > 0.0  # Should have some strength with AK
    
    def test_hand_range_checking(self):
        """Test hand range checking"""
        # Test strong hand in UTG position
        strong_hand = [self.ace_hearts, self.king_spades]  # AKs
        in_range = self.engine.is_hand_in_range(strong_hand, Position.UTG)
        assert in_range == True
        
        # Test weak hand in UTG position
        weak_hand = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.CLUBS)]
        in_range = self.engine.is_hand_in_range(weak_hand, Position.UTG)
        assert in_range == False
        
        # Test same weak hand in BB position (should be in range)
        in_range = self.engine.is_hand_in_range(weak_hand, Position.BB)
        assert in_range == True
    
    def test_bluff_detection(self):
        """Test bluff detection functionality"""
        action_history = [
            {'player': 'opponent', 'action': 'raise', 'amount': 50, 'timing': 0.5},
            {'player': 'opponent', 'action': 'bet', 'amount': 100, 'timing': 8.0},
            {'player': 'opponent', 'action': 'fold', 'timing': 2.0}
        ]
        
        indicators = self.engine.detect_bluff_indicators('opponent', action_history)
        
        assert 'timing_tell' in indicators
        assert 'bet_sizing_tell' in indicators
        assert 'action_pattern' in indicators
        assert 'over_betting' in indicators
        assert 'inconsistent_play' in indicators
        
        # Check that indicators are reasonable values
        for indicator, value in indicators.items():
            assert 0.0 <= value <= 1.0
    
    def test_decision_making(self):
        """Test decision making with game context"""
        # Create game context
        context = GameContext(
            position=Position.BTN,
            stack_size=1000,
            pot_size=100,
            current_bet=20,
            min_raise=40,
            num_players=6,
            active_players=4,
            street='preflop',
            board_cards=[],
            player_cards=[self.ace_hearts, self.king_spades],
            opponent_profiles={},
            action_history=[]
        )
        
        decision = self.engine.make_decision(context)
        
        assert isinstance(decision, BettingAction)
        assert decision in [BettingAction.FOLD, BettingAction.CALL, 
                          BettingAction.RAISE, BettingAction.BET]
    
    def test_opponent_profile_updating(self):
        """Test opponent profile updating"""
        opponent_id = "player123"
        action = {'action': 'raise', 'amount': 50}
        
        self.engine.update_opponent_profile(opponent_id, action)
        
        assert opponent_id in self.engine.opponent_profiles
        profile = self.engine.opponent_profiles[opponent_id]
        assert profile.player_id == opponent_id
        assert profile.hands_played == 1
        assert profile.pfr == 1.0  # 100% PFR after one raise
    
    def test_bot_personalities(self):
        """Test different bot personalities"""
        # Test tight-aggressive bot
        tight_aggressive = create_tight_aggressive_bot()
        assert tight_aggressive is not None
        
        # Test loose-aggressive bot
        loose_aggressive = create_loose_aggressive_bot()
        assert loose_aggressive is not None
        
        # Test tight-passive bot
        tight_passive = create_tight_passive_bot()
        assert tight_passive is not None
        
        # Verify that ranges are modified appropriately
        for position in Position:
            tight_freq = tight_aggressive.position_ranges[position].frequency
            loose_freq = loose_aggressive.position_ranges[position].frequency
            assert tight_freq < loose_freq

class TestStatisticsAnalytics:
    """Test the Statistics and Analytics system"""
    
    def setup_method(self):
        """Set up test environment"""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.analytics = StatisticsAnalytics(self.temp_db.name)
        
        # Create test cards
        self.ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        self.king_spades = Card(Rank.KING, Suit.SPADES)
        self.queen_diamonds = Card(Rank.QUEEN, Suit.DIAMONDS)
    
    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        # Check that tables were created
        import sqlite3
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'hands' in tables
        assert 'sessions' in tables
        assert 'player_profiles' in tables
        
        conn.close()
    
    def test_session_management(self):
        """Test session start and end functionality"""
        player_id = "test_player"
        
        # Start session
        session_id = self.analytics.start_session(player_id)
        assert session_id is not None
        assert session_id.startswith(player_id)
        
        # End session
        session_stats = self.analytics.end_session()
        assert isinstance(session_stats, SessionStats)
        assert session_stats.session_id == session_id
        assert session_stats.total_hands == 0
        assert session_stats.hands_won == 0
    
    def test_hand_recording(self):
        """Test hand result recording"""
        # Start session
        session_id = self.analytics.start_session("test_player")
        
        # Create hand result
        hand_result = HandResult(
            hand_id="hand_123",
            timestamp=datetime.now(),
            player_id="test_player",
            position=PlayerPosition.BTN,
            hole_cards=[self.ace_hearts, self.king_spades],
            board_cards=[self.queen_diamonds],
            action_sequence=[{'action': 'raise', 'amount': 50}],
            final_action=PlayerAction.RAISE,
            pot_size=100,
            amount_won=50,
            hand_strength=0.8,
            hand_rank="preflop",
            session_id=session_id
        )
        
        # Record hand
        self.analytics.record_hand(hand_result)
        
        # End session and check stats
        session_stats = self.analytics.end_session()
        assert session_stats.total_hands == 1
        assert session_stats.total_winnings == 50
        assert session_stats.hands_won == 1
    
    def test_player_profile_creation(self):
        """Test player profile creation and retrieval"""
        # Create some test data
        session_id = self.analytics.start_session("test_player")
        
        # Record multiple hands
        for i in range(5):
            hand_result = HandResult(
                hand_id=f"hand_{i}",
                timestamp=datetime.now(),
                player_id="test_player",
                position=PlayerPosition.BTN,
                hole_cards=[self.ace_hearts, self.king_spades],
                board_cards=[],
                action_sequence=[{'action': 'raise', 'amount': 50}],
                final_action=PlayerAction.RAISE,
                pot_size=100,
                amount_won=25 if i % 2 == 0 else -25,
                hand_strength=0.8,
                hand_rank="preflop",
                session_id=session_id
            )
            self.analytics.record_hand(hand_result)
        
        self.analytics.end_session()
        
        # Get player profile
        profile = self.analytics.get_player_profile("test_player")
        assert profile is not None
        assert profile.player_id == "test_player"
        assert profile.total_sessions == 1
        assert profile.total_hands == 5
        assert profile.win_rate == 0.6  # 3 wins out of 5 hands
    
    def test_heat_map_generation(self):
        """Test heat map generation"""
        # Create test data
        session_id = self.analytics.start_session("test_player")
        
        # Record hands with different positions and streets
        positions = [PlayerPosition.UTG, PlayerPosition.BTN, PlayerPosition.BB]
        streets = ["preflop", "flop", "turn", "river"]
        
        for i, position in enumerate(positions):
            for j, street in enumerate(streets):
                hand_result = HandResult(
                    hand_id=f"hand_{i}_{j}",
                    timestamp=datetime.now(),
                    player_id="test_player",
                    position=position,
                    hole_cards=[self.ace_hearts, self.king_spades],
                    board_cards=[],
                    action_sequence=[{'action': 'call', 'amount': 10}],
                    final_action=PlayerAction.CALL,
                    pot_size=50,
                    amount_won=10,
                    hand_strength=0.7,
                    hand_rank=street,
                    session_id=session_id
                )
                self.analytics.record_hand(hand_result)
        
        self.analytics.end_session()
        
        # Generate heat map
        heat_map = self.analytics.generate_heat_map("test_player")
        assert heat_map.shape == (6, 4)  # 6 positions, 4 streets
        assert np.sum(heat_map) > 0  # Should have some data
    
    def test_hand_analysis(self):
        """Test detailed hand analysis"""
        # Create test hand
        session_id = self.analytics.start_session("test_player")
        
        hand_result = HandResult(
            hand_id="test_hand",
            timestamp=datetime.now(),
            player_id="test_player",
            position=PlayerPosition.BTN,
            hole_cards=[self.ace_hearts, self.king_spades],
            board_cards=[self.queen_diamonds],
            action_sequence=[
                {'action': 'raise', 'amount': 50},
                {'action': 'call', 'amount': 30},
                {'action': 'fold'}
            ],
            final_action=PlayerAction.FOLD,
            pot_size=100,
            amount_won=-50,
            hand_strength=0.6,
            hand_rank="flop",
            session_id=session_id
        )
        
        self.analytics.record_hand(hand_result)
        self.analytics.end_session()
        
        # Get hand analysis
        analysis = self.analytics.get_hand_analysis("test_hand")
        assert analysis is not None
        assert 'hand_data' in analysis
        assert 'action_frequency' in analysis
        assert 'betting_pattern' in analysis
        assert 'position_analysis' in analysis
        assert 'pot_odds_analysis' in analysis

class TestMultiTableManager:
    """Test the Multi-Table Management system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.manager = MultiTableManager(max_tables=5, max_players_per_table=6)
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        assert self.manager.max_tables == 5
        assert self.manager.max_players_per_table == 6
        assert len(self.manager.tables) == 0
        assert len(self.manager.player_assignments) == 0
        assert self.manager.running == False
    
    def test_table_creation(self):
        """Test table creation"""
        # Create cash game table
        table_config = create_cash_game_table(min_buy_in=20, max_buy_in=200)
        table_id = self.manager.create_table(table_config)
        
        assert table_id is not None
        assert table_id in self.manager.tables
        
        table_info = self.manager.tables[table_id]
        assert table_info.table_type == TableType.CASH_GAME
        assert table_info.state == TableState.WAITING
        assert table_info.num_players == 0
        assert table_info.max_players == 9
    
    def test_player_assignment(self):
        """Test player assignment to tables"""
        # Create table
        table_config = create_cash_game_table()
        table_id = self.manager.create_table(table_config)
        
        # Add player to table
        success = self.manager.add_player_to_table("player1", table_id, "aggressive")
        assert success == True
        
        # Check assignment
        assert "player1" in self.manager.player_assignments
        assignment = self.manager.player_assignments["player1"]
        assert assignment.table_id == table_id
        assert assignment.bot_personality == "aggressive"
        
        # Check table state
        table_info = self.manager.tables[table_id]
        assert table_info.num_players == 1
    
    def test_table_selection(self):
        """Test optimal table selection"""
        # Create multiple tables with different characteristics
        table_configs = [
            create_cash_game_table(min_buy_in=20, max_buy_in=100),
            create_cash_game_table(min_buy_in=50, max_buy_in=500),
            create_tournament_table(buy_in=10, players=6)
        ]
        
        table_ids = []
        for config in table_configs:
            table_id = self.manager.create_table(config)
            table_ids.append(table_id)
        
        # Select optimal table
        optimal_table = self.manager.select_optimal_table("player1")
        assert optimal_table is not None
        assert optimal_table in table_ids
    
    def test_cross_table_learning(self):
        """Test cross-table learning functionality"""
        learning = self.manager.cross_table_learning
        
        # Record table pattern
        pattern = {
            'common_actions': ['call', 'fold'],
            'avg_timing': 2.5,
            'success_rate': 0.6
        }
        learning.record_table_pattern("table1", pattern)
        
        # Record player pattern
        player_pattern = {
            'position': 'BTN',
            'action': 'raise',
            'profitability': 0.8
        }
        learning.record_player_pattern("player1", "table1", player_pattern)
        
        # Test strategy adaptation
        current_strategy = {
            'aggression_level': 0.5,
            'position_ranges': {'BTN': {'frequency': 0.3}}
        }
        
        adapted_strategy = learning.adapt_strategy("table1", "player1", current_strategy)
        assert adapted_strategy is not None
        assert 'aggression_level' in adapted_strategy
    
    def test_table_selection_criteria(self):
        """Test table selection criteria scoring"""
        criteria = TableSelectionCriteria()
        
        # Create test table info
        table_info = TableInfo(
            table_id="test_table",
            table_type=TableType.CASH_GAME,
            state=TableState.WAITING,
            num_players=7,
            max_players=9,
            min_buy_in=30,
            max_buy_in=150,
            small_blind=1,
            big_blind=2,
            avg_stack_size=100,
            avg_vpip=0.35,
            avg_pfr=0.20,
            avg_af=1.5,
            player_skill_level=0.4,
            table_dynamics={},
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Score table
        score = criteria.score_table(table_info)
        assert score > 0  # Should be a positive score for good table
    
    def test_manager_status(self):
        """Test manager status reporting"""
        # Create some tables
        table_config = create_cash_game_table()
        self.manager.create_table(table_config)
        self.manager.create_table(table_config)
        
        # Add players
        table_ids = list(self.manager.tables.keys())
        self.manager.add_player_to_table("player1", table_ids[0])
        self.manager.add_player_to_table("player2", table_ids[1])
        
        # Get status
        status = self.manager.get_manager_status()
        assert status['total_tables'] == 2
        assert status['total_players'] == 2
        assert status['max_tables'] == 5
        assert status['running'] == False
    
    def test_table_performance_tracking(self):
        """Test table performance tracking"""
        # Create table and add performance data
        table_config = create_cash_game_table()
        table_id = self.manager.create_table(table_config)
        
        # Simulate performance data
        self.manager.performance_metrics[table_id] = {
            'hands_played': 100,
            'hands_won': 60,
            'total_profit': 500,
            'avg_profit_per_hand': 5.0,
            'win_rate': 0.6
        }
        
        # Get performance
        performance = self.manager.get_table_performance(table_id)
        assert performance['hands_played'] == 100
        assert performance['win_rate'] == 0.6
        assert performance['avg_profit_per_hand'] == 5.0

class TestIntegration:
    """Integration tests for the advanced features"""
    
    def test_decision_engine_with_analytics(self):
        """Test integration between decision engine and analytics"""
        # Create analytics system
        analytics = StatisticsAnalytics(":memory:")  # Use in-memory database
        
        # Create decision engine
        engine = AdvancedDecisionEngine()
        
        # Start session
        session_id = analytics.start_session("test_player")
        
        # Create game context
        context = GameContext(
            position=Position.BTN,
            stack_size=1000,
            pot_size=100,
            current_bet=20,
            min_raise=40,
            num_players=6,
            active_players=4,
            street='preflop',
            board_cards=[],
            player_cards=[Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)],
            opponent_profiles={},
            action_history=[]
        )
        
        # Make decision
        decision = engine.make_decision(context)
        
        # Record hand result
        hand_result = HandResult(
            hand_id="test_hand",
            timestamp=datetime.now(),
            player_id="test_player",
            position=PlayerPosition.BTN,
            hole_cards=context.player_cards,
            board_cards=context.board_cards,
            action_sequence=[{'action': decision.value, 'amount': 50}],
            final_action=PlayerAction(decision.value),
            pot_size=context.pot_size,
            amount_won=25,
            hand_strength=0.8,
            hand_rank="preflop",
            session_id=session_id
        )
        
        analytics.record_hand(hand_result)
        session_stats = analytics.end_session()
        
        # Verify integration worked
        assert session_stats.total_hands == 1
        assert session_stats.hands_won == 1
        assert session_stats.total_winnings == 25
    
    def test_multi_table_with_decision_engine(self):
        """Test multi-table manager with decision engine integration"""
        manager = MultiTableManager(max_tables=3, max_players_per_table=6)
        
        # Create table
        table_config = create_cash_game_table()
        table_id = manager.create_table(table_config)
        
        # Add players with different bot personalities
        manager.add_player_to_table("player1", table_id, "tight_aggressive")
        manager.add_player_to_table("player2", table_id, "loose_aggressive")
        manager.add_player_to_table("player3", table_id, "tight_passive")
        
        # Verify assignments
        assert len(manager.player_assignments) == 3
        assert manager.player_assignments["player1"].bot_personality == "tight_aggressive"
        assert manager.player_assignments["player2"].bot_personality == "loose_aggressive"
        assert manager.player_assignments["player3"].bot_personality == "tight_passive"
        
        # Test cross-table learning
        insights = manager.get_cross_table_insights()
        assert 'table_patterns' in insights
        assert 'player_patterns' in insights
        assert 'strategy_recommendations' in insights

if __name__ == "__main__":
    pytest.main([__file__, "-v"])