"""
Comprehensive Statistics and Analytics System for Poker Simulator
Provides detailed statistics, hand history analysis, bankroll management, and performance tracking.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pickle
import os
from pathlib import Path

from poker import Card, Rank, Suit
from utils import PlayerAction, PlayerPosition

@dataclass
class HandResult:
    """Result of a single hand"""
    hand_id: str
    timestamp: datetime
    player_id: str
    position: PlayerPosition
    hole_cards: List[Card]
    board_cards: List[Card]
    action_sequence: List[Dict]
    final_action: PlayerAction
    pot_size: int
    amount_won: int
    hand_strength: float
    hand_rank: str
    session_id: str

@dataclass
class SessionStats:
    """Statistics for a single session"""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_hands: int
    hands_won: int
    total_winnings: int
    total_losses: int
    net_profit: int
    biggest_pot: int
    biggest_loss: int
    vpip: float  # Voluntarily Put Money In Pot
    pfr: float   # Pre-Flop Raise
    af: float    # Aggression Factor
    avg_bet_size: float
    fold_percentage: float

@dataclass
class PlayerProfile:
    """Complete player profile with statistics"""
    player_id: str
    total_sessions: int
    total_hands: int
    total_winnings: int
    total_losses: int
    net_profit: int
    win_rate: float
    avg_profit_per_hand: float
    best_session: str
    worst_session: str
    preferred_positions: List[PlayerPosition]
    playing_style: str  # tight-aggressive, loose-passive, etc.
    bankroll_management: Dict[str, float]

class StatisticsAnalytics:
    """
    Comprehensive statistics and analytics system for poker performance tracking
    """
    
    def __init__(self, db_path: str = "poker_stats.db"):
        self.db_path = db_path
        self.initialize_database()
        self.current_session = None
        self.session_hands = []
        
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create hands table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hands (
                hand_id TEXT PRIMARY KEY,
                session_id TEXT,
                player_id TEXT,
                position TEXT,
                hole_cards TEXT,
                board_cards TEXT,
                action_sequence TEXT,
                final_action TEXT,
                pot_size INTEGER,
                amount_won INTEGER,
                hand_strength REAL,
                hand_rank TEXT,
                timestamp TEXT
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                player_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_hands INTEGER,
                hands_won INTEGER,
                total_winnings INTEGER,
                total_losses INTEGER,
                net_profit INTEGER,
                biggest_pot INTEGER,
                biggest_loss INTEGER,
                vpip REAL,
                pfr REAL,
                af REAL,
                avg_bet_size REAL,
                fold_percentage REAL
            )
        ''')
        
        # Create player_profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_profiles (
                player_id TEXT PRIMARY KEY,
                total_sessions INTEGER,
                total_hands INTEGER,
                total_winnings INTEGER,
                total_losses INTEGER,
                net_profit INTEGER,
                win_rate REAL,
                avg_profit_per_hand REAL,
                best_session TEXT,
                worst_session TEXT,
                preferred_positions TEXT,
                playing_style TEXT,
                bankroll_management TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, player_id: str) -> str:
        """Start a new session and return session ID"""
        session_id = f"{player_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = {
            'session_id': session_id,
            'player_id': player_id,
            'start_time': datetime.now(),
            'hands': [],
            'total_winnings': 0,
            'total_losses': 0,
            'hands_won': 0,
            'total_hands': 0
        }
        return session_id
    
    def end_session(self) -> SessionStats:
        """End current session and return session statistics"""
        if not self.current_session:
            raise ValueError("No active session to end")
        
        end_time = datetime.now()
        session = self.current_session
        
        # Calculate session statistics
        total_hands = len(session['hands'])
        hands_won = session['hands_won']
        total_winnings = session['total_winnings']
        total_losses = session['total_losses']
        net_profit = total_winnings - total_losses
        
        # Calculate betting statistics
        vpip, pfr, af, avg_bet_size, fold_percentage = self._calculate_betting_stats(session['hands'])
        
        # Find biggest pot and loss
        biggest_pot = max([hand.get('pot_size', 0) for hand in session['hands']]) if session['hands'] else 0
        biggest_loss = max([abs(hand.get('amount_won', 0)) for hand in session['hands'] if hand.get('amount_won', 0) < 0]) if session['hands'] else 0
        
        session_stats = SessionStats(
            session_id=session['session_id'],
            start_time=session['start_time'],
            end_time=end_time,
            total_hands=total_hands,
            hands_won=hands_won,
            total_winnings=total_winnings,
            total_losses=total_losses,
            net_profit=net_profit,
            biggest_pot=biggest_pot,
            biggest_loss=biggest_loss,
            vpip=vpip,
            pfr=pfr,
            af=af,
            avg_bet_size=avg_bet_size,
            fold_percentage=fold_percentage
        )
        
        # Save to database
        self._save_session_to_db(session_stats)
        
        # Reset current session
        self.current_session = None
        
        return session_stats
    
    def record_hand(self, hand_result: HandResult):
        """Record a hand result"""
        if not self.current_session:
            raise ValueError("No active session to record hand")
        
        # Add to current session
        self.current_session['hands'].append(asdict(hand_result))
        self.current_session['total_hands'] += 1
        
        if hand_result.amount_won > 0:
            self.current_session['hands_won'] += 1
            self.current_session['total_winnings'] += hand_result.amount_won
        else:
            self.current_session['total_losses'] += abs(hand_result.amount_won)
        
        # Save to database
        self._save_hand_to_db(hand_result)
    
    def _calculate_betting_stats(self, hands: List[Dict]) -> Tuple[float, float, float, float, float]:
        """Calculate betting statistics from hands"""
        if not hands:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        total_hands = len(hands)
        vpip_hands = 0
        pfr_hands = 0
        aggressive_actions = 0
        passive_actions = 0
        total_bet_size = 0
        fold_hands = 0
        
        for hand in hands:
            action_sequence = json.loads(hand.get('action_sequence', '[]'))
            
            # Count VPIP (voluntarily put money in pot)
            if any(action.get('action') in ['bet', 'raise', 'call'] for action in action_sequence):
                vpip_hands += 1
            
            # Count PFR (pre-flop raise)
            if any(action.get('action') == 'raise' and action.get('street') == 'preflop' 
                   for action in action_sequence):
                pfr_hands += 1
            
            # Count aggression factor
            for action in action_sequence:
                if action.get('action') in ['bet', 'raise']:
                    aggressive_actions += 1
                elif action.get('action') in ['call', 'check']:
                    passive_actions += 1
                
                if action.get('action') == 'fold':
                    fold_hands += 1
                
                if 'amount' in action:
                    total_bet_size += action['amount']
        
        vpip = vpip_hands / total_hands if total_hands > 0 else 0.0
        pfr = pfr_hands / total_hands if total_hands > 0 else 0.0
        af = aggressive_actions / (passive_actions + 1) if passive_actions > 0 else aggressive_actions
        avg_bet_size = total_bet_size / total_hands if total_hands > 0 else 0.0
        fold_percentage = fold_hands / total_hands if total_hands > 0 else 0.0
        
        return vpip, pfr, af, avg_bet_size, fold_percentage
    
    def _save_hand_to_db(self, hand_result: HandResult):
        """Save hand result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO hands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            hand_result.hand_id,
            hand_result.session_id,
            hand_result.player_id,
            hand_result.position.value,
            json.dumps([f"{card.rank.value}{card.suit.value}" for card in hand_result.hole_cards]),
            json.dumps([f"{card.rank.value}{card.suit.value}" for card in hand_result.board_cards]),
            json.dumps(hand_result.action_sequence),
            hand_result.final_action.value,
            hand_result.pot_size,
            hand_result.amount_won,
            hand_result.hand_strength,
            hand_result.hand_rank,
            hand_result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _save_session_to_db(self, session_stats: SessionStats):
        """Save session statistics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_stats.session_id,
            self.current_session['player_id'],
            session_stats.start_time.isoformat(),
            session_stats.end_time.isoformat(),
            session_stats.total_hands,
            session_stats.hands_won,
            session_stats.total_winnings,
            session_stats.total_losses,
            session_stats.net_profit,
            session_stats.biggest_pot,
            session_stats.biggest_loss,
            session_stats.vpip,
            session_stats.pfr,
            session_stats.af,
            session_stats.avg_bet_size,
            session_stats.fold_percentage
        ))
        
        conn.commit()
        conn.close()
    
    def get_player_profile(self, player_id: str) -> PlayerProfile:
        """Get complete player profile with statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT session_id) as total_sessions,
                COUNT(*) as total_hands,
                SUM(amount_won) as total_winnings,
                SUM(CASE WHEN amount_won < 0 THEN ABS(amount_won) ELSE 0 END) as total_losses,
                SUM(amount_won) as net_profit,
                AVG(CASE WHEN amount_won > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(amount_won) as avg_profit_per_hand
            FROM hands 
            WHERE player_id = ?
        ''', (player_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        total_sessions, total_hands, total_winnings, total_losses, net_profit, win_rate, avg_profit_per_hand = row
        
        # Get best and worst sessions
        cursor.execute('''
            SELECT session_id, net_profit FROM sessions 
            WHERE player_id = ? 
            ORDER BY net_profit DESC
        ''', (player_id,))
        
        sessions = cursor.fetchall()
        best_session = sessions[0][0] if sessions else None
        worst_session = sessions[-1][0] if sessions else None
        
        # Get preferred positions
        cursor.execute('''
            SELECT position, COUNT(*) as count 
            FROM hands 
            WHERE player_id = ? 
            GROUP BY position 
            ORDER BY count DESC
        ''', (player_id,))
        
        position_counts = cursor.fetchall()
        preferred_positions = [PlayerPosition(pos) for pos, _ in position_counts[:3]]
        
        # Determine playing style
        cursor.execute('''
            SELECT AVG(vpip), AVG(pfr), AVG(af) 
            FROM sessions 
            WHERE player_id = ?
        ''', (player_id,))
        
        avg_vpip, avg_pfr, avg_af = cursor.fetchone() or (0, 0, 0)
        playing_style = self._determine_playing_style(avg_vpip, avg_pfr, avg_af)
        
        # Calculate bankroll management
        bankroll_management = self._calculate_bankroll_management(player_id)
        
        conn.close()
        
        return PlayerProfile(
            player_id=player_id,
            total_sessions=total_sessions or 0,
            total_hands=total_hands or 0,
            total_winnings=total_winnings or 0,
            total_losses=total_losses or 0,
            net_profit=net_profit or 0,
            win_rate=win_rate or 0.0,
            avg_profit_per_hand=avg_profit_per_hand or 0.0,
            best_session=best_session,
            worst_session=worst_session,
            preferred_positions=preferred_positions,
            playing_style=playing_style,
            bankroll_management=bankroll_management
        )
    
    def _determine_playing_style(self, vpip: float, pfr: float, af: float) -> str:
        """Determine playing style based on statistics"""
        if vpip < 0.25:
            tightness = "tight"
        elif vpip < 0.35:
            tightness = "medium"
        else:
            tightness = "loose"
        
        if af > 2.0:
            aggression = "aggressive"
        elif af > 1.0:
            aggression = "medium"
        else:
            aggression = "passive"
        
        return f"{tightness}-{aggression}"
    
    def _calculate_bankroll_management(self, player_id: str) -> Dict[str, float]:
        """Calculate bankroll management metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session results
        cursor.execute('''
            SELECT net_profit, total_hands 
            FROM sessions 
            WHERE player_id = ? 
            ORDER BY start_time
        ''', (player_id,))
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            return {}
        
        profits = [profit for profit, _ in sessions]
        total_hands = sum(hands for _, hands in sessions)
        
        # Calculate bankroll metrics
        total_profit = sum(profits)
        avg_profit_per_session = np.mean(profits)
        profit_std = np.std(profits)
        max_drawdown = self._calculate_max_drawdown(profits)
        risk_of_ruin = self._calculate_risk_of_ruin(profits)
        
        return {
            'total_profit': total_profit,
            'avg_profit_per_session': avg_profit_per_session,
            'profit_volatility': profit_std,
            'max_drawdown': max_drawdown,
            'risk_of_ruin': risk_of_ruin,
            'profit_per_hand': total_profit / total_hands if total_hands > 0 else 0.0
        }
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from peak"""
        if not profits:
            return 0.0
        
        peak = profits[0]
        max_drawdown = 0.0
        
        for profit in profits:
            if profit > peak:
                peak = profit
            drawdown = peak - profit
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _calculate_risk_of_ruin(self, profits: List[float]) -> float:
        """Calculate risk of ruin based on profit distribution"""
        if not profits:
            return 0.0
        
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        
        if std_profit == 0:
            return 0.0 if mean_profit > 0 else 1.0
        
        # Simplified risk of ruin calculation
        z_score = -mean_profit / std_profit
        risk = 1 - (1 / (1 + np.exp(z_score)))
        
        return risk
    
    def generate_heat_map(self, player_id: str, metric: str = 'profit') -> np.ndarray:
        """Generate heat map for player performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get position vs street performance
        cursor.execute('''
            SELECT position, hand_rank, amount_won 
            FROM hands 
            WHERE player_id = ?
        ''', (player_id,))
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            return np.zeros((6, 4))  # 6 positions, 4 streets
        
        # Create heat map data
        heat_map = np.zeros((6, 4))
        position_map = {pos.value: i for i, pos in enumerate(PlayerPosition)}
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        
        for position, hand_rank, amount_won in data:
            pos_idx = position_map.get(position, 0)
            # Determine street from hand rank (simplified)
            if 'preflop' in hand_rank:
                street_idx = 0
            elif 'flop' in hand_rank:
                street_idx = 1
            elif 'turn' in hand_rank:
                street_idx = 2
            else:
                street_idx = 3
            
            heat_map[pos_idx][street_idx] += amount_won
        
        return heat_map
    
    def plot_performance_trends(self, player_id: str, save_path: str = None):
        """Generate performance trend plots"""
        conn = sqlite3.connect(self.db_path)
        
        # Get session data
        sessions_df = pd.read_sql_query('''
            SELECT * FROM sessions 
            WHERE player_id = ? 
            ORDER BY start_time
        ''', conn, params=(player_id,))
        
        conn.close()
        
        if sessions_df.empty:
            print("No data available for plotting")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Profit over time
        sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
        sessions_df['cumulative_profit'] = sessions_df['net_profit'].cumsum()
        
        axes[0, 0].plot(sessions_df['start_time'], sessions_df['cumulative_profit'])
        axes[0, 0].set_title('Cumulative Profit Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Profit')
        
        # Win rate over time
        sessions_df['cumulative_win_rate'] = sessions_df['hands_won'].cumsum() / sessions_df['total_hands'].cumsum()
        axes[0, 1].plot(sessions_df['start_time'], sessions_df['cumulative_win_rate'])
        axes[0, 1].set_title('Cumulative Win Rate Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Win Rate')
        
        # VPIP and PFR over time
        axes[1, 0].plot(sessions_df['start_time'], sessions_df['vpip'], label='VPIP')
        axes[1, 0].plot(sessions_df['start_time'], sessions_df['pfr'], label='PFR')
        axes[1, 0].set_title('VPIP and PFR Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].legend()
        
        # Session profit distribution
        axes[1, 1].hist(sessions_df['net_profit'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Session Profit Distribution')
        axes[1, 1].set_xlabel('Profit')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def export_hand_history(self, player_id: str, format: str = 'json') -> str:
        """Export hand history in specified format"""
        conn = sqlite3.connect(self.db_path)
        
        if format == 'json':
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM hands WHERE player_id = ?
            ''', (player_id,))
            
            columns = [description[0] for description in cursor.description]
            hands = []
            
            for row in cursor.fetchall():
                hand_dict = dict(zip(columns, row))
                # Parse JSON fields
                hand_dict['hole_cards'] = json.loads(hand_dict['hole_cards'])
                hand_dict['board_cards'] = json.loads(hand_dict['board_cards'])
                hand_dict['action_sequence'] = json.loads(hand_dict['action_sequence'])
                hands.append(hand_dict)
            
            conn.close()
            return json.dumps(hands, indent=2)
        
        elif format == 'csv':
            df = pd.read_sql_query('''
                SELECT * FROM hands WHERE player_id = ?
            ''', conn, params=(player_id,))
            
            conn.close()
            return df.to_csv(index=False)
        
        else:
            conn.close()
            raise ValueError(f"Unsupported format: {format}")
    
    def get_hand_analysis(self, hand_id: str) -> Dict[str, Any]:
        """Get detailed analysis of a specific hand"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM hands WHERE hand_id = ?
        ''', (hand_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {}
        
        columns = [description[0] for description in cursor.description]
        hand_data = dict(zip(columns, row))
        
        # Parse JSON fields
        hand_data['hole_cards'] = json.loads(hand_data['hole_cards'])
        hand_data['board_cards'] = json.loads(hand_data['board_cards'])
        hand_data['action_sequence'] = json.loads(hand_data['action_sequence'])
        
        # Add analysis
        analysis = {
            'hand_data': hand_data,
            'action_frequency': Counter([action.get('action') for action in hand_data['action_sequence']]),
            'betting_pattern': self._analyze_betting_pattern(hand_data['action_sequence']),
            'position_analysis': self._analyze_position_play(hand_data),
            'pot_odds_analysis': self._analyze_pot_odds(hand_data)
        }
        
        return analysis
    
    def _analyze_betting_pattern(self, action_sequence: List[Dict]) -> Dict[str, Any]:
        """Analyze betting pattern in action sequence"""
        if not action_sequence:
            return {}
        
        bet_sizes = [action.get('amount', 0) for action in action_sequence if 'amount' in action]
        actions = [action.get('action') for action in action_sequence]
        
        return {
            'total_actions': len(action_sequence),
            'avg_bet_size': np.mean(bet_sizes) if bet_sizes else 0,
            'max_bet_size': max(bet_sizes) if bet_sizes else 0,
            'action_frequency': Counter(actions),
            'betting_aggression': actions.count('raise') / max(actions.count('call'), 1)
        }
    
    def _analyze_position_play(self, hand_data: Dict) -> Dict[str, Any]:
        """Analyze position-based play"""
        position = hand_data.get('position')
        amount_won = hand_data.get('amount_won', 0)
        
        return {
            'position': position,
            'profitability': amount_won > 0,
            'position_profit': amount_won
        }
    
    def _analyze_pot_odds(self, hand_data: Dict) -> Dict[str, Any]:
        """Analyze pot odds in the hand"""
        pot_size = hand_data.get('pot_size', 0)
        amount_won = hand_data.get('amount_won', 0)
        
        return {
            'pot_size': pot_size,
            'roi': amount_won / pot_size if pot_size > 0 else 0,
            'pot_odds_decision': self._evaluate_pot_odds_decision(hand_data)
        }
    
    def _evaluate_pot_odds_decision(self, hand_data: Dict) -> str:
        """Evaluate if pot odds decision was correct"""
        # Simplified evaluation - in practice would need more sophisticated analysis
        amount_won = hand_data.get('amount_won', 0)
        return "good" if amount_won > 0 else "questionable"