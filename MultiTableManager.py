"""
Multi-Table Management System for Poker Simulator
Handles concurrent games, table selection, cross-table learning, and performance analytics.
"""

import asyncio
import threading
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random
import json
from datetime import datetime, timedelta

from GameBoard import GameBoard
from Player import Player
from AdvancedDecisionEngine import AdvancedDecisionEngine, Position, GameContext
from StatisticsAnalytics import StatisticsAnalytics, HandResult

class TableState(Enum):
    """States a table can be in"""
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"

class TableType(Enum):
    """Types of poker tables"""
    CASH_GAME = "cash_game"
    TOURNAMENT = "tournament"
    SIT_N_GO = "sit_n_go"

@dataclass
class TableInfo:
    """Information about a poker table"""
    table_id: str
    table_type: TableType
    state: TableState
    num_players: int
    max_players: int
    min_buy_in: int
    max_buy_in: int
    small_blind: int
    big_blind: int
    avg_stack_size: float
    avg_vpip: float
    avg_pfr: float
    avg_af: float
    player_skill_level: float  # 0.0 to 1.0
    table_dynamics: Dict[str, float]
    created_at: datetime
    last_activity: datetime

@dataclass
class PlayerTableAssignment:
    """Player assignment to a table"""
    player_id: str
    table_id: str
    assigned_at: datetime
    stack_size: int
    position: Position
    bot_personality: str

class TableSelectionCriteria:
    """Criteria for selecting optimal tables"""
    
    def __init__(self):
        self.min_players = 6
        self.max_players = 9
        self.min_buy_in = 20
        self.max_buy_in = 200
        self.target_vpip_range = (0.25, 0.35)
        self.target_pfr_range = (0.15, 0.25)
        self.max_skill_level = 0.7
        self.min_stack_size = 50
        self.preferred_table_types = [TableType.CASH_GAME]
    
    def score_table(self, table_info: TableInfo) -> float:
        """Score a table based on selection criteria"""
        score = 0.0
        
        # Player count preference
        if self.min_players <= table_info.num_players <= self.max_players:
            score += 10.0
        else:
            score -= 5.0
        
        # Buy-in range preference
        if self.min_buy_in <= table_info.min_buy_in <= self.max_buy_in:
            score += 8.0
        else:
            score -= 3.0
        
        # VPIP preference (want fishy tables)
        if self.target_vpip_range[0] <= table_info.avg_vpip <= self.target_vpip_range[1]:
            score += 15.0
        elif table_info.avg_vpip > self.target_vpip_range[1]:  # Very loose tables
            score += 20.0
        else:
            score -= 5.0
        
        # PFR preference
        if self.target_pfr_range[0] <= table_info.avg_pfr <= self.target_pfr_range[1]:
            score += 10.0
        else:
            score -= 2.0
        
        # Skill level preference (avoid very skilled players)
        if table_info.player_skill_level <= self.max_skill_level:
            score += 12.0
        else:
            score -= 8.0
        
        # Stack size preference
        if table_info.avg_stack_size >= self.min_stack_size:
            score += 5.0
        else:
            score -= 3.0
        
        # Table type preference
        if table_info.table_type in self.preferred_table_types:
            score += 5.0
        
        return score

class CrossTableLearning:
    """Cross-table learning and strategy adaptation"""
    
    def __init__(self):
        self.table_patterns = defaultdict(list)
        self.player_patterns = defaultdict(dict)
        self.strategy_adjustments = defaultdict(dict)
        self.learning_rate = 0.1
        
    def record_table_pattern(self, table_id: str, pattern: Dict):
        """Record a pattern observed at a table"""
        self.table_patterns[table_id].append({
            'pattern': pattern,
            'timestamp': datetime.now(),
            'success_rate': pattern.get('success_rate', 0.0)
        })
        
        # Keep only recent patterns
        if len(self.table_patterns[table_id]) > 100:
            self.table_patterns[table_id] = self.table_patterns[table_id][-50:]
    
    def record_player_pattern(self, player_id: str, table_id: str, pattern: Dict):
        """Record a pattern for a specific player"""
        if player_id not in self.player_patterns:
            self.player_patterns[player_id] = {}
        
        if table_id not in self.player_patterns[player_id]:
            self.player_patterns[player_id][table_id] = []
        
        self.player_patterns[player_id][table_id].append({
            'pattern': pattern,
            'timestamp': datetime.now(),
            'profitability': pattern.get('profitability', 0.0)
        })
    
    def adapt_strategy(self, table_id: str, player_id: str, 
                      current_strategy: Dict) -> Dict:
        """Adapt strategy based on cross-table learning"""
        adapted_strategy = current_strategy.copy()
        
        # Analyze table patterns
        table_patterns = self.table_patterns.get(table_id, [])
        if table_patterns:
            recent_patterns = [p for p in table_patterns 
                             if (datetime.now() - p['timestamp']).seconds < 3600]
            
            if recent_patterns:
                avg_success_rate = np.mean([p['success_rate'] for p in recent_patterns])
                
                # Adjust aggression based on table success
                if avg_success_rate > 0.6:
                    adapted_strategy['aggression_level'] = min(
                        current_strategy.get('aggression_level', 0.5) + 0.1, 1.0
                    )
                elif avg_success_rate < 0.4:
                    adapted_strategy['aggression_level'] = max(
                        current_strategy.get('aggression_level', 0.5) - 0.1, 0.0
                    )
        
        # Analyze player patterns
        player_patterns = self.player_patterns.get(player_id, {}).get(table_id, [])
        if player_patterns:
            recent_player_patterns = [p for p in player_patterns 
                                   if (datetime.now() - p['timestamp']).seconds < 1800]
            
            if recent_player_patterns:
                avg_profitability = np.mean([p['profitability'] for p in recent_player_patterns])
                
                # Adjust position ranges based on profitability
                if avg_profitability > 0:
                    # Expand ranges when profitable
                    for position in adapted_strategy.get('position_ranges', {}):
                        adapted_strategy['position_ranges'][position]['frequency'] *= 1.1
                else:
                    # Tighten ranges when unprofitable
                    for position in adapted_strategy.get('position_ranges', {}):
                        adapted_strategy['position_ranges'][position]['frequency'] *= 0.9
        
        return adapted_strategy
    
    def get_table_insights(self, table_id: str) -> Dict:
        """Get insights about a specific table"""
        patterns = self.table_patterns.get(table_id, [])
        
        if not patterns:
            return {}
        
        recent_patterns = [p for p in patterns 
                         if (datetime.now() - p['timestamp']).seconds < 7200]
        
        if not recent_patterns:
            return {}
        
        insights = {
            'avg_success_rate': np.mean([p['success_rate'] for p in recent_patterns]),
            'pattern_frequency': len(recent_patterns),
            'recommended_strategy': self._recommend_strategy(recent_patterns),
            'table_tendencies': self._analyze_table_tendencies(recent_patterns)
        }
        
        return insights
    
    def _recommend_strategy(self, patterns: List[Dict]) -> Dict:
        """Recommend strategy based on patterns"""
        if not patterns:
            return {}
        
        avg_success = np.mean([p['success_rate'] for p in patterns])
        
        if avg_success > 0.7:
            return {'aggression_level': 'high', 'position_ranges': 'wide'}
        elif avg_success > 0.5:
            return {'aggression_level': 'medium', 'position_ranges': 'balanced'}
        else:
            return {'aggression_level': 'low', 'position_ranges': 'tight'}
    
    def _analyze_table_tendencies(self, patterns: List[Dict]) -> Dict:
        """Analyze table tendencies from patterns"""
        if not patterns:
            return {}
        
        # Extract common patterns
        action_patterns = [p['pattern'].get('common_actions', []) for p in patterns]
        timing_patterns = [p['pattern'].get('avg_timing', 0) for p in patterns]
        
        return {
            'common_actions': self._most_common_pattern(action_patterns),
            'avg_timing': np.mean(timing_patterns) if timing_patterns else 0,
            'pattern_consistency': len(set(str(p['pattern']) for p in patterns)) / len(patterns)
        }
    
    def _most_common_pattern(self, patterns: List[List]) -> List:
        """Find most common pattern in list of patterns"""
        if not patterns:
            return []
        
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_str = str(sorted(pattern))
            pattern_counts[pattern_str] += 1
        
        most_common = max(pattern_counts.items(), key=lambda x: x[1])
        return eval(most_common[0]) if most_common[0] != '[]' else []

class MultiTableManager:
    """
    Manages multiple poker tables with intelligent table selection,
    cross-table learning, and performance analytics.
    """
    
    def __init__(self, max_tables: int = 10, max_players_per_table: int = 9):
        self.max_tables = max_tables
        self.max_players_per_table = max_players_per_table
        self.tables: Dict[str, TableInfo] = {}
        self.active_games: Dict[str, GameBoard] = {}
        self.player_assignments: Dict[str, PlayerTableAssignment] = {}
        self.table_selection_criteria = TableSelectionCriteria()
        self.cross_table_learning = CrossTableLearning()
        self.statistics = StatisticsAnalytics()
        self.performance_metrics = defaultdict(dict)
        
        # Threading
        self.table_threads: Dict[str, threading.Thread] = {}
        self.running = False
        self.lock = threading.Lock()
    
    def start_manager(self):
        """Start the multi-table manager"""
        self.running = True
        print(f"Multi-Table Manager started. Max tables: {self.max_tables}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_tables)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_manager(self):
        """Stop the multi-table manager"""
        self.running = False
        
        # Stop all table threads
        for table_id, thread in self.table_threads.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        print("Multi-Table Manager stopped")
    
    def create_table(self, table_config: Dict) -> str:
        """Create a new poker table"""
        with self.lock:
            if len(self.tables) >= self.max_tables:
                raise ValueError(f"Maximum number of tables ({self.max_tables}) reached")
            
            table_id = f"table_{len(self.tables)}_{int(time.time())}"
            
            table_info = TableInfo(
                table_id=table_id,
                table_type=TableType(table_config.get('type', 'cash_game')),
                state=TableState.WAITING,
                num_players=0,
                max_players=table_config.get('max_players', 9),
                min_buy_in=table_config.get('min_buy_in', 20),
                max_buy_in=table_config.get('max_buy_in', 200),
                small_blind=table_config.get('small_blind', 1),
                big_blind=table_config.get('big_blind', 2),
                avg_stack_size=table_config.get('avg_stack_size', 100),
                avg_vpip=table_config.get('avg_vpip', 0.3),
                avg_pfr=table_config.get('avg_pfr', 0.2),
                avg_af=table_config.get('avg_af', 1.5),
                player_skill_level=table_config.get('player_skill_level', 0.5),
                table_dynamics=table_config.get('table_dynamics', {}),
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.tables[table_id] = table_info
            
            # Start table thread
            table_thread = threading.Thread(
                target=self._run_table,
                args=(table_id,),
                daemon=True
            )
            self.table_threads[table_id] = table_thread
            table_thread.start()
            
            print(f"Created table {table_id}")
            return table_id
    
    def add_player_to_table(self, player_id: str, table_id: str, 
                           bot_personality: str = 'balanced') -> bool:
        """Add a player to a specific table"""
        with self.lock:
            if table_id not in self.tables:
                print(f"Table {table_id} not found")
                return False
            
            table_info = self.tables[table_id]
            
            if table_info.num_players >= table_info.max_players:
                print(f"Table {table_id} is full")
                return False
            
            if table_info.state != TableState.WAITING:
                print(f"Table {table_id} is not in waiting state")
                return False
            
            # Create player assignment
            assignment = PlayerTableAssignment(
                player_id=player_id,
                table_id=table_id,
                assigned_at=datetime.now(),
                stack_size=table_info.min_buy_in,
                position=Position.UTG,  # Will be determined by game
                bot_personality=bot_personality
            )
            
            self.player_assignments[player_id] = assignment
            table_info.num_players += 1
            table_info.last_activity = datetime.now()
            
            print(f"Added player {player_id} to table {table_id}")
            return True
    
    def select_optimal_table(self, player_id: str, 
                           preferences: Dict = None) -> Optional[str]:
        """Select the optimal table for a player based on criteria"""
        available_tables = [
            table_id for table_id, table_info in self.tables.items()
            if table_info.state == TableState.WAITING and 
            table_info.num_players < table_info.max_players
        ]
        
        if not available_tables:
            return None
        
        # Score each table
        table_scores = []
        for table_id in available_tables:
            table_info = self.tables[table_id]
            score = self.table_selection_criteria.score_table(table_info)
            
            # Apply player preferences if provided
            if preferences:
                score = self._apply_player_preferences(score, table_info, preferences)
            
            table_scores.append((table_id, score))
        
        # Sort by score (highest first)
        table_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best table
        return table_scores[0][0] if table_scores else None
    
    def _apply_player_preferences(self, base_score: float, 
                                table_info: TableInfo, 
                                preferences: Dict) -> float:
        """Apply player-specific preferences to table score"""
        adjusted_score = base_score
        
        # Buy-in preference
        if 'preferred_buy_in' in preferences:
            preferred = preferences['preferred_buy_in']
            if abs(table_info.min_buy_in - preferred) < 20:
                adjusted_score += 5.0
            else:
                adjusted_score -= 3.0
        
        # Table type preference
        if 'preferred_table_type' in preferences:
            if table_info.table_type.value == preferences['preferred_table_type']:
                adjusted_score += 3.0
        
        # Skill level preference
        if 'max_skill_level' in preferences:
            if table_info.player_skill_level <= preferences['max_skill_level']:
                adjusted_score += 8.0
            else:
                adjusted_score -= 10.0
        
        return adjusted_score
    
    def get_table_performance(self, table_id: str) -> Dict:
        """Get performance metrics for a specific table"""
        if table_id not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[table_id]
        
        # Calculate additional metrics
        if 'hands_played' in metrics and metrics['hands_played'] > 0:
            metrics['avg_profit_per_hand'] = metrics['total_profit'] / metrics['hands_played']
            metrics['win_rate'] = metrics['hands_won'] / metrics['hands_played']
        
        return metrics
    
    def get_cross_table_insights(self) -> Dict:
        """Get insights from cross-table learning"""
        insights = {
            'table_patterns': {},
            'player_patterns': {},
            'strategy_recommendations': {}
        }
        
        # Analyze patterns for each table
        for table_id in self.tables:
            insights['table_patterns'][table_id] = self.cross_table_learning.get_table_insights(table_id)
        
        # Analyze player patterns
        for player_id in self.player_assignments:
            player_patterns = self.cross_table_learning.player_patterns.get(player_id, {})
            if player_patterns:
                insights['player_patterns'][player_id] = {
                    'tables_played': list(player_patterns.keys()),
                    'total_patterns': sum(len(patterns) for patterns in player_patterns.values()),
                    'avg_profitability': self._calculate_player_avg_profitability(player_id)
                }
        
        return insights
    
    def _calculate_player_avg_profitability(self, player_id: str) -> float:
        """Calculate average profitability for a player across tables"""
        player_patterns = self.cross_table_learning.player_patterns.get(player_id, {})
        
        if not player_patterns:
            return 0.0
        
        all_profitabilities = []
        for table_patterns in player_patterns.values():
            for pattern in table_patterns:
                all_profitabilities.append(pattern.get('profitability', 0.0))
        
        return np.mean(all_profitabilities) if all_profitabilities else 0.0
    
    def _run_table(self, table_id: str):
        """Run a single table in its own thread"""
        try:
            table_info = self.tables[table_id]
            table_info.state = TableState.ACTIVE
            
            # Create game board for this table
            game_board = GameBoard(buy_in=table_info.min_buy_in)
            
            # Add players to the game
            table_players = [
                player_id for player_id, assignment in self.player_assignments.items()
                if assignment.table_id == table_id
            ]
            
            # Start the game
            asyncio.run(self._run_table_game(table_id, game_board, table_players))
            
        except Exception as e:
            print(f"Error running table {table_id}: {e}")
            self.tables[table_id].state = TableState.ERROR
    
    async def _run_table_game(self, table_id: str, game_board: GameBoard, 
                             players: List[str]):
        """Run a single table game"""
        try:
            # Initialize game board
            await game_board.initialize_connections()
            
            # Start players
            player_tasks = []
            for player_id in players:
                player = Player(player_id)
                task = asyncio.create_task(player.main())
                player_tasks.append(task)
            
            # Run the game
            await game_board.main()
            
            # Wait for players to finish
            await asyncio.gather(*player_tasks, return_exceptions=True)
            
            # Update table state
            self.tables[table_id].state = TableState.FINISHED
            
        except Exception as e:
            print(f"Error in table game {table_id}: {e}")
            self.tables[table_id].state = TableState.ERROR
    
    def _monitor_tables(self):
        """Monitor all tables and update performance metrics"""
        while self.running:
            try:
                with self.lock:
                    for table_id, table_info in self.tables.items():
                        # Update table activity
                        if table_info.state == TableState.ACTIVE:
                            table_info.last_activity = datetime.now()
                        
                        # Clean up finished tables
                        if table_info.state == TableState.FINISHED:
                            self._cleanup_table(table_id)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in table monitor: {e}")
                time.sleep(30)
    
    def _cleanup_table(self, table_id: str):
        """Clean up a finished table"""
        # Remove table from active games
        if table_id in self.active_games:
            del self.active_games[table_id]
        
        # Remove table threads
        if table_id in self.table_threads:
            del self.table_threads[table_id]
        
        # Remove player assignments for this table
        players_to_remove = [
            player_id for player_id, assignment in self.player_assignments.items()
            if assignment.table_id == table_id
        ]
        
        for player_id in players_to_remove:
            del self.player_assignments[player_id]
        
        print(f"Cleaned up table {table_id}")
    
    def get_manager_status(self) -> Dict:
        """Get overall status of the multi-table manager"""
        with self.lock:
            active_tables = sum(1 for table in self.tables.values() 
                              if table.state == TableState.ACTIVE)
            waiting_tables = sum(1 for table in self.tables.values() 
                               if table.state == TableState.WAITING)
            total_players = len(self.player_assignments)
            
            return {
                'total_tables': len(self.tables),
                'active_tables': active_tables,
                'waiting_tables': waiting_tables,
                'total_players': total_players,
                'max_tables': self.max_tables,
                'running': self.running
            }

# Factory functions for different table configurations
def create_cash_game_table(min_buy_in: int = 20, max_buy_in: int = 200) -> Dict:
    """Create a cash game table configuration"""
    return {
        'type': 'cash_game',
        'max_players': 9,
        'min_buy_in': min_buy_in,
        'max_buy_in': max_buy_in,
        'small_blind': 1,
        'big_blind': 2,
        'avg_stack_size': 100,
        'avg_vpip': 0.3,
        'avg_pfr': 0.2,
        'avg_af': 1.5,
        'player_skill_level': 0.5,
        'table_dynamics': {}
    }

def create_tournament_table(buy_in: int = 10, players: int = 9) -> Dict:
    """Create a tournament table configuration"""
    return {
        'type': 'tournament',
        'max_players': players,
        'min_buy_in': buy_in,
        'max_buy_in': buy_in,
        'small_blind': 1,
        'big_blind': 2,
        'avg_stack_size': 1500,
        'avg_vpip': 0.25,
        'avg_pfr': 0.15,
        'avg_af': 1.8,
        'player_skill_level': 0.6,
        'table_dynamics': {'tournament': True}
    }