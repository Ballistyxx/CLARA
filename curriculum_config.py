"""
Curriculum Learning Configuration for CLARA Analog Layout Training

This module defines the staged curriculum system that progressively introduces
advanced metrics into RL training rewards, improving layout quality while
maintaining training stability.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    stage_id: int
    name: str
    episode_range: Tuple[int, int]  # (start_episode, end_episode) 
    simple_weight: float
    advanced_weight: float
    metrics: List[str]
    description: str
    success_threshold: float = 0.7  # Success rate needed to advance
    min_episodes: int = 1000  # Minimum episodes before advancing

class CurriculumManager:
    """Manages curriculum learning progression and stage transitions."""
    
    # Stage definitions following the 4-stage curriculum approach
    STAGE_CONFIGS = {
        1: StageConfig(
            stage_id=1,
            name="Basic Placement",
            episode_range=(0, 10000),
            simple_weight=1.0,
            advanced_weight=0.0,
            metrics=["completion", "row_consistency", "basic_compactness"],
            description="Focus on successful component placement and basic spatial organization",
            success_threshold=0.7,
            min_episodes=1000
        ),
        2: StageConfig(
            stage_id=2,
            name="Quality Introduction", 
            episode_range=(10000, 25000),
            simple_weight=0.8,
            advanced_weight=0.2,
            metrics=["completion", "row_consistency", "basic_compactness", 
                    "symmetry_score", "abutment_alignment"],
            description="Introduce symmetry and alignment awareness",
            success_threshold=0.7,
            min_episodes=1000
        ),
        3: StageConfig(
            stage_id=3,
            name="Routing Awareness",
            episode_range=(25000, 40000),
            simple_weight=0.6,
            advanced_weight=0.4,
            metrics=["completion", "row_consistency", "basic_compactness",
                    "symmetry_score", "abutment_alignment", "rail_alignment",
                    "avg_connection_distance", "crossings"],
            description="Add routing and connectivity considerations",
            success_threshold=0.7,
            min_episodes=1000
        ),
        4: StageConfig(
            stage_id=4,
            name="Full Optimization",
            episode_range=(40000, float('inf')),
            simple_weight=0.4,
            advanced_weight=0.6,
            metrics=["all"],  # Special case: use all available metrics
            description="Full analog layout optimization with all constraints",
            success_threshold=0.8,
            min_episodes=1000
        )
    }
    
    def __init__(self, manual_stage: int = None, episode_window: int = 1000):
        """Initialize curriculum manager.
        
        Args:
            manual_stage: Force specific stage (disables automatic progression)
            episode_window: Window size for calculating success rates
        """
        self.manual_stage = manual_stage
        self.episode_window = episode_window
        self.episode_count = 0
        self.success_history = []  # Track recent success rates
        self.stage_history = []  # Track stage transitions
        
        # Performance tracking
        self.stage_performance = {i: {"episodes": 0, "successes": 0, "avg_reward": 0.0} 
                                 for i in range(1, 5)}
        
        logger.info(f"Initialized curriculum manager with {len(self.STAGE_CONFIGS)} stages")
        if manual_stage:
            logger.info(f"Manual stage override: Stage {manual_stage}")
    
    def get_current_stage(self, episode: int = None) -> int:
        """Determine current curriculum stage based on episode count."""
        if self.manual_stage:
            return self.manual_stage
            
        if episode is None:
            episode = self.episode_count
            
        # Find appropriate stage based on episode ranges
        for stage_id, config in self.STAGE_CONFIGS.items():
            start, end = config.episode_range
            if start <= episode < end:
                return stage_id
        
        # Default to final stage if beyond all ranges
        return 4
    
    def get_stage_config(self, stage_id: int) -> StageConfig:
        """Get configuration for specified stage."""
        return self.STAGE_CONFIGS.get(stage_id, self.STAGE_CONFIGS[1])
    
    def should_advance_stage(self, current_stage: int) -> bool:
        """Check if ready to advance to next curriculum stage.
        
        Args:
            current_stage: Current curriculum stage
            
        Returns:
            True if should advance to next stage
        """
        if self.manual_stage or current_stage >= 4:
            return False
            
        config = self.get_stage_config(current_stage)
        stage_perf = self.stage_performance[current_stage]
        
        # Need minimum episodes in current stage
        if stage_perf["episodes"] < config.min_episodes:
            return False
            
        # Calculate recent success rate
        if len(self.success_history) >= self.episode_window:
            recent_successes = self.success_history[-self.episode_window:]
            success_rate = sum(recent_successes) / len(recent_successes)
            
            return success_rate >= config.success_threshold
        
        return False
    
    def update_episode(self, episode: int, success: bool, total_reward: float):
        """Update curriculum manager with episode results.
        
        Args:
            episode: Episode number
            success: Whether episode was successful (all components placed)
            total_reward: Total reward received in episode
        """
        self.episode_count = episode
        self.success_history.append(success)
        
        # Keep history bounded
        if len(self.success_history) > self.episode_window * 2:
            self.success_history = self.success_history[-self.episode_window:]
        
        # Update stage performance
        current_stage = self.get_current_stage(episode)
        perf = self.stage_performance[current_stage]
        perf["episodes"] += 1
        if success:
            perf["successes"] += 1
        
        # Update average reward (exponential moving average)
        alpha = 0.01  # Smoothing factor
        perf["avg_reward"] = (1 - alpha) * perf["avg_reward"] + alpha * total_reward
        
        # Check for stage advancement
        if self.should_advance_stage(current_stage):
            next_stage = current_stage + 1
            if next_stage <= 4:
                self.stage_history.append((episode, current_stage, next_stage))
                logger.info(f"Stage advancement at episode {episode}: {current_stage} -> {next_stage}")
    
    def get_reward_weights(self, stage_id: int) -> Tuple[float, float]:
        """Get simple and advanced reward weights for stage.
        
        Args:
            stage_id: Curriculum stage ID
            
        Returns:
            (simple_weight, advanced_weight) tuple
        """
        config = self.get_stage_config(stage_id)
        return config.simple_weight, config.advanced_weight
    
    def get_stage_metrics(self, stage_id: int) -> List[str]:
        """Get list of metrics to calculate for stage.
        
        Args:
            stage_id: Curriculum stage ID
            
        Returns:
            List of metric names to calculate
        """
        config = self.get_stage_config(stage_id)
        return config.metrics.copy()
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get comprehensive curriculum status for logging/debugging.
        
        Returns:
            Dictionary with curriculum state information
        """
        current_stage = self.get_current_stage()
        config = self.get_stage_config(current_stage)
        
        # Calculate recent success rate
        recent_success_rate = 0.0
        if len(self.success_history) >= min(100, self.episode_window // 10):
            recent_window = min(self.episode_window, len(self.success_history))
            recent_successes = self.success_history[-recent_window:]
            recent_success_rate = sum(recent_successes) / len(recent_successes)
        
        return {
            "current_stage": current_stage,
            "stage_name": config.name,
            "episode_count": self.episode_count,
            "simple_weight": config.simple_weight,
            "advanced_weight": config.advanced_weight,
            "active_metrics": config.metrics,
            "recent_success_rate": recent_success_rate,
            "stage_performance": self.stage_performance[current_stage].copy(),
            "stage_transitions": self.stage_history.copy(),
            "can_advance": self.should_advance_stage(current_stage) if current_stage < 4 else False
        }
    
    def reset_stage_performance(self, stage_id: int):
        """Reset performance tracking for a specific stage."""
        self.stage_performance[stage_id] = {
            "episodes": 0,
            "successes": 0,
            "avg_reward": 0.0
        }
        logger.info(f"Reset performance tracking for stage {stage_id}")

# Predefined metric collections for efficient lookup
BASIC_METRICS = ["completion", "row_consistency", "basic_compactness"]
QUALITY_METRICS = BASIC_METRICS + ["symmetry_score", "abutment_alignment"]
ROUTING_METRICS = QUALITY_METRICS + ["rail_alignment", "avg_connection_distance", "crossings"]
ALL_METRICS = [
    "completion", "row_consistency", "basic_compactness", "symmetry_score",
    "abutment_alignment", "rail_alignment", "avg_connection_distance", "crossings",
    "matching_accuracy", "spacing_violations", "overlap_penalties", "wire_length",
    "area_efficiency", "power_rail_alignment", "signal_integrity"
]

def get_metrics_for_stage(stage_id: int) -> List[str]:
    """Get metrics list for a specific stage (utility function).
    
    Args:
        stage_id: Curriculum stage (1-4)
        
    Returns:
        List of metric names for the stage
    """
    metric_map = {
        1: BASIC_METRICS,
        2: QUALITY_METRICS, 
        3: ROUTING_METRICS,
        4: ALL_METRICS
    }
    return metric_map.get(stage_id, BASIC_METRICS).copy()

def create_curriculum_manager(config: Dict[str, Any] = None) -> CurriculumManager:
    """Factory function to create curriculum manager with optional config.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured CurriculumManager instance
    """
    if config is None:
        config = {}
    
    manual_stage = config.get("manual_stage")
    episode_window = config.get("episode_window", 1000)
    
    manager = CurriculumManager(
        manual_stage=manual_stage,
        episode_window=episode_window
    )
    
    # Override stage configurations if provided
    if "stage_overrides" in config:
        for stage_id, overrides in config["stage_overrides"].items():
            if stage_id in manager.STAGE_CONFIGS:
                current_config = manager.STAGE_CONFIGS[stage_id]
                # Update specific fields
                for field, value in overrides.items():
                    if hasattr(current_config, field):
                        setattr(current_config, field, value)
    
    return manager