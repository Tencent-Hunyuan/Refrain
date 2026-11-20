"""
Anomaly tracking utilities for training stability monitoring.

This module provides the AnomalyTracker class for detecting and logging
various training anomalies across distributed training ranks.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional


class AnomalyTracker:
    """Track and log training anomalies across all ranks"""
    
    def __init__(self, rank: int, world_size: int, logger: logging.Logger):
        """
        Initialize anomaly tracker for a specific rank.
        
        Args:
            rank: Current process rank
            world_size: Total number of processes
            logger: Logger instance for output
        """
        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        self.anomaly_counts = defaultdict(int)
        self.last_anomaly_step = defaultdict(int)
        self.anomaly_history = []
        self.max_history = 100  # Will be updated from config
        
    def log_anomaly(self, anomaly_type: str, step: int, details: Dict[str, Any]) -> None:
        """Log an anomaly with detailed information"""
        self.anomaly_counts[anomaly_type] += 1
        self.last_anomaly_step[anomaly_type] = step
        
        # Add to history
        self.anomaly_history.append({
            'type': anomaly_type,
            'step': step,
            'details': details,
            'rank': self.rank
        })
        
        # Log immediately
        print(f"[ANOMALY-RANK-{self.rank}] {anomaly_type} at step {step}: {details}")
        
        # Keep only recent history (configurable limit)
        max_history = getattr(self, 'max_history', 100)
        if len(self.anomaly_history) > max_history:
            self.anomaly_history = self.anomaly_history[-max_history:]
    
    def log_loss_spike(self, step: int, loss_value: float, threshold: float, 
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log a loss spike anomaly"""
        details = {
            'loss_value': loss_value,
            'threshold': threshold
        }
        if additional_info:
            details.update(additional_info)
        
        self.log_anomaly('LOSS_SPIKE', step, details)
    
    def log_grad_spike(self, step: int, grad_norm: float, threshold: float, 
                       top_params: Optional[List] = None) -> None:
        """Log a gradient norm spike anomaly"""
        details = {
            'grad_norm': grad_norm,
            'threshold': threshold
        }
        if top_params:
            details['top_contributing_params'] = top_params[:5]  # Top 5 parameters
        
        self.log_anomaly('GRAD_SPIKE', step, details)
    
    def log_nan_inf(self, step: int, value_type: str, 
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log NaN or Inf detection"""
        details = {
            'value_type': value_type,  # 'loss', 'gradient_norm', etc.
        }
        if additional_info:
            details.update(additional_info)
        
        self.log_anomaly('NAN_INF', step, details)
    
    def should_log_warning(self, warning_type: str, step: int, min_interval: int = 50) -> bool:
        """
        Check if a warning should be logged based on frequency control.
        
        Args:
            warning_type: Type of warning
            step: Current training step
            min_interval: Minimum steps between same warning type
            
        Returns:
            True if warning should be logged
        """
        last_step = self.last_anomaly_step.get(warning_type, -min_interval)
        return step - last_step >= min_interval
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all anomalies detected by this rank"""
        return {
            'rank': self.rank,
            'total_anomalies': len(self.anomaly_history),
            'by_type': dict(self.anomaly_counts),
            'recent_steps': [h['step'] for h in self.anomaly_history[-10:]],
            'recent_types': [h['type'] for h in self.anomaly_history[-10:]]
        }
    
    def get_recent_anomalies(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent anomalies"""
        return self.anomaly_history[-count:] if self.anomaly_history else []
    
    def reset_history(self) -> None:
        """Reset anomaly history (useful for checkpointing)"""
        self.anomaly_history.clear()
        self.anomaly_counts.clear()
        self.last_anomaly_step.clear() 