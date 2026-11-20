"""Curriculum learning scheduler for PhysRoPE training stages."""

import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Phase:
    """Definition of a training phase."""
    name: str
    tasks: Dict[str, float]  # task_name -> probability
    target_loss: float
    max_steps: Optional[int] = None
    lr_scale: float = 1.0  # Learning rate multiplier
    warmup_steps: int = 1000
    description: str = ""


class CurriculumScheduler:
    """Manages curriculum progression through training phases."""
    
    PHASE_DEFINITIONS = [
        Phase(
            name="L0",
            tasks={"t2v": 1.0},
            target_loss=1.5,
            max_steps=50000,
            lr_scale=1.0,
            description="Pure multishot baseline (L0)",
        ),
        Phase(
            name="L0+L1",
            tasks={"t2v": 0.8, "image_anchor": 0.2},
            target_loss=1.2,
            max_steps=30000,
            lr_scale=0.5,
            description="Add image anchor binding (L1)",
        ),
        Phase(
            name="L0+L1+L2",
            tasks={"t2v": 0.6, "image_anchor": 0.2, "l2_video": 0.2},
            target_loss=1.1,
            max_steps=30000,
            lr_scale=0.5,
            description="Add first/last frame anchors (L2)",
        ),
        Phase(
            name="L0+L1+L2+L3",
            tasks={"t2v": 0.5, "l3_condition": 0.5},
            target_loss=1.0,
            max_steps=40000,
            lr_scale=0.3,
            description="Add clip insertion (L3 equal-length)",
        ),
        Phase(
            name="L0+L1+L2+L3+L4",
            tasks={"t2v": 0.4, "l3_condition": 0.4, "l4_resample": 0.2},
            target_loss=0.9,
            max_steps=100000,
            lr_scale=0.2,
            description="Add temporal resampling (L4)",
        ),
    ]
    
    def __init__(
        self,
        start_phase: int = 0,
        enable_curriculum: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            start_phase: Starting phase index (0=L0 only)
            enable_curriculum: If False, skip all phases and use final config
            strict_mode: If True, require target loss before progression
        """
        self.current_phase = start_phase
        self.enable_curriculum = enable_curriculum
        self.strict_mode = strict_mode
        self.phase_start_step = 0
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0
        
        logger.info(f"Initialized CurriculumScheduler at phase {start_phase}")
        logger.info(f"Curriculum enabled: {enable_curriculum}")
    
    def get_current_phase(self) -> Phase:
        """Get current training phase definition."""
        return self.PHASE_DEFINITIONS[self.current_phase]
    
    def get_task_probabilities(self) -> Dict[str, float]:
        """Get task probability distribution for current phase."""
        if not self.enable_curriculum:
            return self.PHASE_DEFINITIONS[-1].tasks
        
        phase = self.get_current_phase()
        return phase.tasks
    
    def get_lr_scale(self) -> float:
        """Get learning rate scale for current phase."""
        if not self.enable_curriculum:
            return 1.0
        
        phase = self.get_current_phase()
        return phase.lr_scale
    
    def should_transition(
        self,
        current_step: int,
        val_loss: float,
    ) -> bool:
        """
        Check if should transition to next phase.
        
        Args:
            current_step: Current training step
            val_loss: Latest validation loss
            
        Returns:
            True if should transition to next phase
        """
        if not self.enable_curriculum:
            return False
        
        if self.current_phase >= len(self.PHASE_DEFINITIONS) - 1:
            return False  # Already at final phase
        
        phase = self.get_current_phase()
        
        # Check loss criterion
        loss_criterion_met = val_loss < phase.target_loss
        
        # Check step limit
        steps_in_phase = current_step - self.phase_start_step
        step_criterion_met = (
            phase.max_steps is not None and
            steps_in_phase >= phase.max_steps
        )
        
        # Strict mode: require loss criterion
        if self.strict_mode:
            should_progress = loss_criterion_met
        else:
            # Either loss criterion OR step limit
            should_progress = loss_criterion_met or step_criterion_met
        
        return should_progress
    
    def transition_to_next_phase(self, current_step: int):
        """Transition to next phase."""
        self.current_phase = min(
            self.current_phase + 1,
            len(self.PHASE_DEFINITIONS) - 1
        )
        self.phase_start_step = current_step
        self.steps_since_improvement = 0
        
        next_phase = self.get_current_phase()
        logger.info(f"Transitioned to phase {self.current_phase}: {next_phase.name}")
        logger.info(f"  Description: {next_phase.description}")
        logger.info(f"  Task distribution: {next_phase.tasks}")
        logger.info(f"  LR scale: {next_phase.lr_scale}")
        logger.info(f"  Target loss: {next_phase.target_loss}")
    
    def update_validation_loss(self, val_loss: float):
        """Track validation loss improvements."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
    
    def get_config_dict(self) -> Dict:
        """
        Get configuration dict for multitask training.
        Ready to be JSON-serialized for multitask_mask_training_args.
        """
        task_probs = self.get_task_probabilities()
        
        config = {}
        if "t2v" in task_probs:
            config["t2v"] = {"prob": task_probs["t2v"]}
        if "image_anchor" in task_probs:
            config["image_anchor"] = {"prob": task_probs["image_anchor"]}
        if "l2_video" in task_probs:
            config["l2_video"] = {"prob": task_probs["l2_video"]}
        if "l3_condition" in task_probs:
            config["l3_condition"] = {"prob": task_probs["l3_condition"]}
        if "l4_resample" in task_probs:
            config["l4_resample"] = {"prob": task_probs["l4_resample"]}
        
        return config
    
    def get_config_json(self) -> str:
        """Get JSON string for multitask_mask_training_args."""
        return json.dumps(self.get_config_dict())
    
    def save_checkpoint(self, path: str):
        """Save curriculum state for resuming."""
        checkpoint = {
            'current_phase': self.current_phase,
            'phase_start_step': self.phase_start_step,
            'best_val_loss': self.best_val_loss,
            'steps_since_improvement': self.steps_since_improvement,
        }
        import json
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Saved curriculum checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load curriculum state for resuming."""
        import json
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.current_phase = checkpoint['current_phase']
        self.phase_start_step = checkpoint['phase_start_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.steps_since_improvement = checkpoint['steps_since_improvement']
        
        logger.info(f"Loaded curriculum checkpoint from {path}")
        logger.info(f"Resumed at phase {self.current_phase}")
    
    def get_summary(self) -> str:
        """Get human-readable summary of curriculum state."""
        phase = self.get_current_phase()
        tasks = self.get_task_probabilities()
        
        summary = f"""
=== Curriculum Learning Status ===
Current Phase: {self.current_phase}/{len(self.PHASE_DEFINITIONS)-1}
Phase Name: {phase.name}
Description: {phase.description}

Task Probabilities:
  {json.dumps(tasks, indent=2)}

Target Loss: {phase.target_loss}
Current LR Scale: {self.get_lr_scale()}
Best Val Loss: {self.best_val_loss:.4f}
Steps Since Improvement: {self.steps_since_improvement}
"""
        return summary.strip()
