"""
Hybrid RL package: Adaptive switching between model-based and model-free control
"""

from .agent import HybridAgent
from .planner_pets import PETSAgent, EnsembleDynamicsModel
from .gate import SwitchingGate

__all__ = ['HybridAgent', 'PETSAgent', 'EnsembleDynamicsModel', 'SwitchingGate']