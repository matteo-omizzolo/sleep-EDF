"""
Base class for HMM models.

Defines common interface for all HMM implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseHMM(ABC):
    """Base class for Hidden Markov Models."""
    
    def __init__(self, **kwargs):
        """Initialize model with hyperparameters."""
        self.hyperparameters = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        X: List[np.ndarray],
        y: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> "BaseHMM":
        """
        Fit model to data.
        
        Args:
            X: List of feature matrices (one per subject)
               Each matrix has shape (n_epochs, n_features)
            y: Optional list of labels (for supervised/semi-supervised)
            **kwargs: Additional fitting arguments
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        method: str = "viterbi"
    ) -> np.ndarray:
        """
        Predict state sequence for new data.
        
        Args:
            X: Feature matrix (n_epochs, n_features)
            method: Decoding method ("viterbi" or "posterior")
        
        Returns:
            State sequence (n_epochs,)
        """
        pass
    
    @abstractmethod
    def log_likelihood(
        self,
        X: np.ndarray
    ) -> float:
        """
        Compute log-likelihood of observed sequence.
        
        Args:
            X: Feature matrix (n_epochs, n_features)
        
        Returns:
            Log-likelihood
        """
        pass
    
    @abstractmethod
    def sample_posterior(
        self,
        n_samples: int = 1
    ) -> List[Dict]:
        """
        Sample from posterior distribution.
        
        Args:
            n_samples: Number of posterior samples
        
        Returns:
            List of posterior samples (dicts with parameters)
        """
        pass
    
    def get_num_states(self) -> int:
        """
        Get current number of active states.
        
        Returns:
            Number of states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.K_
    
    def get_transition_matrix(
        self,
        subject_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get transition probability matrix.
        
        Args:
            subject_idx: Subject index (for hierarchical models)
        
        Returns:
            Transition matrix (K, K)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.transition_matrix_
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Output file path
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> "BaseHMM":
        """
        Load model from file.
        
        Args:
            filepath: Input file path
        
        Returns:
            Loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
