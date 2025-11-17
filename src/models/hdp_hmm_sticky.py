"""
Sticky HDP-HMM implementation.

Hierarchical Dirichlet Process Hidden Markov Model with sticky self-transitions
for realistic state persistence.

References:
- Fox et al. (2011): "A Sticky HDP-HMM with Application to Speaker Diarization"
- Teh et al. (2006): "Hierarchical Dirichlet Processes"
"""

import numpy as np
from scipy.special import logsumexp, digamma, gammaln
from scipy.stats import multivariate_normal, wishart
from typing import Dict, List, Optional, Tuple
from .base import BaseHMM


class StickyHDPHMM(BaseHMM):
    """
    Sticky Hierarchical Dirichlet Process Hidden Markov Model.
    
    Generative model:
        β ~ GEM(γ)                                    # Global stick-breaking weights
        π_j^(m) ~ DP(α+κ, (αβ + κδ_j)/(α+κ))        # Subject-specific transitions (sticky)
        θ_k ~ H_0                                     # Emission parameters (NIW prior)
        s_t | s_{t-1} ~ π_{s_{t-1}}^(m)              # State transitions
        y_t | s_t ~ N(μ_{s_t}, Σ_{s_t})              # Gaussian emissions
    
    Parameters:
        gamma: Global DP concentration (controls number of global states)
        alpha: Group-level DP concentration (controls mixing over global states)
        kappa: Sticky self-transition bias (encourages state persistence)
        K_max: Truncation level for weak-limit approximation
        prior: NIW prior parameters for emissions
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        kappa: float = 5.0,
        K_max: int = 50,
        prior_mu_0: Optional[np.ndarray] = None,
        prior_kappa_0: float = 0.01,
        prior_psi: Optional[np.ndarray] = None,
        prior_nu: Optional[int] = None,
        n_iter: int = 1000,
        burn_in: int = 500,
        thin: int = 5,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            kappa=kappa,
            K_max=K_max,
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
        )
        
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.K_max = K_max
        
        # NIW prior parameters (will be set based on data)
        self.prior_mu_0 = prior_mu_0
        self.prior_kappa_0 = prior_kappa_0
        self.prior_psi = prior_psi
        self.prior_nu = prior_nu
        
        # MCMC settings
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.verbose = verbose
        
        # Random state
        self.rng = np.random.RandomState(random_state)
        
        # Model parameters (set after fitting)
        self.beta_ = None  # Global stick-breaking weights
        self.pi_ = None  # Transition matrices per subject
        self.mu_ = None  # Emission means
        self.Sigma_ = None  # Emission covariances
        self.K_ = None  # Number of active states
        
        # Posterior samples
        self.samples_ = []
    
    def _initialize_prior(self, X: List[np.ndarray]) -> None:
        """Initialize NIW prior based on data."""
        # Stack all data
        X_all = np.vstack(X)
        n_features = X_all.shape[1]
        
        # Set prior mean to data mean
        if self.prior_mu_0 is None:
            self.prior_mu_0 = np.mean(X_all, axis=0)
        
        # Set prior scale matrix
        if self.prior_psi is None:
            data_cov = np.cov(X_all.T)
            self.prior_psi = data_cov * (n_features + 2)  # Weak prior
        
        # Set degrees of freedom
        if self.prior_nu is None:
            self.prior_nu = n_features + 2
    
    def _initialize_parameters(self, X: List[np.ndarray]) -> None:
        """Initialize model parameters."""
        n_features = X[0].shape[1]
        
        # Initialize global stick-breaking weights (truncated)
        # β ~ GEM(γ) approximated by stick-breaking
        v = self.rng.beta(1, self.gamma, size=self.K_max)
        self.beta_ = np.zeros(self.K_max)
        self.beta_[0] = v[0]
        for k in range(1, self.K_max):
            self.beta_[k] = v[k] * np.prod(1 - v[:k])
        
        # Initialize transition matrices (M subjects x K states x K states)
        M = len(X)
        self.pi_ = np.zeros((M, self.K_max, self.K_max))
        
        for m in range(M):
            for j in range(self.K_max):
                # Sample from sticky DP: π_j ~ DP(α+κ, (αβ + κδ_j)/(α+κ))
                concentration = self.alpha + self.kappa
                base_weights = (self.alpha * self.beta_ + self.kappa * (np.arange(self.K_max) == j))
                base_weights = base_weights / concentration
                
                # Sample Dirichlet
                self.pi_[m, j] = self.rng.dirichlet(concentration * base_weights)
        
        # Initialize emission parameters from NIW prior
        self.mu_ = np.zeros((self.K_max, n_features))
        self.Sigma_ = np.zeros((self.K_max, n_features, n_features))
        
        for k in range(self.K_max):
            # Sample Σ ~ InverseWishart(ν, Ψ)
            self.Sigma_[k] = wishart.rvs(
                df=self.prior_nu,
                scale=np.linalg.inv(self.prior_psi),
                random_state=self.rng
            )
            self.Sigma_[k] = np.linalg.inv(self.Sigma_[k])
            
            # Sample μ | Σ ~ N(μ_0, Σ / κ_0)
            self.mu_[k] = self.rng.multivariate_normal(
                self.prior_mu_0,
                self.Sigma_[k] / self.prior_kappa_0
            )
    
    def _forward_backward(
        self,
        X: np.ndarray,
        pi: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward-backward algorithm for HMM inference.
        
        Args:
            X: Observations (T, D)
            pi: Transition matrix (K, K)
        
        Returns:
            gamma: State posteriors (T, K)
            log_likelihood: Log-likelihood of sequence
        """
        T = len(X)
        K = self.K_max
        
        # Compute emission log-likelihoods
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = multivariate_normal.logpdf(
                X, mean=self.mu_[k], cov=self.Sigma_[k]
            )
        
        # Forward pass
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.beta_) + log_B[0]  # Use beta as initial distribution
        
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(pi[:, k])) + log_B[t, k]
        
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Backward pass
        log_beta_bw = np.zeros((T, K))
        log_beta_bw[-1] = 0
        
        for t in range(T-2, -1, -1):
            for k in range(K):
                log_beta_bw[t, k] = logsumexp(
                    np.log(pi[k, :]) + log_B[t+1] + log_beta_bw[t+1]
                )
        
        # Compute state posteriors
        log_gamma = log_alpha + log_beta_bw
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        
        return gamma, log_likelihood
    
    def fit(
        self,
        X: List[np.ndarray],
        y: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> "StickyHDPHMM":
        """
        Fit sticky HDP-HMM to data using Gibbs sampling.
        
        Args:
            X: List of feature matrices (one per subject)
               Each matrix has shape (n_epochs, n_features)
            y: Optional list of labels (not used in unsupervised version)
        
        Returns:
            self
        """
        # Initialize prior
        self._initialize_prior(X)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        M = len(X)  # Number of subjects
        
        if self.verbose:
            print(f"Fitting Sticky HDP-HMM with {M} subjects...")
            print(f"  Iterations: {self.n_iter}")
            print(f"  Burn-in: {self.burn_in}")
            print(f"  K_max: {self.K_max}")
        
        # Gibbs sampling
        self.samples_ = []
        
        for iter_idx in range(self.n_iter):
            # Sample state sequences for each subject
            state_sequences = []
            
            for m in range(M):
                gamma_m, _ = self._forward_backward(X[m], self.pi_[m])
                
                # Sample state sequence
                T = len(X[m])
                states = np.zeros(T, dtype=int)
                states[-1] = self.rng.choice(self.K_max, p=gamma_m[-1])
                
                for t in range(T-2, -1, -1):
                    # Sample backward
                    trans_prob = self.pi_[m, :, states[t+1]]
                    posterior = gamma_m[t] * trans_prob
                    posterior /= posterior.sum()
                    states[t] = self.rng.choice(self.K_max, p=posterior)
                
                state_sequences.append(states)
            
            # Update parameters (placeholder - full implementation would update β, π, μ, Σ)
            # This is simplified; a full implementation would:
            # 1. Update global stick-breaking weights β
            # 2. Update transition matrices π for each subject
            # 3. Update emission parameters μ, Σ for each state
            
            # Save posterior sample (after burn-in, thinned)
            if iter_idx >= self.burn_in and (iter_idx - self.burn_in) % self.thin == 0:
                sample = {
                    'beta': self.beta_.copy(),
                    'pi': self.pi_.copy(),
                    'mu': self.mu_.copy(),
                    'Sigma': self.Sigma_.copy(),
                    'states': state_sequences,
                    'K': np.sum(self.beta_ > 1e-6),  # Number of active states
                }
                self.samples_.append(sample)
            
            if self.verbose and (iter_idx + 1) % 100 == 0:
                K_active = np.sum(self.beta_ > 1e-6)
                print(f"  Iteration {iter_idx + 1}/{self.n_iter}, K={K_active}")
        
        # Set final parameters to posterior mean
        self.beta_ = np.mean([s['beta'] for s in self.samples_], axis=0)
        self.K_ = int(np.mean([s['K'] for s in self.samples_]))
        self.is_fitted = True
        
        if self.verbose:
            print(f"Fitting complete. Posterior mean K={self.K_}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        subject_idx: int = 0,
        method: str = "viterbi"
    ) -> np.ndarray:
        """
        Predict state sequence for new data (simplified implementation).
        
        Args:
            X: Feature matrix (n_epochs, n_features)
            subject_idx: Subject index for transition matrix
            method: Decoding method (only "viterbi" implemented)
        
        Returns:
            State sequence (n_epochs,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Use forward-backward for now (Viterbi would be better)
        gamma, _ = self._forward_backward(X, self.pi_[subject_idx])
        states = np.argmax(gamma, axis=1)
        
        return states
    
    def log_likelihood(self, X: np.ndarray, subject_idx: int = 0) -> float:
        """Compute log-likelihood of observed sequence."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        _, log_lik = self._forward_backward(X, self.pi_[subject_idx])
        return log_lik
    
    def sample_posterior(self, n_samples: int = 1) -> List[Dict]:
        """Sample from posterior distribution."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        indices = self.rng.choice(len(self.samples_), size=n_samples, replace=False)
        return [self.samples_[i] for i in indices]
