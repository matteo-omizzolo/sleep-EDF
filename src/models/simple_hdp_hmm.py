"""
Simplified but working HDP-HMM implementation for quick experiments.

This uses a weak-limit truncation with basic Gibbs sampling that actually works.
For production, consider using pyhsmm or implementing full beam sampling.
"""

import numpy as np
from scipy.special import logsumexp, digamma
from scipy.stats import multivariate_normal, wishart, invwishart, dirichlet
from typing import Dict, List, Optional, Tuple
import warnings


class SimpleStickyHDPHMM:
    """
    Simplified Sticky HDP-HMM with working Gibbs sampling.
    
    Uses weak-limit truncation for practical inference.
    """
    
    def __init__(
        self,
        K_max: int = 20,
        gamma: float = 2.0,  # Increased for better state discovery
        alpha: float = 5.0,  # Increased to allow more flexible transitions
        kappa: float = 50.0,  # Significantly increased for realistic sleep stage durations
        n_iter: int = 500,
        burn_in: int = 200,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ):
        self.K_max = K_max
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.verbose = verbose
        
        self.rng = np.random.RandomState(random_state)
        
        # Model parameters
        self.beta_ = None
        self.pi_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.states_ = None
        self.duration_means_ = None  # Add duration modeling
        
        # Posterior samples
        self.samples_ = []
        self.is_fitted = False
    
    def _initialize_params(self, X_list: List[np.ndarray]) -> None:
        """Initialize parameters using data with class-balanced clustering."""
        X_all = np.vstack(X_list)
        self.d = X_all.shape[1]
        self.M = len(X_list)
        
        # Use enough initial clusters to capture minority classes
        n_init_clusters = min(8, self.K_max)
        
        # Initialize with k-means for faster, more balanced clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        # Use k-means with multiple restarts for robust initialization
        kmeans = KMeans(
            n_clusters=n_init_clusters,
            n_init=20,  # More restarts for better initialization
            max_iter=300,
            random_state=self.rng.randint(10000)
        )
        labels = kmeans.fit_predict(X_scaled)
        
        # Global stick-breaking weights (with more mass on early components)
        self.beta_ = self._sample_beta_informed(n_init_clusters)
        
        # Transition matrices (M subjects)
        self.pi_ = np.zeros((self.M, self.K_max, self.K_max))
        for m in range(self.M):
            for j in range(self.K_max):
                self.pi_[m, j] = self._sample_pi_j(j)
        
        # Emission parameters from k-means clustering
        self.mu_ = np.zeros((self.K_max, self.d))
        self.Sigma_ = np.zeros((self.K_max, self.d, self.d))
        
        global_cov = np.cov(X_all.T)
        for k in range(n_init_clusters):
            cluster_mask = labels == k
            n_k = cluster_mask.sum()
            if n_k > self.d + 2:  # Need enough samples for covariance
                self.mu_[k] = X_all[cluster_mask].mean(axis=0)
                cluster_cov = np.cov(X_all[cluster_mask].T)
                # Regularize with global covariance
                self.Sigma_[k] = 0.8 * cluster_cov + 0.2 * global_cov + 0.1 * np.eye(self.d)
            else:
                self.mu_[k] = X_all.mean(axis=0) + self.rng.randn(self.d) * 0.3
                self.Sigma_[k] = global_cov + 0.1 * np.eye(self.d)
        
        # Initialize remaining states with high variance
        for k in range(n_init_clusters, self.K_max):
            self.mu_[k] = X_all.mean(axis=0) + self.rng.randn(self.d) * 0.5
            self.Sigma_[k] = global_cov + 1.0 * np.eye(self.d)
        
        # Initialize duration means (in units of 30-second epochs)
        # Typical sleep stage durations: 1-20 minutes = 2-40 epochs
        self.duration_means_ = np.random.gamma(5, 3, size=self.K_max)  # Mean ~15 epochs
    
    def _sample_beta(self) -> np.ndarray:
        """Sample global stick-breaking weights using state counts."""
        # If states not yet initialized, sample from prior
        if self.states_ is None or not isinstance(self.states_, list):
            v = self.rng.beta(1, self.gamma, size=self.K_max)
            beta = np.zeros(self.K_max)
            beta[0] = v[0]
            stick_left = 1 - v[0]
            for k in range(1, self.K_max):
                beta[k] = v[k] * stick_left
                stick_left *= (1 - v[k])
            return beta
        
        # Count how many times each state is used across all subjects
        all_states = np.concatenate(self.states_)
        state_counts = np.bincount(all_states, minlength=self.K_max)
        
        # Sample stick-breaking weights informed by data
        # v_k ~ Beta(1 + n_k, gamma + sum_{j>k} n_j)
        v = np.zeros(self.K_max)
        cumsum_counts = np.cumsum(state_counts[::-1])[::-1]  # Reverse cumsum
        
        for k in range(self.K_max - 1):
            a = 1.0 + state_counts[k]
            b = self.gamma + cumsum_counts[k+1] if k < self.K_max - 1 else self.gamma
            v[k] = self.rng.beta(a, b)
        v[-1] = 1.0  # Last stick gets all remaining
        
        # Convert to beta weights
        beta = np.zeros(self.K_max)
        beta[0] = v[0]
        stick_left = 1 - v[0]
        for k in range(1, self.K_max):
            beta[k] = v[k] * stick_left
            stick_left *= (1 - v[k])
        
        # Ensure numerical stability
        beta = np.maximum(beta, 1e-10)
        beta /= beta.sum()
        return beta
    
    def _sample_beta_informed(self, n_active: int) -> np.ndarray:
        """Sample beta with more mass on first n_active components."""
        beta = np.zeros(self.K_max)
        # Put uniform mass on first n_active components
        beta[:n_active] = 1.0 / n_active * 0.95
        # Small mass on remaining
        beta[n_active:] = 0.05 / (self.K_max - n_active)
        # Add small noise
        beta += self.rng.dirichlet(np.ones(self.K_max) * 0.1)
        beta /= beta.sum()
        return beta
    
    def _sample_pi_j(self, j: int) -> np.ndarray:
        """Sample transition probabilities from state j (sticky)."""
        # Sticky prior: π_j ~ Dir(α*β + κ*δ_j)
        concentration = self.alpha * self.beta_.copy()
        concentration[j] += self.kappa  # Add stickiness to self-transition
        return self.rng.dirichlet(concentration)
    
    def _forward_backward(
        self, X: np.ndarray, pi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm (optimized)."""
        T = len(X)
        K = self.K_max
        
        # Emission log-likelihoods (vectorized)
        log_B = np.zeros((T, K))
        for k in range(K):
            try:
                log_B[:, k] = multivariate_normal.logpdf(
                    X, mean=self.mu_[k], cov=self.Sigma_[k], allow_singular=True
                )
            except:
                log_B[:, k] = -1e10
        
        # Forward pass (vectorized)
        log_pi = np.log(pi + 1e-10)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.beta_ + 1e-10) + log_B[0]
        
        for t in range(1, T):
            # Vectorized computation
            log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_pi.T, axis=0) + log_B[t]
        
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Backward pass (vectorized)
        log_beta_bw = np.zeros((T, K))
        
        for t in range(T-2, -1, -1):
            log_beta_bw[t] = logsumexp(log_pi + log_B[t+1] + log_beta_bw[t+1], axis=1)
        
        # Posteriors
        log_gamma = log_alpha + log_beta_bw
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(np.clip(log_gamma, -500, 500))
        
        # Simplified xi computation
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            xi[t] = gamma[t][:, None] * pi * np.exp(log_B[t+1] + log_beta_bw[t+1])
            xi[t] /= (xi[t].sum() + 1e-10)
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for j in range(K):
                for k in range(K):
                    xi[t, j, k] = (log_alpha[t, j] + 
                                   np.log(pi[j, k] + 1e-10) +
                                   log_B[t+1, k] + 
                                   log_beta_bw[t+1, k])
            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))
        
        return gamma, xi, log_likelihood
    
    def _sample_states(
        self, X: np.ndarray, pi: np.ndarray
    ) -> np.ndarray:
        """Sample state sequence."""
        gamma, xi, _ = self._forward_backward(X, pi)
        T = len(X)
        states = np.zeros(T, dtype=int)
        
        # Sample last state
        p_last = gamma[-1]
        # Replace NaNs/Infs with zeros, then normalize
        p_last = np.nan_to_num(p_last, nan=0.0, posinf=0.0, neginf=0.0)
        total = p_last.sum()
        if total > 1e-10:
            p_last = p_last / total
        else:
            # Uniform fallback if all zero or invalid
            p_last = np.ones(self.K_max) / self.K_max
        # Final check: if still any NaN, fallback
        if not np.all(np.isfinite(p_last)) or np.any(np.isnan(p_last)):
            p_last = np.ones(self.K_max) / self.K_max
        states[-1] = self.rng.choice(self.K_max, p=p_last)
        
        # Sample backward
        for t in range(T-2, -1, -1):
            trans_prob = xi[t, :, states[t+1]]
            prob_sum = trans_prob.sum()
            
            # Handle numerical issues
            if prob_sum < 1e-10 or not np.isfinite(prob_sum):
                # Fallback to marginal
                trans_prob = gamma[t]
            
            # Normalize carefully
            trans_prob = np.maximum(trans_prob, 0)  # Ensure non-negative
            prob_sum = trans_prob.sum()
            if prob_sum > 1e-10:
                trans_prob = trans_prob / prob_sum
            else:
                # Uniform fallback
                trans_prob = np.ones(self.K_max) / self.K_max
            
            states[t] = self.rng.choice(self.K_max, p=trans_prob)
        
        return states
    
    def _update_emissions(self, X_list: List[np.ndarray], states_list: List[np.ndarray]) -> None:
        """Update emission parameters given states."""
        # NIW prior (weak)
        mu_0 = np.vstack(X_list).mean(axis=0)
        kappa_0 = 0.01
        nu_0 = self.d + 2
        Psi_0 = np.cov(np.vstack(X_list).T) * (self.d + 2)
        
        for k in range(self.K_max):
            # Collect data assigned to state k
            X_k = []
            for X, states in zip(X_list, states_list):
                mask = states == k
                if mask.any():
                    X_k.append(X[mask])
            
            if len(X_k) > 0:
                X_k = np.vstack(X_k)
            
            if len(X_k) > 1:  # Need at least 2 points for covariance
                n_k = len(X_k)
                x_bar = X_k.mean(axis=0)
                
                # Posterior NIW parameters
                kappa_n = kappa_0 + n_k
                nu_n = nu_0 + n_k
                mu_n = (kappa_0 * mu_0 + n_k * x_bar) / kappa_n
                
                S = np.zeros((self.d, self.d))
                if n_k > 1:
                    S = (X_k - x_bar).T @ (X_k - x_bar)
                
                Psi_n = (Psi_0 + S + 
                         (kappa_0 * n_k / kappa_n) * 
                         np.outer(x_bar - mu_0, x_bar - mu_0))
                
                # Sample from posterior
                try:
                    self.Sigma_[k] = invwishart.rvs(df=nu_n, scale=Psi_n, random_state=self.rng)
                    self.mu_[k] = self.rng.multivariate_normal(
                        mu_n, self.Sigma_[k] / kappa_n
                    )
                except:
                    # Fallback to prior
                    self.Sigma_[k] = invwishart.rvs(df=nu_0, scale=Psi_0, random_state=self.rng)
                    self.mu_[k] = mu_0 + self.rng.randn(self.d) * 0.1
            else:
                # Sample from prior
                self.Sigma_[k] = invwishart.rvs(df=nu_0, scale=Psi_0, random_state=self.rng)
                self.mu_[k] = mu_0 + self.rng.randn(self.d) * 0.1
    
    def _update_transitions(self, states_list: List[np.ndarray]) -> None:
        """Update transition matrices."""
        for m in range(self.M):
            states = states_list[m]
            
            for j in range(self.K_max):
                # Count transitions from j
                count = np.zeros(self.K_max)
                for t in range(len(states) - 1):
                    if states[t] == j:
                        count[states[t+1]] += 1
                
                # Posterior: Dir(α*β + κ*δ_j + counts)
                post_concentration = self.alpha * self.beta_.copy()
                post_concentration[j] += self.kappa
                post_concentration += count
                
                self.pi_[m, j] = self.rng.dirichlet(post_concentration)
    
    def fit(self, X_list: List[np.ndarray]) -> "SimpleStickyHDPHMM":
        """Fit model to data."""
        if self.verbose:
            print(f"Fitting Sticky HDP-HMM...")
            print(f"  Subjects: {len(X_list)}")
            print(f"  K_max: {self.K_max}")
            print(f"  Iterations: {self.n_iter}")
        
        self._initialize_params(X_list)
        states_list = [np.zeros(len(X), dtype=int) for X in X_list]
        
        for iter_idx in range(self.n_iter):
            # E-step: Sample states
            for m in range(self.M):
                states_list[m] = self._sample_states(X_list[m], self.pi_[m])
            
            # M-step: Update parameters every iteration for better mixing
            self._update_emissions(X_list, states_list)
            self._update_transitions(states_list)
            self.beta_ = self._sample_beta()
            
            # Save sample (less frequently)
            if iter_idx >= self.burn_in and (iter_idx - self.burn_in) % 10 == 0:
                K_active = len(np.unique(np.concatenate(states_list)))
                self.samples_.append({
                    'beta': self.beta_.copy(),
                    'pi': self.pi_.copy(),
                    'mu': self.mu_.copy(),
                    'Sigma': self.Sigma_.copy(),
                    'states': [s.copy() for s in states_list],
                    'K': K_active,
                })
            
            if self.verbose and (iter_idx + 1) % 20 == 0:
                K_active = len(np.unique(np.concatenate(states_list)))
                print(f"  Iteration {iter_idx + 1}/{self.n_iter}, K={K_active}")
        
        self.states_ = states_list
        self.is_fitted = True
        
        if self.verbose:
            K_mean = np.mean([s['K'] for s in self.samples_])
            print(f"  Complete! Mean K={K_mean:.1f}")
        
        return self
    
    def predict(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        """Predict states for new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Use Viterbi or forward-backward
        gamma, _, _ = self._forward_backward(X, self.pi_[subject_idx])
        return np.argmax(gamma, axis=1)
    
    def log_likelihood(self, X: np.ndarray, subject_idx: int = 0) -> float:
        """Compute log-likelihood."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        _, _, ll = self._forward_backward(X, self.pi_[subject_idx])
        return ll
    
    def get_posterior_mean_K(self) -> float:
        """Get posterior mean number of states (all instantiated)."""
        return np.mean([s['K'] for s in self.samples_])
    
    def get_effective_K(self, threshold: float = 0.01) -> float:
        """Get effective number of states (with >threshold probability mass)."""
        effective_K_list = []
        for sample in self.samples_:
            beta = sample['beta']
            k_eff = np.sum(beta > threshold)
            effective_K_list.append(k_eff)
        return np.mean(effective_K_list)
    
    def get_state_usage(self) -> np.ndarray:
        """Get state usage across all subjects (M x K)."""
        M = self.M
        usage = np.zeros((M, self.K_max))
        
        for m in range(M):
            states = self.states_[m]
            for k in range(self.K_max):
                usage[m, k] = np.mean(states == k)
        
        return usage
