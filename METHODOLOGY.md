# Methodology: Mathematical Formulation and Inference

This document provides the complete mathematical specification of the HDP-HMM and independent DP-HMM models, inference algorithms, and hyperparameter choices.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Model Specifications](#model-specifications)
3. [Inference Algorithms](#inference-algorithms)
4. [Hyperparameter Rationale](#hyperparameter-rationale)
5. [Evaluation Metrics](#evaluation-metrics)

---

## Problem Formulation

### Input Data Structure

We have **M subjects**, each with a time series of observations:

- **Subject m**: $\{x_{m,1}, x_{m,2}, \ldots, x_{m,T_m}\}$
- **Observations**: $x_{m,t} \in \mathbb{R}^D$ (D-dimensional feature vectors)
- **Hidden states**: $z_{m,t} \in \{1, 2, \ldots, K\}$ (latent sleep stages)
- **Ground truth labels** (for evaluation only): $y_{m,t} \in \{0,1,2,3,4\}$ (W, N1, N2, N3, REM)

### Research Questions

1. **State Discovery**: How many latent states does each model discover?
2. **State Sharing**: Do subjects share common states (HDP) or use independent states (iDP)?
3. **Generalization**: Which model better predicts held-out subject data?
4. **Biological Plausibility**: Which discovered states better align with true sleep stages?

---

## Model Specifications

### 1. Independent DP-HMM (Baseline)

Each subject has an **independent** infinite HMM with no sharing across subjects.

#### Generative Model

For **each subject m independently**:

$$
\begin{align}
\beta^{(m)} &\sim \text{GEM}(\gamma) && \text{[Stick-breaking weights]} \\
\pi_j^{(m)} &\sim \text{Dir}(\alpha \beta^{(m)} + \kappa \delta_j) && \text{[Sticky transition from state j]} \\
\theta_k &\sim H && \text{[Emission parameters]} \\
z_{m,1} &\sim \text{Cat}(\beta^{(m)}) && \text{[Initial state]} \\
z_{m,t} | z_{m,t-1} &\sim \text{Cat}(\pi_{z_{m,t-1}}^{(m)}) && \text{[State transitions]} \\
x_{m,t} | z_{m,t} &\sim F(\theta_{z_{m,t}}) && \text{[Observations]}
\end{align}
$$

**Key property**: Each subject discovers **independent** states. No information sharing.

#### Parameters

- **$\gamma$**: DP concentration for state discovery (higher → more states)
- **$\alpha$**: DP concentration for transition flexibility
- **$\kappa$**: Sticky self-transition bias (encourages persistence)
- **$\beta^{(m)}$**: Subject-specific state probabilities (infinite vector)
- **$\pi_j^{(m)}$**: Subject-specific transition matrix from state j
- **$\theta_k = (\mu_k, \Sigma_k)$**: Emission parameters (mean, covariance)

### 2. Sticky HDP-HMM (Target Model)

Hierarchical model with **shared global state distribution** across subjects.

#### Generative Model

**Global level** (shared across all subjects):

$$
\begin{align}
\beta &\sim \text{GEM}(\gamma) && \text{[Global stick-breaking weights]} \\
\theta_k &\sim H && \text{[Shared emission parameters]}
\end{align}
$$

**Subject level** (per subject m):

$$
\begin{align}
\pi_j^{(m)} &\sim \text{Dir}(\alpha \beta + \kappa \delta_j) && \text{[Sticky transitions use global β]} \\
z_{m,1} &\sim \text{Cat}(\beta) && \text{[Initial state from global distribution]} \\
z_{m,t} | z_{m,t-1} &\sim \text{Cat}(\pi_{z_{m,t-1}}^{(m)}) && \text{[Subject-specific transitions]} \\
x_{m,t} | z_{m,t} &\sim F(\theta_{z_{m,t}}) && \text{[Observations from shared emissions]}
\end{align}
$$

**Key property**: **Global $\beta$ shared** → subjects reuse the same states with subject-specific transition dynamics.

#### Hierarchical Structure

```
Level 0 (Global):    G₀ ~ DP(γ, H)          [Discovers K shared sleep stages]
                      ↓
                     β ~ GEM(γ)              [Global state probabilities]
                      ↓
Level 1 (Subjects):  Gₘ ~ DP(α, G₀)         [Subject m's distribution]
                      ↓
                     πⱼ⁽ᵐ⁾ ~ Dir(αβ + κδⱼ)  [Sticky transitions]
                      ↓
Observations:        xₘₜ | zₘₜ ~ F(θ_zₘₜ)  [Emissions]
```

---

## Inference Algorithms

### Weak-Limit Truncation

Both models use **weak-limit approximation** for computational tractability:

- **Truncation level**: $K_{\max} = 12$
- **Stick-breaking truncation**: $\beta = (\beta_1, \ldots, \beta_{K_{\max}})$ with $\sum_{k=1}^{K_{\max}} \beta_k \approx 1$
- **Justification**: Stick-breaking weights decay exponentially → mass beyond $K_{\max}$ negligible

**Theoretical guarantee** (Ishwaran & James 2001):
$$
\|\beta_{\text{truncated}} - \beta_{\text{true}}\|_1 < \epsilon \quad \text{with} \quad K_{\max} = O\left(\frac{\log(1/\epsilon)}{\gamma}\right)
$$

For $\epsilon = 10^{-6}$ and $\gamma = 5.0$: $K_{\max} \geq 12$ ensures negligible error.

### Collapsed Gibbs Sampling

We use **collapsed Gibbs sampling** where transition matrices $\pi$ are marginalized out.

#### Algorithm Overview

**For iteration $t = 1, \ldots, n_{\text{iter}}$:**

1. **Sample states** $z_{1:M}$ given current parameters (forward-backward)
2. **Update emission parameters** $\theta_{1:K}$ given states (conjugate update)
3. **Update global weights** $\beta$ given state counts (stick-breaking)
4. **Update transition matrices** $\pi_{j}^{(m)}$ given transition counts
5. **Store sample** (after burn-in)

#### 1. State Sampling (Forward-Backward Algorithm)

For each subject $m$, sample state sequence $z_m = (z_{m,1}, \ldots, z_{m,T_m})$ using forward-backward algorithm.

**Forward Pass** (compute filtering distributions):

$$
\alpha_t(k) = P(z_{m,t} = k | x_{m,1:t}) \propto \sum_{j=1}^K \alpha_{t-1}(j) \pi_{jk}^{(m)} \cdot p(x_{m,t} | \theta_k)
$$

**Backward Pass** (compute smoothing distributions):

$$
\beta_t(k) = P(x_{m,t+1:T_m} | z_{m,t} = k)
$$

**Posterior**:

$$
\gamma_t(k) = P(z_{m,t} = k | x_{m,1:T_m}) \propto \alpha_t(k) \beta_t(k)
$$

**State sampling**: Sample $z_{m,t} \sim \text{Cat}(\gamma_t)$ backward from $t = T_m$ to $t = 1$.

#### 2. Emission Parameter Update

**Prior**: Normal-Inverse-Wishart (NIW)

$$
\begin{align}
\mu_k | \Sigma_k &\sim \mathcal{N}(\mu_0, \kappa_0^{-1} \Sigma_k) \\
\Sigma_k &\sim \text{InvWishart}(\nu_0, \Psi_0)
\end{align}
$$

**Posterior** (given data assigned to state k):

$$
\begin{align}
\bar{x}_k &= \frac{1}{n_k} \sum_{m,t: z_{m,t}=k} x_{m,t} \\
S_k &= \sum_{m,t: z_{m,t}=k} (x_{m,t} - \bar{x}_k)(x_{m,t} - \bar{x}_k)^\top \\
\\
\mu_k^\text{new} | \Sigma_k &\sim \mathcal{N}(\mu_k', \kappa_k'^{-1} \Sigma_k) \\
\Sigma_k^\text{new} &\sim \text{InvWishart}(\nu_k', \Psi_k')
\end{align}
$$

where:

$$
\begin{align}
\kappa_k' &= \kappa_0 + n_k \\
\mu_k' &= \frac{\kappa_0 \mu_0 + n_k \bar{x}_k}{\kappa_k'} \\
\nu_k' &= \nu_0 + n_k \\
\Psi_k' &= \Psi_0 + S_k + \frac{\kappa_0 n_k}{\kappa_k'}(\bar{x}_k - \mu_0)(\bar{x}_k - \mu_0)^\top
\end{align}
$$

#### 3. Global Weight Update (HDP only)

**Stick-breaking construction**:

$$
\beta_k = v_k \prod_{l=1}^{k-1} (1 - v_l), \quad v_k \sim \text{Beta}(1 + n_k, \gamma + \sum_{j>k} n_j)
$$

where $n_k = \sum_{m,t} \mathbb{1}[z_{m,t} = k]$ is the count of observations in state k across all subjects.

#### 4. Transition Matrix Update

**Posterior** for transitions from state j:

$$
\pi_j^{(m)} \sim \text{Dir}(\alpha \beta + \kappa \delta_j + n_j^{(m)})
$$

where $n_j^{(m)} = (n_{j \to 1}^{(m)}, \ldots, n_{j \to K}^{(m)})$ are transition counts from state j.

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Forward-backward (per subject) | $O(T_m K^2)$ | Dominates computation |
| Emission updates | $O(K D^2)$ | Covariance computation |
| Global weight update | $O(K)$ | Stick-breaking |
| Transition updates | $O(M K^2)$ | All subjects |
| **Total per iteration** | $O(M T K^2 + K D^2)$ | $M$ subjects, $T$ epochs |

**Example**: 5 subjects, $T \approx 2500$ epochs, $K = 12$, $D = 72$
- Per iteration: ~2-3 seconds
- 500 iterations: ~15-20 minutes

---

## Hyperparameter Rationale

### Model Hyperparameters

| Parameter | Value | Interpretation | Rationale |
|-----------|-------|----------------|-----------|
| **$\gamma$** | 5.0 | Global DP concentration | Expected K ≈ 5-8 states (matches 5 true stages) |
| **$\alpha$** | 10.0 | Transition flexibility | Allows flexible data-driven transitions |
| **$\kappa$** | 50.0 | Sticky bias | Self-transition prob ≈ 0.83 (realistic persistence) |
| **$K_{\max}$** | 12 | Truncation level | Sufficient capacity without over-proliferation |

### Derivations

#### Expected Number of States

Under Dirichlet Process with concentration $\gamma$:

$$
\mathbb{E}[K] \approx \gamma \log\left(1 + \frac{N}{\gamma}\right)
$$

For $N \approx 12000$ epochs and $\gamma = 5.0$:

$$
\mathbb{E}[K] \approx 5.0 \times \log\left(1 + \frac{12000}{5.0}\right) \approx 5.0 \times 7.8 \approx 39 \text{ states (too many!)}
$$

**But**: Hierarchical structure and sticky dynamics **constrain** state proliferation → observed K ≈ 5-7.

#### Sticky Self-Transition Bias

Transition from state j to itself has **enhanced concentration**:

$$
\pi_{jj}^{(m)} \sim \text{Dir}(\alpha \beta_j + \kappa, \alpha \beta_{k \neq j})
$$

**Self-transition probability**:

$$
\mathbb{E}[\pi_{jj}^{(m)}] \approx \frac{\alpha \beta_j + \kappa}{\alpha + \kappa} \approx \frac{\kappa}{\alpha + \kappa} = \frac{50}{10 + 50} = 0.83
$$

**Interpretation**: 83% chance of staying in same state → median dwell time:

$$
\text{Dwell time} \approx \frac{1}{1 - 0.83} \times 30\text{s} \approx 6 \text{ epochs} = 180\text{s} = 3 \text{ min}
$$

This matches **realistic sleep stage durations** (1-20 minutes).

### Prior Hyperparameters (NIW)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| **$\mu_0$** | $\mathbf{0}$ | Prior mean (after standardization) |
| **$\kappa_0$** | 0.01 | Weak prior on mean location |
| **$\nu_0$** | $D + 2 = 74$ | Minimal degrees of freedom |
| **$\Psi_0$** | $I_D$ | Identity scale matrix |

**Rationale**: Weakly informative priors → data-driven parameter estimation.

---

## Evaluation Metrics

### Unsupervised Metrics (Clustering Quality)

#### 1. Adjusted Rand Index (ARI)

Measures agreement between predicted clusters and true labels, **adjusted for chance**:

$$
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
$$

- **Range**: $[-1, 1]$ (0 = random, 1 = perfect agreement)
- **Interpretation**: 
  - ARI < 0.2: Poor clustering
  - ARI 0.2-0.4: Weak agreement
  - ARI 0.4-0.6: Moderate agreement
  - ARI > 0.6: Strong agreement

#### 2. Normalized Mutual Information (NMI)

Information-theoretic measure of clustering quality:

$$
\text{NMI} = \frac{2 \cdot I(Y; Z)}{H(Y) + H(Z)}
$$

where $I(Y; Z)$ is mutual information and $H(\cdot)$ is entropy.

- **Range**: $[0, 1]$ (0 = independent, 1 = identical)
- **Advantage**: Less sensitive to cluster imbalance than ARI

### Supervised Metrics (After Hungarian Alignment)

#### 3. Macro F1 Score

Average F1 score across all classes (equal weighting):

$$
\text{Macro-F1} = \frac{1}{5} \sum_{c \in \{W, N1, N2, N3, REM\}} F1_c
$$

where:

$$
F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

**Rationale**: Handles severe class imbalance (N1 = 3%, Wake = 70%).

#### 4. Per-Class F1 Scores

Individual F1 scores for each sleep stage:
- Identifies which stages are easiest/hardest to discover
- Expected difficulty: **N1 < N3 < REM < N2 < Wake**

### Generalization Metrics

#### 5. Test Log-Likelihood

Predictive performance on held-out subject:

$$
\log p(x_{\text{test}} | \text{train}) = \sum_{t=1}^{T_{\text{test}}} \log p(x_t | x_{1:t-1}, \text{train})
$$

**Computation**: Forward algorithm with trained parameters.

**Interpretation**: Higher log-likelihood → better generalization.

### Model Property Metrics

#### 6. Effective Number of States

States with non-negligible probability mass:

$$
K_{\text{eff}} = \#\{k : \beta_k > 0.01\}
$$

**Interpretation**: How many meaningful states does the model discover?

#### 7. Median Dwell Time

Persistence in each state:

$$
\text{Dwell}_k = \text{median}\{\text{consecutive epochs in state } k\} \times 30\text{s}
$$

**Expected**: 1-20 minutes for realistic sleep stages.

---

## References

1. **Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006)**. Hierarchical Dirichlet processes. *Journal of the American Statistical Association*, 101(476), 1566-1581.

2. **Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2008)**. An HDP-HMM for systems with state persistence. *ICML*.

3. **Ishwaran, H., & James, L. F. (2001)**. Gibbs sampling methods for stick-breaking priors. *Journal of the American Statistical Association*, 96(453), 161-173.

4. **Murphy, K. P. (2012)**. Machine Learning: A Probabilistic Perspective. MIT Press. (Chapter 17: HMMs)

---

**Last Updated**: November 2025
