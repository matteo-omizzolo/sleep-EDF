"""
Visualization functions for HDP-HMM results.

Generates all 9 figures for the presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_context("talk")  # Larger fonts for presentations
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']


def plot_posterior_num_states(
    hdp_samples: List[Dict],
    idp_samples_per_subject: List[List[Dict]],
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 1: Posterior over number of states K.
    
    Shows HDP-HMM global K vs sum of iDP-HMM per-subject K's.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # HDP-HMM: Global K
    hdp_K = [s['K'] for s in hdp_samples]
    axes[0].hist(hdp_K, bins=range(3, max(hdp_K)+2), alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(hdp_K), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(hdp_K):.1f}')
    axes[0].set_xlabel('Number of Global States (K)')
    axes[0].set_ylabel('Posterior Density')
    axes[0].set_title('HDP-HMM: Global State Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # iDP-HMM: Per-subject K (show distribution of total unique states)
    idp_K_total = []
    for samples in idp_samples_per_subject:
        # Total unique states across all subjects in this sample
        K_sample = sum([s['K'] for s in samples])
        idp_K_total.append(K_sample)
    
    axes[1].hist(idp_K_total, bins=15, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(np.mean(idp_K_total), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(idp_K_total):.1f}')
    axes[1].set_xlabel('Total States Across All Subjects')
    axes[1].set_ylabel('Posterior Density')
    axes[1].set_title('iDP-HMM: Fragmented State Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Posterior over Number of States K', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_state_sharing_heatmap(
    model,
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 2: State-sharing heatmap (subjects × global states).
    
    Shows which subjects use which states.
    """
    usage = model.get_state_usage()  # M x K_max
    K_active = int(model.get_posterior_mean_K())
    
    # Only show active states
    usage_active = usage[:, :K_active]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        usage_active,
        cmap='YlOrRd',
        cbar_kws={'label': 'Usage Probability'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('Global State Index', fontsize=14)
    ax.set_ylabel('Subject Index', fontsize=14)
    ax.set_title('State Sharing Across Subjects (HDP-HMM)', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_dwell_times(
    hdp_samples: List[Dict],
    idp_samples: List[Dict],
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 3: Dwell-time distributions.
    
    Shows sticky HDP-HMM has longer, more realistic dwell times.
    """
    def compute_dwell_times(states_list):
        """Compute segment lengths from state sequences."""
        dwells = []
        for states in states_list:
            current = states[0]
            length = 1
            for s in states[1:]:
                if s == current:
                    length += 1
                else:
                    dwells.append(length)
                    current = s
                    length = 1
            dwells.append(length)
        return np.array(dwells) * 30  # Convert to seconds (30s epochs)
    
    # Collect dwell times
    hdp_dwells = []
    for sample in hdp_samples[-50:]:  # Use last 50 samples
        hdp_dwells.extend(compute_dwell_times(sample['states']))
    
    idp_dwells = []
    for sample in idp_samples[-50:]:
        idp_dwells.extend(compute_dwell_times(sample['states']))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograms
    bins = np.arange(0, 600, 30)
    axes[0].hist(hdp_dwells, bins=bins, alpha=0.7, label='HDP-HMM (sticky)', color='steelblue', density=True)
    axes[0].hist(idp_dwells, bins=bins, alpha=0.7, label='iDP-HMM', color='coral', density=True)
    axes[0].set_xlabel('Dwell Time (seconds)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Dwell Time Distributions')
    axes[0].legend()
    axes[0].set_xlim(0, 600)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_box = [hdp_dwells, idp_dwells]
    bp = axes[1].boxplot(data_box, labels=['HDP-HMM\n(sticky)', 'iDP-HMM'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    axes[1].set_ylabel('Dwell Time (seconds)')
    axes[1].set_title('Median Dwell Times')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add median values as text
    axes[1].text(1, np.median(hdp_dwells), f'{np.median(hdp_dwells):.0f}s', 
                 ha='center', va='bottom', fontweight='bold')
    axes[1].text(2, np.median(idp_dwells), f'{np.median(idp_dwells):.0f}s',
                 ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Sticky Prior Yields Realistic State Persistence', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_predictive_performance(
    loso_results: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 4: Predictive performance (LOSO cross-validation).
    
    Test log-likelihood for held-out subjects.
    """
    hdp_ll = loso_results['hdp_test_ll']
    idp_ll = loso_results['idp_test_ll']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(hdp_ll))
    width = 0.35
    
    ax.bar(x - width/2, hdp_ll, width, label='HDP-HMM (sticky)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, idp_ll, width, label='iDP-HMM', color='coral', alpha=0.8)
    
    ax.set_xlabel('Test Subject', fontsize=14)
    ax.set_ylabel('Test Log-Likelihood', fontsize=14)
    ax.set_title('Generalization Performance (LOSO)', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    ax.axhline(np.mean(hdp_ll), color='darkblue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(np.mean(idp_ll), color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add text annotations
    ax.text(len(hdp_ll)-1, np.mean(hdp_ll), f'Mean: {np.mean(hdp_ll):.1f}', 
            ha='right', va='bottom', color='darkblue', fontweight='bold')
    ax.text(len(hdp_ll)-1, np.mean(idp_ll), f'Mean: {np.mean(idp_ll):.1f}',
            ha='right', va='top', color='darkred', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_label_agreement(
    loso_results: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 5: External validity (ARI, NMI, F1 vs ground truth).
    """
    metrics = ['ARI', 'NMI', 'Macro-F1']
    hdp_scores = [
        np.mean(loso_results['hdp_ari']),
        np.mean(loso_results['hdp_nmi']),
        np.mean(loso_results['hdp_f1'])
    ]
    idp_scores = [
        np.mean(loso_results['idp_ari']),
        np.mean(loso_results['idp_nmi']),
        np.mean(loso_results['idp_f1'])
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, hdp_scores, width, label='HDP-HMM (sticky)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, idp_scores, width, label='iDP-HMM', color='coral', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Biological Plausibility: Agreement with Ground Truth Labels', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (h, d) in enumerate(zip(hdp_scores, idp_scores)):
        ax.text(i - width/2, h + 0.02, f'{h:.3f}', ha='center', fontweight='bold')
        ax.text(i + width/2, d + 0.02, f'{d:.3f}', ha='center', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_stick_breaking_weights(
    hdp_samples: List[Dict],
    n_weights: int = 15,
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 6: Stick-breaking weights β.
    
    Shows HDP concentrates mass on few states.
    """
    # Collect β samples
    betas = np.array([s['beta'][:n_weights] for s in hdp_samples])
    
    beta_mean = betas.mean(axis=0)
    beta_std = betas.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot with error bars
    x = np.arange(n_weights)
    axes[0].bar(x, beta_mean, yerr=beta_std, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('State Index', fontsize=14)
    axes[0].set_ylabel('Weight β_k', fontsize=14)
    axes[0].set_title('Global Stick-Breaking Weights', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative mass
    cum_mass = np.cumsum(beta_mean)
    axes[1].plot(x, cum_mass, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[1].axhline(0.95, color='red', linestyle='--', label='95% mass')
    axes[1].set_xlabel('State Index', fontsize=14)
    axes[1].set_ylabel('Cumulative Mass', fontsize=14)
    axes[1].set_title('Mass Concentration', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    plt.suptitle('HDP Concentrates Mass on Few Shared States', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_states_vs_subjects(
    results_by_n_subjects: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 7: E[K] growth with number of subjects.
    
    Shows controlled growth under HDP vs proliferation under iDP.
    """
    n_subjects = sorted(results_by_n_subjects.keys())
    hdp_K = [np.mean(results_by_n_subjects[n]['hdp_K']) for n in n_subjects]
    hdp_K_std = [np.std(results_by_n_subjects[n]['hdp_K']) for n in n_subjects]
    idp_K = [np.mean(results_by_n_subjects[n]['idp_K_total']) for n in n_subjects]
    idp_K_std = [np.std(results_by_n_subjects[n]['idp_K_total']) for n in n_subjects]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(n_subjects, hdp_K, yerr=hdp_K_std, marker='o', linewidth=2, 
                markersize=8, capsize=5, label='HDP-HMM (global K)', color='steelblue')
    ax.errorbar(n_subjects, idp_K, yerr=idp_K_std, marker='s', linewidth=2,
                markersize=8, capsize=5, label='iDP-HMM (total K)', color='coral')
    
    ax.set_xlabel('Number of Subjects (M)', fontsize=14)
    ax.set_ylabel('Expected Number of States E[K]', fontsize=14)
    ax.set_title('Nonparametric Growth: Controlled vs Uncontrolled', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_hypnogram_reconstruction(
    true_labels: np.ndarray,
    hdp_pred: np.ndarray,
    idp_pred: np.ndarray,
    subject_id: str = "Subject 1",
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 8: Representative hypnogram reconstruction.
    
    Shows true vs predicted sleep stages for one subject.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    time = np.arange(len(true_labels)) / 120  # Convert to hours (30s epochs)
    
    # True hypnogram
    axes[0].plot(time, true_labels, color='black', linewidth=1.5)
    axes[0].set_ylabel('Sleep Stage', fontsize=12)
    axes[0].set_title(f'{subject_id}: Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(STAGE_NAMES)
    axes[0].grid(True, alpha=0.3)
    
    # HDP-HMM prediction
    axes[1].plot(time, hdp_pred, color='steelblue', linewidth=1.5)
    axes[1].set_ylabel('Sleep Stage', fontsize=12)
    axes[1].set_title('HDP-HMM (Sticky) Prediction', fontsize=14, fontweight='bold')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(STAGE_NAMES)
    axes[1].grid(True, alpha=0.3)
    
    # iDP-HMM prediction
    axes[2].plot(time, idp_pred, color='coral', linewidth=1.5)
    axes[2].set_ylabel('Sleep Stage', fontsize=12)
    axes[2].set_xlabel('Time (hours)', fontsize=14)
    axes[2].set_title('iDP-HMM Prediction', fontsize=14, fontweight='bold')
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(STAGE_NAMES)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def plot_kappa_ablation(
    kappa_values: List[float],
    ari_scores: List[float],
    ll_scores: List[float],
    output_path: Optional[Path] = None
) -> None:
    """
    Figure 9: Ablation study - effect of stickiness κ.
    
    Shows how performance varies with sticky parameter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ARI vs kappa
    axes[0].plot(kappa_values, ari_scores, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Stickiness Parameter κ', fontsize=14)
    axes[0].set_ylabel('Adjusted Rand Index', fontsize=14)
    axes[0].set_title('Label Agreement vs Stickiness', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Log-likelihood vs kappa
    axes[1].plot(kappa_values, ll_scores, 'o-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('Stickiness Parameter κ', fontsize=14)
    axes[1].set_ylabel('Test Log-Likelihood', fontsize=14)
    axes[1].set_title('Predictive Performance vs Stickiness', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.suptitle('Ablation: Role of Sticky Self-Transitions', fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.show()


def create_summary_table(
    results: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Table 1: Summary comparison table.
    
    Returns formatted LaTeX/Markdown table.
    """
    table = """
| Model            | E[K]  | Median Dwell (s) | Test Log-Lik | ARI   | NMI   | Macro-F1 |
|------------------|-------|------------------|--------------|-------|-------|----------|
| iDP-HMM          | {idp_K:5.1f} | {idp_dwell:14.0f} | {idp_ll:11.1f} | {idp_ari:5.3f} | {idp_nmi:5.3f} | {idp_f1:8.3f} |
| HDP-HMM (sticky) | {hdp_K:5.1f} | {hdp_dwell:14.0f} | {hdp_ll:11.1f} | {hdp_ari:5.3f} | {hdp_nmi:5.3f} | {hdp_f1:8.3f} |
""".format(**results)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Saved: {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(table)
    print("="*80)
    
    return table
