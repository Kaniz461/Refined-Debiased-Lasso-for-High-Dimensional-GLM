# Refined-Debiased-Lasso-for-High-Dimensional-GLM

"""
DIRECT SIMULATION - Shows Results Inline
Replicates paper methodology for p=40, 100, 200
Displays plots and statistics directly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Simulation parameters
np.random.seed(159)
n = 500
sig_level = 0.05
z_val = norm.ppf(1 - sig_level/2)
n_sim = 50  # Reduced for faster execution

# True coefficients
s0 = 4
large_signal = 1.0
small_signal = 0.5
beta1_values = np.linspace(0, 1.5, 20)  # Reduced for speed

print("="*70)
print("SIMULATION STUDY - INLINE RESULTS")
print(f"n={n}, p=[40,100], {n_sim} simulations")
print("="*70)

# Estimation functions
def fit_lasso(X, y):
    model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5,
                                  max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return np.concatenate([[model.intercept_[0]], model.coef_[0]])

def refined_debiased_lasso(X, y, beta_lasso):
    n = len(y)
    X_int = np.column_stack([np.ones(n), X])
    mu = np.clip(1 / (1 + np.exp(-X_int @ beta_lasso)), 1e-10, 1-1e-10)
    neg_dloglik = -(X_int.T @ (y - mu)) / n
    W = mu * (1 - mu)
    neg_ddloglik = (X_int.T @ (X_int * W[:, None])) / n + 1e-6 * np.eye(len(beta_lasso))
    theta_inv = np.linalg.inv(neg_ddloglik)
    beta = beta_lasso - theta_inv @ neg_dloglik
    se = np.sqrt(np.diag(theta_inv)) / np.sqrt(n)
    return beta, se

def original_debiased_lasso(X, y, beta_lasso):
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    mu = np.clip(1 / (1 + np.exp(-X_int @ beta_lasso)), 1e-10, 1-1e-10)
    neg_dloglik = -(X_int.T @ (y - mu)) / n
    W = mu * (1 - mu)
    neg_ddloglik = (X_int.T @ (X_int * W[:, None])) / n
    C = np.sqrt(W[:, None]) / np.sqrt(n) * X_int

    theta = np.eye(p + 1)
    tau = np.zeros(p + 1)

    for j in range(min(15, p + 1)):
        mask = [i for i in range(p+1) if i != j]
        model = LassoCV(cv=3, max_iter=300, n_jobs=1, random_state=42)
        model.fit(np.sqrt(n) * C[:, mask], np.sqrt(n) * C[:, j])
        theta[j, mask] = -model.coef_
        tau[j] = neg_ddloglik[j, j] - neg_ddloglik[j, mask] @ model.coef_

    tau = np.clip(tau, 1e-6, None)
    theta = theta / tau[:, None]
    beta = beta_lasso - theta @ neg_dloglik
    se = np.sqrt(np.diag(theta @ neg_ddloglik @ theta.T)) / np.sqrt(n)
    return beta, se

def mle_logistic(X, y):
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    beta = fit_lasso(X, y)

    for _ in range(10):
        mu = np.clip(1 / (1 + np.exp(-X_int @ beta)), 1e-10, 1-1e-10)
        W = mu * (1 - mu)
        hess = X_int.T @ (X_int * W[:, None]) + 1e-6 * np.eye(len(beta))
        grad = X_int.T @ (y - mu)
        try:
            beta = beta + np.linalg.solve(hess, grad)
        except:
            break

    mu = np.clip(1 / (1 + np.exp(-X_int @ beta)), 1e-10, 1-1e-10)
    W = mu * (1 - mu)
    hess = X_int.T @ (X_int * W[:, None]) / n + 1e-6 * np.eye(len(beta))
    se = np.sqrt(np.diag(np.linalg.inv(hess))) / np.sqrt(n)
    return beta, se

def generate_data(n, p, beta_true, rho=0.7):
    covmat = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = np.clip(np.random.multivariate_normal(np.zeros(p), covmat, size=n), -6, 6)
    prob = 1 / (1 + np.exp(-np.column_stack([np.ones(n), X]) @ beta_true))
    y = np.random.binomial(1, prob)
    return X, y

# Run simulations
def run_sim(p):
    print(f"\n{'='*70}")
    print(f"Running: p={p}")
    print(f"{'='*70}")

    results = {'beta1': [], 'REF_bias': [], 'REF_coverage': [], 'REF_emp_se': [], 'REF_model_se': [],
               'ORIG_bias': [], 'ORIG_coverage': [], 'ORIG_emp_se': [], 'ORIG_model_se': [],
               'MLE_bias': [], 'MLE_coverage': [], 'MLE_emp_se': [], 'MLE_model_se': []}

    for idx, beta1_val in enumerate(beta1_values):
        if (idx + 1) % 5 == 0:
            print(f"  Progress: {idx + 1}/{len(beta1_values)}")

        beta_true = np.zeros(p + 1)
        beta_true[1] = beta1_val
        signal_idx = np.random.choice(range(2, p+1), size=s0, replace=False)
        beta_true[signal_idx[:2]] = small_signal
        beta_true[signal_idx[2:]] = large_signal

        ref_ests, orig_ests, mle_ests = [], [], []
        ref_ses, orig_ses, mle_ses = [], [], []
        ref_covers, orig_covers, mle_covers = [], [], []

        for sim in range(n_sim):
            try:
                X, y = generate_data(n, p, beta_true)
                beta_lasso = fit_lasso(X, y)

                beta_ref, se_ref = refined_debiased_lasso(X, y, beta_lasso)
                ci_l, ci_u = beta_ref[1] - z_val * se_ref[1], beta_ref[1] + z_val * se_ref[1]
                ref_ests.append(beta_ref[1])
                ref_ses.append(se_ref[1])
                ref_covers.append(1 if ci_l <= beta1_val <= ci_u else 0)

                beta_orig, se_orig = original_debiased_lasso(X, y, beta_lasso)
                ci_l, ci_u = beta_orig[1] - z_val * se_orig[1], beta_orig[1] + z_val * se_orig[1]
                orig_ests.append(beta_orig[1])
                orig_ses.append(se_orig[1])
                orig_covers.append(1 if ci_l <= beta1_val <= ci_u else 0)

                beta_mle, se_mle = mle_logistic(X, y)
                ci_l, ci_u = beta_mle[1] - z_val * se_mle[1], beta_mle[1] + z_val * se_mle[1]
                mle_ests.append(beta_mle[1])
                mle_ses.append(se_mle[1])
                mle_covers.append(1 if ci_l <= beta1_val <= ci_u else 0)
            except:
                continue

        results['beta1'].append(beta1_val)
        results['REF_bias'].append(np.mean(ref_ests) - beta1_val)
        results['REF_coverage'].append(np.mean(ref_covers))
        results['REF_emp_se'].append(np.std(ref_ests))
        results['REF_model_se'].append(np.mean(ref_ses))
        results['ORIG_bias'].append(np.mean(orig_ests) - beta1_val)
        results['ORIG_coverage'].append(np.mean(orig_covers))
        results['ORIG_emp_se'].append(np.std(orig_ests))
        results['ORIG_model_se'].append(np.mean(orig_ses))
        results['MLE_bias'].append(np.mean(mle_ests) - beta1_val)
        results['MLE_coverage'].append(np.mean(mle_covers))
        results['MLE_emp_se'].append(np.std(mle_ests))
        results['MLE_model_se'].append(np.mean(mle_ses))

    return pd.DataFrame(results)

# Run for all p
results = {}
for p in [40, 100]:
    results[p] = run_sim(p)

print("\n" + "="*70)
print("✓ SIMULATIONS COMPLETE - Creating Plots")
print("="*70)

# Create publication-quality plot
fig, axes = plt.subplots(4, 3, figsize=(15, 16))

colors = {'REF': '#FF8C00', 'ORIG': '#0000FF', 'MLE': '#FF0000', 'Oracle': '#FFA500'}
lines = {'REF': '-.', 'ORIG': '--', 'MLE': ':', 'Oracle': '-'}

for col, p in enumerate([40, 100]):
    df = results[p]

    # Bias
    ax = axes[0, col]
    ax.plot(df['beta1'], df['REF_bias'], color=colors['REF'], linestyle=lines['REF'],
            linewidth=2, label='REF-DS')
    ax.plot(df['beta1'], df['ORIG_bias'], color=colors['ORIG'], linestyle=lines['ORIG'],
            linewidth=2, label='ORIG-DS')
    ax.plot(df['beta1'], df['MLE_bias'], color=colors['MLE'], linestyle=lines['MLE'],
            linewidth=2, label='MLE')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Signal', fontsize=11)
    ax.set_ylabel('Bias', fontsize=11)
    ax.set_title(f'p = {p}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc='best', fontsize=9)

    # Coverage
    ax = axes[1, col]
    ax.plot(df['beta1'], df['REF_coverage'], color=colors['REF'], linestyle=lines['REF'],
            linewidth=2, label='REF-DS')
    ax.plot(df['beta1'], df['ORIG_coverage'], color=colors['ORIG'], linestyle=lines['ORIG'],
            linewidth=2, label='ORIG-DS')
    ax.plot(df['beta1'], df['MLE_coverage'], color=colors['MLE'], linestyle=lines['MLE'],
            linewidth=2, label='MLE')
    ax.axhline(0.95, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Signal', fontsize=11)
    ax.set_ylabel('Coverage Probability', fontsize=11)
    ax.set_title(f'p = {p}', fontsize=12, fontweight='bold')
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc='best', fontsize=9)

    # Empirical SE
    ax = axes[2, col]
    ax.plot(df['beta1'], df['REF_emp_se'], color=colors['REF'], linestyle=lines['REF'],
            linewidth=2, label='REF-DS')
    ax.plot(df['beta1'], df['ORIG_emp_se'], color=colors['ORIG'], linestyle=lines['ORIG'],
            linewidth=2, label='ORIG-DS')
    ax.plot(df['beta1'], df['MLE_emp_se'], color=colors['MLE'], linestyle=lines['MLE'],
            linewidth=2, label='MLE')
    ax.plot(df['beta1'], df['REF_model_se'], color=colors['Oracle'], linestyle=lines['Oracle'],
            linewidth=2, label='Oracle', alpha=0.7)
    ax.set_xlabel('Signal', fontsize=11)
    ax.set_ylabel('Empirical SE', fontsize=11)
    ax.set_title(f'p = {p}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc='best', fontsize=9)

    # Model-based SE
    ax = axes[3, col]
    ax.plot(df['beta1'], df['REF_model_se'], color=colors['REF'], linestyle=lines['REF'],
            linewidth=2, label='REF-DS')
    ax.plot(df['beta1'], df['ORIG_model_se'], color=colors['ORIG'], linestyle=lines['ORIG'],
            linewidth=2, label='ORIG-DS')
    ax.plot(df['beta1'], df['MLE_model_se'], color=colors['MLE'], linestyle=lines['MLE'],
            linewidth=2, label='MLE')
    ax.plot(df['beta1'], df['REF_emp_se'], color=colors['Oracle'], linestyle=lines['Oracle'],
            linewidth=2, label='Oracle', alpha=0.7)
    ax.set_xlabel('Signal', fontsize=11)
    ax.set_ylabel('Model-based SE', fontsize=11)
    ax.set_title(f'p = {p}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for p in [40, 100]:
    df = results[p]
    print(f"\np = {p}:")
    print(f"{'Method':<12} {'Mean |Bias|':<15} {'Mean Coverage':<15} {'Mean SE':<12}")
    print("-" * 65)

    for method in ['REF', 'ORIG', 'MLE']:
        mean_bias = df[f'{method}_bias'].abs().mean()
        mean_cov = df[f'{method}_coverage'].mean()
        mean_se = df[f'{method}_model_se'].mean()
        print(f"{method}-DS{'':<7} {mean_bias:<15.4f} {mean_cov:<15.3f} {mean_se:<12.4f}")

print("\n" + "="*70)
print("✓ COMPLETE!")
print("="*70)
