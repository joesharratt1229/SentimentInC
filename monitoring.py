import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------
# 1. PSI Implementation
# ----------------------

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate the Population Stability Index (PSI) for a feature.
    
    Parameters:
    -----------
    expected: 1-D numpy array or pandas Series 
        The distribution values used as the baseline
    actual: 1-D numpy array or pandas Series
        The distribution values to compare to the baseline
    buckettype: str, optional (default='bins')
        'bins' for equal-width bins, 'quantiles' for equal-frequency bins
    buckets: int, optional (default=10)
        Number of buckets to use in the calculation
    axis: int, optional (default=0)
        Axis to calculate over
    
    Returns:
    --------
    psi_value: float
        The PSI value comparing the distributions
    bucket_details: pandas DataFrame
        Details of each bucket for further analysis
    """
    
    def psi_formula(expected_pct, actual_pct):
        """Formula for calculating PSI contribution for a single bin"""
        # Add a small epsilon to prevent division by zero or log(0)
        epsilon = 1e-6
        
        # Replace zeros with epsilon
        expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
        actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
        
        # Calculate the PSI contribution for this bin
        psi_bin = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        return psi_bin
    
    # Ensure inputs are numpy arrays
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Create buckets
    if buckettype == 'bins':
        # Equal-width bins
        breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
    else:
        # Equal-frequency bins on the expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Ensure we have unique breakpoints
    breakpoints = np.unique(breakpoints)
    
    # Assign observations to buckets
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Calculate percentages in each bin
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Calculate PSI for each bin
    psi_values = psi_formula(expected_pct, actual_pct)
    
    # Create result DataFrame with details of each bucket
    bin_edges = list(zip(breakpoints[:-1], breakpoints[1:]))
    bucket_details = pd.DataFrame({
        'Bucket': [f"{bucket[0]:.2f}-{bucket[1]:.2f}" for bucket in bin_edges],
        'Expected_Count': expected_counts,
        'Actual_Count': actual_counts,
        'Expected_Pct': expected_pct,
        'Actual_Pct': actual_pct,
        'PSI_Contribution': psi_values
    })
    
    # Calculate overall PSI
    psi_value = np.sum(psi_values)
    
    return psi_value, bucket_details

# Function to interpret PSI values
def interpret_psi(psi_value):
    """
    Interpret PSI values according to common industry standards.
    
    Parameters:
    -----------
    psi_value: float
        The PSI value to interpret
    
    Returns:
    --------
    interpretation: str
        Interpretation of the PSI value
    """
    if psi_value < 0.1:
        return "Insignificant change (PSI < 0.1). Model should be stable."
    elif psi_value < 0.2:
        return "Slight change (0.1 <= PSI < 0.2). Monitor the model."
    elif psi_value < 0.5:
        return "Significant change (0.2 <= PSI < 0.5). Consider model retraining."
    else:
        return "Severe change (PSI >= 0.5). Model needs immediate retraining."

# ----------------------
# 2. Generate Synthetic Data with Controlled Drift
# ----------------------

def generate_baseline_data(n_samples=10000, n_features=5):
    """
    Generate baseline dataset for a simple classification problem.
    
    Parameters:
    -----------
    n_samples: int, optional (default=10000)
        Number of samples to generate
    n_features: int, optional (default=5)
        Number of features to generate
    
    Returns:
    --------
    X: pandas DataFrame
        Feature matrix
    y: pandas Series
        Target variable
    """
    # Generate feature matrix from normal distributions
    X = pd.DataFrame({
        f'feature_{i}': np.random.normal(loc=0, scale=1, size=n_samples)
        for i in range(n_features)
    })
    
    # Generate target variable as a function of features
    # We'll use feature_0 and feature_1 with some noise
    logit = 1.5 * X['feature_0'] - 2.0 * X['feature_1'] + 0.5 * X['feature_2']
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    
    return X, y

def generate_drift_data(X_baseline, drift_type='mean', drift_magnitude=1.0, feature_idx=0):
    """
    Generate a new dataset with controlled drift from the baseline.
    
    Parameters:
    -----------
    X_baseline: pandas DataFrame
        The baseline feature matrix
    drift_type: str, optional (default='mean')
        Type of drift to introduce: 'mean', 'variance', 'skew'
    drift_magnitude: float, optional (default=1.0)
        Magnitude of the drift to introduce
    feature_idx: int or list, optional (default=0)
        Index or indices of features to apply drift to
    
    Returns:
    --------
    X_drift: pandas DataFrame
        Feature matrix with drift applied
    """
    # Create a copy of the baseline data
    X_drift = X_baseline.copy()
    
    # Convert feature_idx to list if it's a single index
    if isinstance(feature_idx, int):
        feature_idx = [feature_idx]
    
    # Apply drift to specified features
    for idx in feature_idx:
        feature_name = f'feature_{idx}'
        
        if drift_type == 'mean':
            # Shift the mean
            X_drift[feature_name] = X_drift[feature_name] + drift_magnitude
        
        elif drift_type == 'variance':
            # Change the variance
            X_drift[feature_name] = X_drift[feature_name] * drift_magnitude
        
        elif drift_type == 'skew':
            # Introduce skewness
            X_drift[feature_name] = np.exp(drift_magnitude * X_drift[feature_name]) - 1
        
        elif drift_type == 'mixture':
            # Create a mixture distribution (bimodal)
            mask = np.random.binomial(n=1, p=0.3, size=len(X_drift))
            X_drift.loc[mask == 1, feature_name] = X_drift.loc[mask == 1, feature_name] + drift_magnitude * 3
    
    return X_drift

def create_y_with_drift(X_drift, original_model, noise_level=0.0):
    """
    Create target variable for the drift data, assuming the same underlying relationship
    but potentially with added noise to simulate concept drift.
    
    Parameters:
    -----------
    X_drift: pandas DataFrame
        Feature matrix with drift
    original_model: fitted model
        The model trained on the baseline data
    noise_level: float, optional (default=0.0)
        Level of noise to add (0.0 means no concept drift)
    
    Returns:
    --------
    y_drift: numpy array
        Target variable for the drift data
    """
    # Get the original predictions
    y_prob = original_model.predict_proba(X_drift)[:, 1]
    
    # Add noise to simulate concept drift
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=len(y_prob))
        y_prob = y_prob + noise
        
        # Ensure probabilities are between 0 and 1
        y_prob = np.clip(y_prob, 0, 1)
    
    # Convert to binary labels
    y_drift = (y_prob > 0.5).astype(int)
    
    return y_drift

# ----------------------
# 3. Experiment Setup
# ----------------------

def run_psi_experiment(drift_type='mean', feature_idx=0, drift_values=None):
    """
    Run an experiment to evaluate the relationship between PSI and model performance
    under different levels of data drift.
    
    Parameters:
    -----------
    drift_type: str, optional (default='mean')
        Type of drift to introduce
    feature_idx: int or list, optional (default=0)
        Index or indices of features to apply drift to
    drift_values: list, optional
        List of drift magnitudes to evaluate
    
    Returns:
    --------
    results: pandas DataFrame
        Results of the experiment with PSI values and performance metrics
    """
    if drift_values is None:
        if drift_type == 'mean':
            drift_values = np.linspace(0, 2.0, 11)  # 0 to 2 in 11 steps
        elif drift_type == 'variance':
            drift_values = np.linspace(1, 3.0, 11)  # 1 to 3 in 11 steps
        elif drift_type == 'skew':
            drift_values = np.linspace(0, 1.0, 11)  # 0 to 1 in 11 steps
        elif drift_type == 'mixture':
            drift_values = np.linspace(0, 1.5, 11)  # 0 to 1.5 in 11 steps
    
    # Generate baseline data
    X, y = generate_baseline_data(n_samples=20000)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a model on the baseline data
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on baseline test data
    baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    baseline_f1 = f1_score(y_test, model.predict(X_test_scaled))
    
    print(f"Baseline Performance - AUC: {baseline_auc:.4f}, Accuracy: {baseline_accuracy:.4f}, F1: {baseline_f1:.4f}")
    
    # Initialize results storage
    results = []
    
    # Dictionary to store feature distributions for visualization
    distributions = {
        'baseline': X_test[f'feature_{feature_idx[0] if isinstance(feature_idx, list) else feature_idx}'].values
    }
    
    # Evaluate for each drift magnitude
    for drift_mag in drift_values:
        # Generate drift data
        X_drift = generate_drift_data(X_test, drift_type=drift_type, 
                                     drift_magnitude=drift_mag, feature_idx=feature_idx)
        
        # Calculate PSI for the drifted feature(s)
        if isinstance(feature_idx, list):
            psi_values = []
            for idx in feature_idx:
                feature_name = f'feature_{idx}'
                psi_val, _ = calculate_psi(X_test[feature_name].values, X_drift[feature_name].values)
                psi_values.append(psi_val)
            psi = np.mean(psi_values)  # Average PSI across features
        else:
            feature_name = f'feature_{feature_idx}'
            psi, bucket_details = calculate_psi(X_test[feature_name].values, X_drift[feature_name].values)
            
            # Store distribution for major drift points for visualization
            if drift_mag in [drift_values[0], drift_values[len(drift_values)//2], drift_values[-1]]:
                distributions[f'drift_{drift_mag:.1f}'] = X_drift[feature_name].values
        
        # Scale drifted data
        X_drift_scaled = scaler.transform(X_drift)
        
        # Create target variable with the same relationship (no concept drift)
        y_drift = create_y_with_drift(X_drift, model, noise_level=0.0)
        
        # Evaluate model on drifted data
        drift_auc = roc_auc_score(y_drift, model.predict_proba(X_drift_scaled)[:, 1])
        drift_accuracy = accuracy_score(y_drift, model.predict(X_drift_scaled))
        drift_f1 = f1_score(y_drift, model.predict(X_drift_scaled))
        
        # Store results
        results.append({
            'Drift_Type': drift_type,
            'Drift_Magnitude': drift_mag,
            'PSI': psi,
            'AUC': drift_auc,
            'Accuracy': drift_accuracy,
            'F1': drift_f1,
            'AUC_Relative': drift_auc / baseline_auc,
            'Accuracy_Relative': drift_accuracy / baseline_accuracy,
            'F1_Relative': drift_f1 / baseline_f1,
            'PSI_Interpretation': interpret_psi(psi)
        })
        
        print(f"Drift {drift_type} = {drift_mag:.2f}, PSI = {psi:.4f}, AUC = {drift_auc:.4f}, " +
              f"Accuracy = {drift_accuracy:.4f}, F1 = {drift_f1:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot distributions for selected drift points
    plt.figure(figsize=(12, 6))
    for label, data in distributions.items():
        sns.kdeplot(data, label=label)
    plt.title(f'Feature Distribution Changes ({drift_type} drift)')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Plot PSI vs performance metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PSI vs AUC
    axes[0].plot(results_df['PSI'], results_df['AUC_Relative'], marker='o', linestyle='-')
    axes[0].set_title('PSI vs Relative AUC')
    axes[0].set_xlabel('PSI')
    axes[0].set_ylabel('Relative AUC (drift/baseline)')
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[0].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[0].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[0].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[0].set_ylim(0.8, 1.05)
    axes[0].legend()
    
    # PSI vs Accuracy
    axes[1].plot(results_df['PSI'], results_df['Accuracy_Relative'], marker='o', linestyle='-')
    axes[1].set_title('PSI vs Relative Accuracy')
    axes[1].set_xlabel('PSI')
    axes[1].set_ylabel('Relative Accuracy (drift/baseline)')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[1].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[1].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[1].set_ylim(0.8, 1.05)
    axes[1].legend()
    
    # PSI vs F1
    axes[2].plot(results_df['PSI'], results_df['F1_Relative'], marker='o', linestyle='-')
    axes[2].set_title('PSI vs Relative F1 Score')
    axes[2].set_xlabel('PSI')
    axes[2].set_ylabel('Relative F1 (drift/baseline)')
    axes[2].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[2].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[2].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[2].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[2].set_ylim(0.8, 1.05)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df, distributions

def visualize_bucket_details(expected, actual, buckets=10):
    """
    Visualize the bucket details of a PSI calculation.
    
    Parameters:
    -----------
    expected: numpy array or pandas Series
        The expected distribution
    actual: numpy array or pandas Series
        The actual distribution to compare against
    buckets: int, optional (default=10)
        Number of buckets to use
    """
    # Calculate PSI and get bucket details
    psi, bucket_details = calculate_psi(expected, actual, buckets=buckets)
    
    # Plot the bucket details
    plt.figure(figsize=(14, 10))
    
    # Distribution comparison
    plt.subplot(2, 1, 1)
    width = 0.35
    x = np.arange(len(bucket_details))
    
    plt.bar(x - width/2, bucket_details['Expected_Pct'], width, label='Expected')
    plt.bar(x + width/2, bucket_details['Actual_Pct'], width, label='Actual')
    
    plt.xlabel('Bucket')
    plt.ylabel('Proportion')
    plt.title(f'Distribution Comparison - PSI = {psi:.4f}')
    plt.xticks(x, bucket_details['Bucket'], rotation=45)
    plt.legend()
    
    # PSI contribution by bucket
    plt.subplot(2, 1, 2)
    plt.bar(x, bucket_details['PSI_Contribution'], color='orange')
    plt.xlabel('Bucket')
    plt.ylabel('PSI Contribution')
    plt.title('PSI Contribution by Bucket')
    plt.xticks(x, bucket_details['Bucket'], rotation=45)
    
    # Add total PSI as text
    plt.text(0.02, 0.95, f'Total PSI: {psi:.4f}\n{interpret_psi(psi)}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return psi, bucket_details

# ----------------------
# 4. Run Experiments and Visualize Results
# ----------------------

def main():
    """Main function to run the PSI experiments"""
    print("Starting PSI experiments...\n")
    
    # Mean shift experiment
    print("\n=== Mean Shift Experiment ===")
    results_mean, dist_mean = run_psi_experiment(drift_type='mean', feature_idx=0)
    
    # Variance change experiment
    print("\n=== Variance Change Experiment ===")
    results_var, dist_var = run_psi_experiment(drift_type='variance', feature_idx=0)
    
    # Skew experiment
    print("\n=== Skew Experiment ===")
    results_skew, dist_skew = run_psi_experiment(drift_type='skew', feature_idx=0)
    
    # Multi-feature experiment
    print("\n=== Multi-feature Experiment ===")
    results_multi, _ = run_psi_experiment(drift_type='mean', feature_idx=[0, 1])
    
    # Create summary table with recommended thresholds
    summary = pd.DataFrame({
        'PSI_Threshold': [0.1, 0.2, 0.5],
        'Interpretation': ['Minor Change', 'Moderate Change', 'Significant Change'],
        'Action': ['Monitor', 'Investigate', 'Retrain Model'],
        'Expected_Performance_Impact': ['< 5%', '5-10%', '> 10%']
    })
    
    print("\n=== PSI Threshold Guidelines ===")
    print(summary)
    
    # Detailed bucket analysis example
    print("\n=== Detailed Bucket Analysis Example ===")
    # Generate baseline and drift data
    X, y = generate_baseline_data(n_samples=10000)
    X_drift = generate_drift_data(X, drift_type='mean', drift_magnitude=1.0, feature_idx=0)
    
    # Analyze feature_0
    feature_name = 'feature_0'
    visualize_bucket_details(X[feature_name].values, X_drift[feature_name].values)
    
    print("\nExperiments completed!")

if __name__ == "__main__":
    main()

# ----------------------
# 5. Educational Notes on PSI
# ----------------------

"""
UNDERSTANDING PSI AND DATA DRIFT MONITORING

What is PSI?
-----------
Population Stability Index (PSI) is a statistical measure used to quantify how much a distribution 
has changed compared to a baseline distribution. It is commonly used in model monitoring to detect 
data drift, which can significantly impact model performance.

PSI Formula:
-----------
PSI = SUM[ (Actual% - Expected%) * ln(Actual% / Expected%) ]

Where:
- Expected% is the percentage of observations in a bucket in the baseline distribution
- Actual% is the percentage of observations in the same bucket in the new distribution
- SUM[] means we sum this calculation across all buckets

Interpreting PSI Thresholds:
---------------------------
0.0 < PSI < 0.1: No significant change, model should be stable
0.1 ≤ PSI < 0.2: Slight change, worth monitoring the model
0.2 ≤ PSI < 0.5: Significant change, consider model retraining
PSI ≥ 0.5: Severe change, model needs immediate retraining

Advantages of PSI:
-----------------
1. Single metric: PSI condenses distribution changes into a single number
2. Interpretable: Industry standards for thresholds are well-established
3. Sensitive: Detects subtle changes in distributions
4. Non-parametric: No assumptions about the underlying distribution

Limitations:
-----------
1. Bucket selection can influence results
2. Sensitive to small counts in buckets (can be mitigated with smoothing)
3. Only measures distributional changes, not changes in relationships between features and target

Best Practices:
--------------
1. Calculate PSI for each input feature independently
2. Monitor PSI trends over time, not just absolute values
3. Combine with performance monitoring metrics (AUC, accuracy, etc.)
4. Set appropriate thresholds based on business context and model criticality
5. Consider using both equal-width and equal-frequency buckets for different insights

Integration with Model Monitoring:
--------------------------------
1. Calculate baseline distributions during model training/validation
2. Calculate PSI for production data periodically (daily, weekly, monthly)
3. Set alerts based on PSI thresholds
4. Investigate features with high PSI values
5. Retrain models when PSI values exceed acceptable thresholds

The experiments in this notebook demonstrate how different types of data drift 
(mean shifts, variance changes, skewness) affect PSI values and model performance.
"""
