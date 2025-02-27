# Population Stability Index (PSI) for Model Monitoring
# ====================================================
#
# This notebook demonstrates how to calculate and interpret Population Stability Index (PSI)
# for monitoring data drift, and how different levels of drift affect model performance.

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
        breakpoints = np.linspace(min(expected.min(), actual.min()), 
                                 max(expected.max(), actual.max()), 
                                 buckets + 1)
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
# 2. Data Generation and Model Building
# ----------------------

def generate_data(n_samples=5000, n_features=5, random_state=42):
    """
    Generate synthetic data for a binary classification problem.
    
    Parameters:
    -----------
    n_samples: int, optional (default=5000)
        Number of samples to generate
    n_features: int, optional (default=5)
        Number of features to generate
    random_state: int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    --------
    X: pandas DataFrame
        Feature matrix
    y: pandas Series
        Target variable
    """
    np.random.seed(random_state)
    
    # Generate features
    X = pd.DataFrame({
        f'feature_{i}': np.random.normal(loc=0, scale=1, size=n_samples)
        for i in range(n_features)
    })
    
    # Generate target as a function of features with some noise
    # We'll use feature_0, feature_1, and feature_2 as the most important ones
    logit = (1.5 * X['feature_0'] - 
             0.8 * X['feature_1'] + 
             0.5 * X['feature_2'] + 
             0.2 * np.random.normal(size=n_samples))  # Add noise
    
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    
    return X, y

def introduce_drift(X, drift_type='mean', feature_idx=0, drift_magnitude=1.0):
    """
    Introduce a specific type of drift to the feature data.
    
    Parameters:
    -----------
    X: pandas DataFrame
        Original feature data
    drift_type: str, optional (default='mean')
        Type of drift to introduce: 'mean', 'variance', 'skew', 'correlation'
    feature_idx: int or list, optional (default=0)
        Index of feature(s) to apply drift to
    drift_magnitude: float, optional (default=1.0)
        Magnitude of drift to introduce
    
    Returns:
    --------
    X_drift: pandas DataFrame
        Feature data with drift applied
    """
    X_drift = X.copy()
    
    # Make feature_idx a list if it's a single value
    if isinstance(feature_idx, int):
        feature_idx = [feature_idx]
    
    for idx in feature_idx:
        feature_name = f'feature_{idx}'
        
        if drift_type == 'mean':
            # Shift the mean
            X_drift[feature_name] = X_drift[feature_name] + drift_magnitude
            
        elif drift_type == 'variance':
            # Change the variance (scale the data)
            X_drift[feature_name] = X_drift[feature_name] * drift_magnitude
            
        elif drift_type == 'skew':
            # Introduce skewness using exponential transformation
            X_drift[feature_name] = np.exp(drift_magnitude * X_drift[feature_name]) - 1
            
        elif drift_type == 'bimodal':
            # Create a bimodal distribution by shifting a portion of the data
            mask = np.random.binomial(n=1, p=0.4, size=len(X_drift))
            X_drift.loc[mask == 1, feature_name] = X_drift.loc[mask == 1, feature_name] + drift_magnitude * 2
    
    return X_drift

def compute_target_with_true_relationship(X_drift, noise_level=0.2):
    """
    Compute the target variable based on the true underlying relationship.
    This simulates what would happen in reality if the feature distribution changed.
    
    Parameters:
    -----------
    X_drift: pandas DataFrame
        Feature data with drift applied
    noise_level: float, optional (default=0.2)
        Level of noise to add to the relationship
    
    Returns:
    --------
    y_drift: numpy array
        Target variable based on the true relationship with the drifted features
    """
    # Use the same relationship as in the generate_data function
    logit = (1.5 * X_drift['feature_0'] - 
             0.8 * X_drift['feature_1'] + 
             0.5 * X_drift['feature_2'] + 
             noise_level * np.random.normal(size=len(X_drift)))
    
    prob = 1 / (1 + np.exp(-logit))
    y_drift = (prob > 0.5).astype(int)
    
    return y_drift

# ----------------------
# 3. PSI Analysis Functions
# ----------------------

def compute_feature_psi_values(X_train, X_test):
    """
    Compute PSI values for all features between two datasets.
    
    Parameters:
    -----------
    X_train: pandas DataFrame
        Training dataset (expected distribution)
    X_test: pandas DataFrame
        Test dataset (actual distribution)
    
    Returns:
    --------
    psi_results: pandas DataFrame
        DataFrame with PSI values for each feature
    """
    results = []
    
    for col in X_train.columns:
        psi_value, _ = calculate_psi(X_train[col].values, X_test[col].values)
        results.append({
            'Feature': col,
            'PSI': psi_value,
            'Interpretation': interpret_psi(psi_value)
        })
    
    return pd.DataFrame(results).sort_values('PSI', ascending=False)

def visualize_psi_distribution(X_train, X_test, feature, buckets=10):
    """
    Visualize the distribution differences and PSI contribution for a specific feature.
    
    Parameters:
    -----------
    X_train: pandas DataFrame
        Training dataset (expected distribution)
    X_test: pandas DataFrame
        Test dataset (actual distribution)
    feature: str
        Feature name to analyze
    buckets: int, optional (default=10)
        Number of buckets for PSI calculation
    """
    # Calculate PSI
    psi_value, bucket_details = calculate_psi(
        X_train[feature].values, 
        X_test[feature].values,
        buckets=buckets
    )
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: KDE of distributions
    sns.kdeplot(X_train[feature], ax=ax1, label='Train (Expected)', color='blue')
    sns.kdeplot(X_test[feature], ax=ax1, label='Test (Actual)', color='red')
    ax1.set_title(f'Distribution Comparison for {feature}')
    ax1.legend()
    
    # Plot 2: Histogram by bucket
    x = np.arange(len(bucket_details))
    width = 0.35
    
    ax2.bar(x - width/2, bucket_details['Expected_Pct'], width, label='Train (Expected)')
    ax2.bar(x + width/2, bucket_details['Actual_Pct'], width, label='Test (Actual)')
    ax2.set_title(f'Bucket Distribution - PSI = {psi_value:.4f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_details['Bucket'], rotation=45)
    ax2.set_ylabel('Proportion')
    ax2.legend()
    
    # Plot 3: PSI contribution by bucket
    ax3.bar(x, bucket_details['PSI_Contribution'], color='orange')
    ax3.set_title('PSI Contribution by Bucket')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bucket_details['Bucket'], rotation=45)
    ax3.set_ylabel('PSI Contribution')
    
    # Add text with interpretation
    ax3.text(0.02, 0.8, f'Total PSI: {psi_value:.4f}\n{interpret_psi(psi_value)}', 
             transform=ax3.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return psi_value, bucket_details

# ----------------------
# 4. Drift Experiments
# ----------------------

def experiment_with_different_drift_levels(drift_type='mean', feature_idx=0, n_drift_levels=10):
    """
    Run an experiment to measure how different levels of data drift (measured by PSI)
    impact model performance.
    
    Parameters:
    -----------
    drift_type: str, optional (default='mean')
        Type of drift to introduce
    feature_idx: int or list, optional (default=0)
        Index of feature(s) to apply drift to
    n_drift_levels: int, optional (default=10)
        Number of different drift magnitudes to test
    
    Returns:
    --------
    results_df: pandas DataFrame
        Results of the experiment
    """
    # Generate original data
    X, y = generate_data(n_samples=10000)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get baseline performance
    baseline_prob = model.predict_proba(X_val_scaled)[:, 1]
    baseline_pred = model.predict(X_val_scaled)
    baseline_auc = roc_auc_score(y_val, baseline_prob)
    baseline_accuracy = accuracy_score(y_val, baseline_pred)
    baseline_f1 = f1_score(y_val, baseline_pred)
    
    print(f"Baseline Performance - AUC: {baseline_auc:.4f}, Accuracy: {baseline_accuracy:.4f}, F1: {baseline_f1:.4f}")
    
    # Determine drift magnitudes based on drift type
    if drift_type == 'mean':
        drift_magnitudes = np.linspace(0, 2.0, n_drift_levels)
    elif drift_type == 'variance':
        drift_magnitudes = np.linspace(1.0, 3.0, n_drift_levels)
    elif drift_type == 'skew':
        drift_magnitudes = np.linspace(0, 1.0, n_drift_levels)
    elif drift_type == 'bimodal':
        drift_magnitudes = np.linspace(0, 2.0, n_drift_levels)
    else:
        drift_magnitudes = np.linspace(0, 2.0, n_drift_levels)
    
    # Initialize results
    results = []
    
    # Dictionary to store distributions for visualization
    distributions = {
        'train': X_train[f'feature_{feature_idx[0] if isinstance(feature_idx, list) else feature_idx}'].values
    }
    
    # Measure performance on different drift levels
    for drift_mag in drift_magnitudes:
        # Apply drift to validation data
        X_drift = introduce_drift(X_val, drift_type=drift_type, 
                                 feature_idx=feature_idx, 
                                 drift_magnitude=drift_mag)
        
        # Calculate PSI for the drifted feature(s)
        feature_psi_values = []
        
        if isinstance(feature_idx, list):
            for idx in feature_idx:
                feature_name = f'feature_{idx}'
                psi_val, _ = calculate_psi(X_train[feature_name].values, X_drift[feature_name].values)
                feature_psi_values.append(psi_val)
            
            # Use average PSI across features
            avg_psi = np.mean(feature_psi_values)
            max_psi = np.max(feature_psi_values)
        else:
            feature_name = f'feature_{feature_idx}'
            avg_psi, _ = calculate_psi(X_train[feature_name].values, X_drift[feature_name].values)
            max_psi = avg_psi
            
            # Store distributions for key drift points
            if drift_mag in [drift_magnitudes[0], drift_magnitudes[len(drift_magnitudes)//2], drift_magnitudes[-1]]:
                distributions[f'drift_{drift_mag:.1f}'] = X_drift[feature_name].values
        
        # Generate new target values using the true relationship
        y_drift = compute_target_with_true_relationship(X_drift)
        
        # Apply the same scaling to drifted data
        X_drift_scaled = scaler.transform(X_drift)
        
        # Evaluate model on drifted data
        drift_prob = model.predict_proba(X_drift_scaled)[:, 1]
        drift_pred = model.predict(X_drift_scaled)
        drift_auc = roc_auc_score(y_drift, drift_prob)
        drift_accuracy = accuracy_score(y_drift, drift_pred)
        drift_f1 = f1_score(y_drift, drift_pred)
        
        # Store results
        results.append({
            'Drift_Type': drift_type,
            'Drift_Magnitude': drift_mag,
            'Avg_PSI': avg_psi,
            'Max_PSI': max_psi,
            'AUC': drift_auc,
            'Accuracy': drift_accuracy,
            'F1': drift_f1,
            'AUC_Relative': drift_auc / baseline_auc,
            'Accuracy_Relative': drift_accuracy / baseline_accuracy,
            'F1_Relative': drift_f1 / baseline_f1,
            'PSI_Interpretation': interpret_psi(avg_psi)
        })
        
        print(f"Drift {drift_type} = {drift_mag:.2f}, PSI = {avg_psi:.4f}, AUC = {drift_auc:.4f}, " +
              f"Accuracy = {drift_accuracy:.4f}, F1 = {drift_f1:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    for label, data in distributions.items():
        sns.kdeplot(data, label=label)
    plt.title(f'Feature Distribution Changes ({drift_type} drift)')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Create performance vs PSI plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PSI vs AUC
    axes[0].plot(results_df['Avg_PSI'], results_df['AUC_Relative'], marker='o', linestyle='-')
    axes[0].set_title('PSI vs Relative AUC')
    axes[0].set_xlabel('PSI')
    axes[0].set_ylabel('Relative AUC (drift/baseline)')
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[0].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[0].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[0].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[0].set_ylim(0.7, 1.02)
    axes[0].legend()
    
    # PSI vs Accuracy
    axes[1].plot(results_df['Avg_PSI'], results_df['Accuracy_Relative'], marker='o', linestyle='-')
    axes[1].set_title('PSI vs Relative Accuracy')
    axes[1].set_xlabel('PSI')
    axes[1].set_ylabel('Relative Accuracy (drift/baseline)')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[1].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[1].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[1].set_ylim(0.7, 1.02)
    axes[1].legend()
    
    # PSI vs F1
    axes[2].plot(results_df['Avg_PSI'], results_df['F1_Relative'], marker='o', linestyle='-')
    axes[2].set_title('PSI vs Relative F1 Score')
    axes[2].set_xlabel('PSI')
    axes[2].set_ylabel('Relative F1 (drift/baseline)')
    axes[2].axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    axes[2].axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    axes[2].axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    axes[2].axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    axes[2].set_ylim(0.7, 1.02)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def visualize_psi_vs_drift_impact(results_list, drift_types):
    """
    Visualize the relationship between PSI and performance degradation 
    across different drift types.
    
    Parameters:
    -----------
    results_list: list of pandas DataFrames
        Results from different experiments
    drift_types: list of str
        Names of the drift types corresponding to each result DataFrame
    """
    if len(results_list) != len(drift_types):
        raise ValueError("Length of results_list must match length of drift_types")
    
    # Create one big DataFrame with all results
    for i, results in enumerate(results_list):
        results['Drift_Type'] = drift_types[i]
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Create a plot showing how different types of drift affect performance
    plt.figure(figsize=(14, 8))
    
    # Plot for each drift type
    for drift_type in drift_types:
        subset = all_results[all_results['Drift_Type'] == drift_type]
        plt.plot(subset['Avg_PSI'], subset['AUC_Relative'], marker='o', label=drift_type)
    
    # Add reference lines
    plt.axhline(y=0.95, color='r', linestyle='--', label='5% degradation')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='10% degradation')
    plt.axvline(x=0.1, color='green', linestyle='--', label='PSI = 0.1')
    plt.axvline(x=0.2, color='orange', linestyle='--', label='PSI = 0.2')
    
    plt.title('Impact of Different Drift Types on Model Performance')
    plt.xlabel('Population Stability Index (PSI)')
    plt.ylabel('Relative AUC (drift/baseline)')
    plt.ylim(0.7, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def validate_psi_on_real_data_split():
    """
    Demonstrate how PSI can be used to measure distribution differences
    between train and test splits of the same dataset.
    """
    # Generate a large dataset
    X, y = generate_data(n_samples=10000)
    
    # Create an intentionally biased train/test split
    # We'll use stratification by a binned feature to create a skewed distribution
    feature_0_bins = pd.qcut(X['feature_0'], q=5, labels=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=feature_0_bins
    )
    
    # Add additional bias to the test set
    X_test['feature_0'] = X_test['feature_0'] + 0.3  # Slight mean shift
    X_test['feature_1'] = X_test['feature_1'] * 1.2  # Slight variance increase
    
    # Calculate PSI values for all features
    psi_results = compute_feature_psi_values(X_train, X_test)
    
    print("PSI between train and test splits:")
    print(psi_results)
    
    # Visualize the most drifted feature
    most_drifted_feature = psi_results.iloc[0]['Feature']
    visualize_psi_distribution(X_train, X_test, most_drifted_feature)
    
    # Train a model and see how it performs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Performance on Test Set:")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ----------------------
# 5. Main Demo
# ----------------------

def main():
    """Main function to run the PSI demonstration"""
    print("PSI and Data Drift Monitoring Demo\n")
    
    # Part 1: Calculate PSI between train and test splits
    print("\n=== Part 1: PSI Between Train and Test Splits ===")
    validate_psi_on_real_data_split()
    
    # Part 2: Run experiments with different types of drift
    print("\n=== Part 2: Impact of Different Drift Types ===")
    
    # Mean shift experiment
    print("\nMean Shift Experiment:")
    results_mean = experiment_with_different_drift_levels(drift_type='mean', feature_idx=0)
    
    # Variance change experiment
    print("\nVariance Change Experiment:")
    results_variance = experiment_with_different_drift_levels(drift_type='variance', feature_idx=0)
    
    # Skewness experiment
    print("\nSkewness Experiment:")
    results_skew = experiment_with_different_drift_levels(drift_type='skew', feature_idx=0)
    
    # Bimodal distribution experiment
    print("\nBimodal Distribution Experiment:")
    results_bimodal = experiment_with_different_drift_levels(drift_type='bimodal', feature_idx=0)
    
    # Visualize all results together
    print("\n=== Part 3: Comparing Different Drift Types ===")
    visualize_psi_vs_drift_impact(
        [results_mean, results_variance, results_skew, results_bimodal],
        ['mean', 'variance', 'skew', 'bimodal']
    )
    
    # Part 4: Create PSI threshold guidelines
    print("\n=== Part 4: PSI Threshold Guidelines ===")
    
    # Build a summary table with PSI thresholds
    psi_thresholds = pd.DataFrame({
        'PSI_Threshold': [0.1, 0.2, 0.5],
        'Interpretation': ['Minor Change', 'Moderate Change', 'Significant Change'],
        'Action': ['Monitor', 'Investigate', 'Retrain Model'],
        'Expected_Performance_Impact': ['< 5%', '5-10%', '> 10%']
    })
    
    print("\nPSI Threshold Guidelines:")
    print(psi_thresholds)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()

# ----------------------
# 6. Educational Notes
# ----------------------

"""
UNDERSTANDING PSI AND DATA DRIFT MONITORING

What is PSI?
-----------
Population Stability Index (PSI) is a statistical measure used to quantify distribution differences
between a baseline (expected) distribution and a new (actual) distribution. It's commonly used
in model monitoring to detect data drift that could impact model performance.

PSI Formula:
-----------
PSI = SUM[ (Actual% - Expected%) * ln(Actual% / Expected%) ]

Where:
- Expected% is the percentage of observations in a bucket in the baseline distribution
- Actual% is the percentage of observations in the same bucket in the new distribution
- SUM[] means we sum this calculation across all buckets

Interpreting PSI Thresholds:
---------------------------
0.0 < PSI < 0.1: No significant change (model should be stable)
0.1 ≤ PSI < 0.2: Slight change (worth monitoring the model)
0.2 ≤ PSI < 0.5: Significant change (consider model retraining)
PSI ≥ 0.5: Severe change (model needs immediate retraining)

Types of Data Drift:
------------------
1. Mean Shift: The central tendency of a feature changes
2. Variance Change: The spread or dispersion of a feature changes
3. Skewness: The symmetry of the distribution changes
4. Multimodality: The distribution shape changes (e.g., from unimodal to bimodal)

Why PSI is Important:
-------------------
1. Single, Interpretable Metric: PSI condenses distribution changes into one number
2. Early Warning System: Helps detect data drift before model performance degrades
3. Feature-Level Insights: Can identify which specific features are experiencing drift
4. Threshold-Based Monitoring: Enables automated alerting based on preset thresholds

Best Practices for Using PSI:
---------------------------
1. Calculate PSI for each feature independently
2. Monitor trends over time, not just absolute values
3. Combine with performance monitoring metrics (AUC, accuracy, etc.)
4. Consider feature importance when prioritizing PSI alerts
5. Establish appropriate thresholds based on business context
6. Use both equal-width and equal-frequency buckets for different insights

Implementation in Production:
---------------------------
1. Store baseline distributions during model training
2. Calculate PSI periodically in production (daily/weekly/monthly)
3. Set up automated alerts when PSI exceeds thresholds
4. Investigate features with high PSI values
5. Retrain models when significant drift is detected

Limitations of PSI:
-----------------
1. Sensitive to bucket selection
2. Can be affected by small counts in buckets
3. Measures distributional changes but not relationship changes between features and target
4. May miss complex, multivariate drift patterns

This notebook demonstrates how different levels and types of data drift affect PSI values
and model performance, providing a practical guide for using PSI in model monitoring.
"""
