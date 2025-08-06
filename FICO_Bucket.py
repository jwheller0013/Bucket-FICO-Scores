import pandas as pd
import numpy as np

def load_data():
    try:
        df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
    except FileNotFoundError:
        print("File not found. Please ensure the file is in the correct directory.")
        return None
    return df

#Decided to use Mean Squared Error (MSE) as the evaluation metric

class FICOQuantizer:
    """
    FICO Score Quantization using Mean Squared Error (MSE) optimization.
    
    The MSE method minimizes the total squared error by finding optimal bucket boundaries
    such that the sum of squared differences between each FICO score and its bucket mean
    is minimized.
    """
    
    def __init__(self, n_buckets=10):
        self.n_buckets = n_buckets
        self.boundaries = None
        self.bucket_stats = None
        self.total_mse = None

    def mse_quantization(self, fico_scores):
        """
        Mean Squared Error quantization using dynamic programming.
        
        Minimizes: Σ(xi - bucket_mean)²
        
        Returns optimal bucket boundaries.
        """
        print(f"Performing MSE quantization with {self.n_buckets} buckets...")
        
        # Sort scores for processing
        sorted_scores = np.sort(fico_scores)
        n = len(sorted_scores)
        
        # Dynamic programming approach
        # dp[i][j] = minimum MSE for first i points using j buckets
        dp = np.full((n + 1, self.n_buckets + 1), np.inf)
        split_points = np.zeros((n + 1, self.n_buckets + 1), dtype=int)
        
        # Base case: 0 points, 0 buckets
        dp[0][0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i, self.n_buckets) + 1):
                # Try all possible previous split points
                for k in range(j - 1, i):
                    # Calculate MSE for bucket from k to i-1
                    bucket_scores = sorted_scores[k:i]
                    bucket_mean = np.mean(bucket_scores)
                    bucket_mse = np.sum((bucket_scores - bucket_mean) ** 2)
                    
                    total_mse = dp[k][j-1] + bucket_mse
                    
                    if total_mse < dp[i][j]:
                        dp[i][j] = total_mse
                        split_points[i][j] = k
        
        # Reconstruct optimal boundaries
        boundaries = [sorted_scores[0]]  # Start with minimum
        i, j = n, self.n_buckets
        
        splits = []
        while j > 1:
            split_point = split_points[i][j]
            splits.append(split_point)
            i, j = split_point, j - 1
        
        # Convert split indices to actual FICO scores
        for split_idx in reversed(splits):
            if split_idx > 0:
                boundaries.append(sorted_scores[split_idx])
        
        boundaries.append(sorted_scores[-1])  # End with maximum
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Store total MSE
        self.total_mse = dp[n][self.n_buckets]
        print(f"Optimal Total MSE: {self.total_mse:.2f}")
        
        return boundaries
    
    def fit(self, fico_scores, default_flags=None):
        """
        Fit the MSE quantizer to the data.
        
        Parameters:
        - fico_scores: array of FICO scores
        - default_flags: array of default indicators (optional, for analysis)
        """
        self.boundaries = self.mse_quantization(fico_scores)
        
        # Calculate bucket statistics
        self._calculate_bucket_stats(fico_scores, default_flags)
        
        return self
    
    def _calculate_bucket_stats(self, fico_scores, default_flags=None):
        """Calculate statistics for each bucket."""
        df = pd.DataFrame({'fico': fico_scores})
        if default_flags is not None:
            df['default'] = default_flags
        
        # Assign buckets (ratings)
        df['bucket'] = pd.cut(df['fico'], bins=self.boundaries, include_lowest=True, labels=False)
        df['bucket'] = df['bucket'] + 1  # Start ratings from 1
        
        # Calculate statistics for each bucket
        stats = []
        for rating in sorted(df['bucket'].unique()):
            bucket_data = df[df['bucket'] == rating]
            bucket_scores = bucket_data['fico']
            
            # Calculate MSE for this bucket
            bucket_mean = bucket_scores.mean()
            bucket_mse = np.sum((bucket_scores - bucket_mean) ** 2)
            
            stat_dict = {
                'Rating': rating,
                'FICO_Range': f"{self.boundaries[rating-1]:.0f}-{self.boundaries[rating]:.0f}",
                'Count': len(bucket_data),
                'Count_Pct': len(bucket_data) / len(df) * 100,
                'Min_FICO': bucket_scores.min(),
                'Max_FICO': bucket_scores.max(),
                'Avg_FICO': bucket_mean,
                'Std_FICO': bucket_scores.std(),
                'Bucket_MSE': bucket_mse,
            }
            
            # Add default statistics if available
            if default_flags is not None:
                default_rate = bucket_data['default'].mean()
                stat_dict.update({
                    'Default_Rate': default_rate * 100,
                    'Default_Count': int(bucket_data['default'].sum()),
                    'Good_Count': int((bucket_data['default'] == 0).sum()),
                })
            
            stats.append(stat_dict)
        
        self.bucket_stats = pd.DataFrame(stats)

    def get_rating(self, fico_score):
        """
        Get rating for a FICO score.
        
        Parameters:
        - fico_score: FICO score to rate
        
        Returns:
        - rating: integer rating (1=best, higher=worse)
        """
        if self.boundaries is None:
            raise ValueError("Quantizer not fitted yet! Call fit() first.")
        
        # Find which bucket the score falls into
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= fico_score <= self.boundaries[i + 1]:
                return i + 1
        
        # Handle edge cases
        if fico_score < self.boundaries[0]:
            return len(self.boundaries) - 1  # Worst rating for very low scores
        elif fico_score > self.boundaries[-1]:
            return 1  # Best rating for very high scores
        
        return len(self.boundaries) - 1
    
    def transform(self, fico_scores):
        """Transform multiple FICO scores to ratings."""
        if isinstance(fico_scores, (int, float)):
            return self.get_rating(fico_scores)
        
        return [self.get_rating(score) for score in fico_scores]
    
    def display_results(self):
        """Display quantization results."""
        if self.bucket_stats is None:
            print("Quantizer not fitted yet!")
            return
        
        print("\n" + "="*100)
        print("FICO SCORE MSE QUANTIZATION RESULTS")
        print("="*100)
        print("Note: Rating 1 = Best credit score, Higher rating = Worse credit score")
        print(f"Total MSE: {self.total_mse:.2f}")
        print("-"*100)
        
        # Format display
        display_cols = ['Rating', 'FICO_Range', 'Count', 'Count_Pct', 'Avg_FICO', 'Std_FICO', 'Bucket_MSE']
        if 'Default_Rate' in self.bucket_stats.columns:
            display_cols.extend(['Default_Count', 'Default_Rate'])
        
        display_df = self.bucket_stats[display_cols].copy()
        
        # Round numeric columns
        for col in ['Count_Pct', 'Avg_FICO', 'Std_FICO', 'Bucket_MSE', 'Default_Rate']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        print(display_df.to_string(index=False))
        print("-"*100)
        
        print(f"Bucket Boundaries: {[round(b, 1) for b in self.boundaries]}")
        print(f"Number of Buckets: {len(self.boundaries) - 1}")

def test_fico_rating(fico_score, quantizer):
    """Test rating for a specific FICO score with detailed output."""
    rating = quantizer.get_rating(fico_score)
    
    # Get bucket information
    bucket_info = quantizer.bucket_stats[quantizer.bucket_stats['Rating'] == rating].iloc[0]
    
    print(f"FICO SCORE RATING ANALYSIS")
    print(f"FICO Score: {fico_score}")
    print(f"Rating: {rating} (1=Best, Higher=Worse)")
    print(f"Rating Range: {bucket_info['FICO_Range']}")
    print(f"Average FICO in Rating: {bucket_info['Avg_FICO']:.1f}")
    print(f"Standard Deviation: {bucket_info['Std_FICO']:.1f}")
    print(f"Population in Rating: {bucket_info['Count']:,} ({bucket_info['Count_Pct']:.1f}%)")
    
    if 'Default_Rate' in bucket_info:
        print(f"Expected Default Rate: {bucket_info['Default_Rate']:.2f}%")
    
    return rating

print("FICO SCORE MSE QUANTIZATION SYSTEM")
# Load data
df = load_data()

print(f"\nDataset Overview:")
print(f"FICO score range: {df['fico_score'].min()} - {df['fico_score'].max()}")
print(f"Default rate: {df['default'].mean():.2%}")
print(f"Sample size: {len(df):,}")

# Initialize and fit quantizer
quantizer = FICOQuantizer(n_buckets=8)
quantizer.fit(df['fico_score'], df['default'])

# Display results
quantizer.display_results()

# Test individual FICO scores
test_scores = [300, 500, 620, 680, 720, 780, 850]
print(f"\n{'='*80}")
print("TESTING INDIVIDUAL FICO SCORES")
print(f"{'='*80}")

for score in test_scores:
    test_fico_rating(score, quantizer)

# Simple function for easy use
def get_fico_rating(fico_score):
    """
    Get FICO rating using MSE quantization.
    
    Parameters:
    - fico_score: FICO score to rate
    
    Returns:
    - rating: integer rating (1=best, higher=worse)
    """
    return quantizer.get_rating(fico_score)

# Integration with loan assessment
def enhanced_loan_assessment_with_rating(loan_data):
    """
    Enhanced loan assessment that includes MSE-optimized FICO rating.
    """
    fico_score = loan_data['fico_score']
    fico_rating = get_fico_rating(fico_score)
    
    # Add rating to loan data
    enhanced_loan_data = loan_data.copy()
    enhanced_loan_data['fico_rating'] = fico_rating
    
    # Get rating statistics
    bucket_info = quantizer.bucket_stats[quantizer.bucket_stats['Rating'] == fico_rating].iloc[0]
    
    print(f"\n{'='*60}")
    print("ENHANCED LOAN ASSESSMENT WITH MSE FICO RATING")
    print(f"{'='*60}")
    print(f"FICO Score: {fico_score}")
    print(f"MSE-Optimized Rating: {fico_rating} (1=Best, Higher=Worse)")
    print(f"Rating Range: {bucket_info['FICO_Range']}")
    print(f"Avg FICO in Rating: {bucket_info['Avg_FICO']:.1f}")
    
    if 'Default_Rate' in bucket_info:
        print(f"Expected Default Rate: {bucket_info['Default_Rate']:.2f}%")
    
    return enhanced_loan_data, fico_rating

# Example usage
sample_loan = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 15000.0,
    'total_debt_outstanding': 5000.0,
    'income': 75000.0,
    'years_employed': 5,
    'fico_score': 680
}

enhanced_data, rating = enhanced_loan_assessment_with_rating(sample_loan)

print(f"\n{'='*80}")
print("MSE QUANTIZATION SYSTEM READY!")
print(f"{'='*80}")
print("Use get_fico_rating(fico_score) to get MSE-optimized ratings")
print("Use enhanced_loan_assessment_with_rating(loan_data) for full analysis")

# Quick demonstration
print(f"\nQuick Examples:")
for score in [600, 700, 800]:
    rating = get_fico_rating(score)
    print(f"FICO {score} -> Rating {rating}")