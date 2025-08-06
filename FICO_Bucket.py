import pandas as pd
import numpy as np

def load_data():
    try:
        df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
    except FileNotFoundError:
        print("File not found. Please ensure the file is in the correct directory.")
        return None
    return 

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