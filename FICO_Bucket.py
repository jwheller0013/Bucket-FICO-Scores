import pandas as pd
import numpy as np

def load_data():
    try:
        df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
    except FileNotFoundError:
        print("File not found. Please ensure the file is in the correct directory.")
        return None
    return df