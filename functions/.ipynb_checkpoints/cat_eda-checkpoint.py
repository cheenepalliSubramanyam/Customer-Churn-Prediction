import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
def cat_fill_mode(df,features):
    for col in features:
        df[col].fillna(df[col].mode().iloc[0],inplace=True)
        
def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorial-categorial association."""
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    
def cramers_v_matrix(df, cat_cols):
    """Build Cramér's V matrix for categorical columns."""
    matrix = np.zeros((len(cat_cols), len(cat_cols)))
    for i, col1 in enumerate(cat_cols):
        for j, col2 in enumerate(cat_cols):
            matrix[i, j] = cramers_v(df[col1], df[col2])
    return matrix
    
def plot_heatmap(matrix, columns):
    """Plot heatmap for Cramér's V matrix."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=columns, yticklabels=columns)
    plt.title("Cramér's V Association Matrix")
    plt.show()
