import pandas as pd 
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
def calculate_vif(data : Union[pl.DataFrame,pd.DataFrame],target_col,head=5):
    if isinstance(data,pl.DataFrame):
        data=data.to_pandas()
    num_cols=[col for col in data.drop(target_col,axis=1).columns if data[col].dtype!= pl.String]
    df = sm.add_constant(data[num_cols])
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif.sort_values(by='VIF',ascending=False).head(head)

def num_scatter_plot(df : Union[pl.DataFrame,pd.DataFrame],num_features : list,target_col : pl.String):
    for col in num_features:
        fig, axes = plt.subplots(1, figsize=(18, 6))

        sns.regplot(data=df, x=col,y=target_col, ax=axes)
        axes.set_title(f'Bar Plot for {col}')
        axes.tick_params(axis='x', rotation=45)
        
        # Add percentage annotations on top of bars
        total = len(df[col])
        for p in axes.patches:
            height = p.get_height()
            axes.text(p.get_x() + p.get_width() / 2.,
                        height + 3,
                        '{:1.2f}%'.format((height / total) * 100),
                        ha="center")

        plt.tight_layout()
        plt.show()
def num_dodge_plot(data : Union[pl.DataFrame,pd.DataFrame], num_cols: list, target_col: pl.String):
    if isinstance(data,pl.DataFrame):
        data=data.to_pandas()
    # Determine the number of rows needed based on the number of numerical columns
    n_cols = 2
    n_rows = (len(num_cols) + 1) // n_cols  # Round up if there is an odd number of columns
    
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
    ax = ax.flatten()  # Flatten the array of axes for easy indexing
    
    for index, col in enumerate(num_cols):
        sns.histplot(data, x=col, hue=target_col, multiple="dodge",fill=True, ax=ax[index])
        ax[index].set_title(f'Distribution of {col} by {target_col}')
    
    # Hide any unused subplots
    for j in range(index + 1, len(ax)):
        fig.delaxes(ax[j])
    
    plt.tight_layout()
    plt.show()