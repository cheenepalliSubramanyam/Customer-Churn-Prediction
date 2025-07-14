import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
def get_date_columns(df : pl.DataFrame):
    columns=[]
    for col in df.columns:
        if(df[col].dtype==pl.Date):
            columns.append(col)
    return columns

def classify_cols(df : pl.DataFrame,cat_features : list,decide_factor=10):
    oneHot_cols = []
    ord_cols = []
    
    # Check if num_cols is not empty
    if cat_features:
        for col in cat_features:
            # Check if the column dtype is string
            if df[col].dtype == pl.String:
                unique_count = df[col].unique().shape[0]
                if unique_count <= decide_factor:
                    oneHot_cols.append(col)
                else:
                    ord_cols.append(col)
            else:
                raise NotCategoricalError(col)
        return oneHot_cols,ord_cols
    
    else:
        raise ValueError("No columns provided to classify.")
import matplotlib.pyplot as plt
import seaborn as sns

def categorical_hist_plot(df : Union[pl.DataFrame,pd.DataFrame], cat_features : list, target_col : pl.String):
    # Determine the number of rows needed based on the number of categorical columns
    n_cols = 2
    n_rows = (len(cat_features) + 1) // n_cols  # Round up if there is an odd number of columns
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten the array of axes for easy indexing
    
    for index, col in enumerate(cat_features):
        sns.countplot(data=df, x=col, hue=target_col, ax=axes[index])
        axes[index].set_title(f'Bar Plot for {col}')
        axes[index].tick_params(axis='x', rotation=45)
        
        # Add percentage annotations on top of bars
        total = len(df[col])
        for p in axes[index].patches:
            height = p.get_height()
            axes[index].text(p.get_x() + p.get_width() / 2.,
                             height + 3,
                             '{:1.2f}%'.format((height / total) * 100),
                             ha="center")

    # Hide any unused subplots
    for j in range(index + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def categorical_violin_plot(df : Union[pl.DataFrame,pd.DataFrame], cat_features : list, target_col : pl.String):
    # Determine the number of rows needed based on the number of categorical columns
    n_cols = 2
    n_rows = (len(cat_features) + 1) // n_cols  # Round up if there is an odd number of columns
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten the array of axes for easy indexing
    
    for index, col in enumerate(cat_features):
        sns.violinplot(data=df, x=col, y=target_col, ax=axes[index])
        axes[index].set_title(f'Violin Plot for {col}')
        axes[index].tick_params(axis='x', rotation=45)
    
    # Hide any unused subplots
    for j in range(index + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
def categorical_box_plot(df : Union[pl.DataFrame,pd.DataFrame], cat_features : list, target_col : pl.String):
    for col in cat_features:
        fig, axes = plt.subplots(1, figsize=(18, 6))

        # Count Plot with hue
        sns.boxplot(data=df, x=col, y=target_col, ax=axes)
        axes.set_title(f'Bar Plot for {col}')
        axes.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
def categorical_pie_plot(df : Union[pl.DataFrame,pd.DataFrame], cat_features : list):
    for col in cat_features:
        
        fig, axes = plt.subplots(1, figsize=(18, 6))

        # Create pie chart
        df[col].plot(kind='pie', subplots=True, ax=axes, autopct='%1.1f%%', startangle=90)
        
        axes.set_title(f'Pie Chart for {col}')
        axes.set_ylabel('')
        
        plt.tight_layout()
        plt.show()

def univariavte_lineplots(df1 : Union[pl.DataFrame,pd.DataFrame], df2 : Union[pl.DataFrame,pd.DataFrame], columns : list):
    num_cols = len(columns)
    cols_per_figure = 4
    num_figures = (num_cols + cols_per_figure - 1) // cols_per_figure
    
    for i in range(num_figures):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18, 10))
        axes = axes.flatten()
        
        for j in range(cols_per_figure):
            col_idx = i * cols_per_figure + j
            if col_idx >= num_cols:
                axes[j].axis('off')
                continue

            col = columns[col_idx]
            axes[j].plot(df1[col], np.zeros_like(df1[col]), 'o')
            axes[j].plot(df2[col], np.zeros_like(df2[col]), 'o')
            axes[j].legend(["Exited", "NotExited"])
            axes[j].set_title(f'{col} Distribution')
            axes[j].set_xlabel(col)
            axes[j].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
def remove_single_value_cols(df : pl.DataFrame, cat_features : list):
    for col in cat_features:
        unique_values = df[col].drop_nulls().unique()
        
        # Check if the column has only one unique value and the rest are null
        if len(unique_values) == 1:
            df=df.drop(col)
    
    return df

from scipy.stats import ttest_ind

def perform_ttest(data : Union[pl.DataFrame,pd.DataFrame], columns: list, category1 : pl.String, category2: pl.String,target_col: pl.String, alpha=0.05):
    if isinstance(data,pd.DataFrame):
        data = pl.from_pandas(data)
    # Filter data for each category
    for column in columns:
        category1_data=data.filter(data[column]==category1)[target_col]
        category2_data=data.filter(data[column]==category2)[target_col]
        # Perform t-test
        t_stat, p_value = ttest_ind(category1_data, category2_data, equal_var=False)
        
        # Interpretation
        if p_value < alpha:
            print(f'T-statistic: {t_stat}')
            print(f'P-value: {p_value}')
            print(f"There is a significant difference in {target_col} between {category1} and {category2} for column '{column}'.")
        else:
            print(f'T-statistic: {t_stat}')
            print(f'P-value: {p_value}')
            print(f"There is no significant difference in {target_col} between {category1} and {category2} for column '{column}'.")


def compute_z_scores(data : pl.DataFrame, columns : list,threshold=3):
    for column in columns:
        mean = data[column].mean()
        std_dev = data[column].std()
        z_score_column = ((data[column] - mean) / std_dev).alias(f"{column}_zscore")
        data = data.with_columns(z_score_column)

    for column in columns:
        z_score_column = f"{column}_zscore"
        data = data.filter(data[z_score_column].abs() < threshold)

    data = data.drop([f"{column}_zscore" for column in columns])
    return data

from scipy.stats import kruskal
def perform_non_normal_kruskal_hypothesis(data : Union[pl.DataFrame,pd.DataFrame],cat_cols: list,group_col: pl.String,alpha=0.05):
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    for i in cat_cols:
        groups = [group[group_col].values for name, group in data.groupby(i)]

        # Perform Kruskal-Wallis test
        stat, p_value = kruskal(*groups)
        print(f'Kruskal-Wallis H-statistic: {stat} P-value: {p_value}')

        # Interpretation
        if p_value < alpha:
            print(f"There is a significant difference in {group_col} between different {i} categories.\n")
        else:
            print(f"There is no significant difference in {group_col} between different {i} categories.\n")

from sklearn.metrics import matthews_corrcoef
def plot_binarycols_heatmap(data : Union[pl.DataFrame,pd.DataFrame],binary_cols : list):
    cols_width= data[binary_cols].columns
    phi_matrix = np.zeros((len(cols_width),len(cols_width)))
    for i in range(len(cols_width)):
        for j in range(len(cols_width)):
            phi_matrix[i][j]=matthews_corrcoef(data[cols_width[i]],data[cols_width[j]])
    plt.figure(figsize=(18,5))
    sns.heatmap(data=phi_matrix,annot=True,xticklabels=cols_width,yticklabels=cols_width)
    plt.title("matthews corrcoef between binary columns")
    plt.show()

if __name__ == "__main__":
    class NotCategoricalError(Exception):
        """Exception raised for non-categorical columns."""

        def __init__(self, column):
            self.column = column
            super().__init__(f"{column} is not a categorical column.")