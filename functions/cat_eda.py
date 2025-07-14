from functions.pipeline_helpers import get_features
import sys
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from typing import Union

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k-1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def cramers_v_matrix(df: Union[pl.DataFrame, pd.DataFrame], cat_cols: list):
    """Build Cramér's V matrix for categorical columns."""
    matrix = np.zeros((len(cat_cols), len(cat_cols)))
    for i, col1 in enumerate(cat_cols):
        for j, col2 in enumerate(cat_cols):
            matrix[i, j] = cramers_v(df[col1], df[col2])
    return matrix
    
def plot_Association_heatmap(df: Union[pl.DataFrame, pd.DataFrame], columns: list):
    """Plot heatmap for Cramér's V matrix."""
    matrix = cramers_v_matrix(df, columns)
    plt.figure(figsize=(18, 5))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=columns, yticklabels=columns)
    plt.title("Cramér's V Association Matrix")
    plt.show()

def plot_evaluation(fpr, tpr,y_true,y_pred, auc=0.5,title=''):
    """
    Plots a ROC curve given the false positive rate (fpr) and 
    true positive rate (tpr) of a classifier, including the AUC.
    """
    fig,ax = plt.subplots(ncols=2,figsize=(18,5))
    # Plot ROC curve
    ax[0].plot(fpr, tpr, color='green', label=f'ROC (AUC = {auc:.3f})')
    
    # Plot line with no predictive power (baseline)
    ax[0].plot([0, 1], [0, 1], color='red', linestyle='--', label=f'Guess AUC {0.5} ')
    
    # Customize the plot
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=15)
    ax[0].legend()
    fig.suptitle(title, fontsize=18, y=1.02) 
    # Plot Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
    fig=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_pred))
    fig.plot(ax=ax[1])
    plt.show()


def cat_proportion_plot(data: Union[pl.DataFrame, pd.DataFrame], cat_cols: list, target_col: str):
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a Polars or Pandas DataFrame")
    
    # Determine the number of rows needed based on the number of categorical columns
    n_cols = 2
    n_rows = (len(cat_cols) + 1) // n_cols  # Round up if there is an odd number of columns
    
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
    ax = ax.flatten()  # Flatten the array of axes for easy indexing
    
    for index, col in enumerate(cat_cols):
        prop_df = data.groupby([col, target_col]).size().unstack().apply(lambda x: x / x.sum(), axis=1)
        prop_df.plot(kind='bar', stacked=True, ax=ax[index])
        ax[index].set_title(f'Proportion of {target_col} by {col}')
        ax[index].set_ylabel('Proportion')
        ax[index].set_xlabel(col)
        ax[index].legend(['No', 'Yes'], loc='upper right')

    # Hide any unused subplots
    for j in range(index + 1, len(ax)):
        fig.delaxes(ax[j])
    
    plt.tight_layout()
    plt.show()

from matplotlib.colors import ListedColormap

def decision_boundary(X,y,feature1,feature2,classifier,transformer=None,mesh_step_size=1,title=''):

    # Create a mesh grid
    h = mesh_step_size  # step size in the mesh
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Prepare the data for the prediction on the mesh grid
    grid_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature1, feature2])

    # Add other features with default values to match the original dataset
    for col in X.columns:
        if col not in [feature1, feature2]:
            grid_data[col] = X[col].mean()  # or some default value
    if transformer is None:
        grid_data_transformed = grid_data
        grid_data_transformed=grid_data_transformed[X.columns] # Same feature order
    # Apply the same transformations to the grid data
    else:   
        grid_data_transformed = transformer.transform(grid_data)
        grid_data_transformed = pd.DataFrame(grid_data_transformed,columns=get_features(transformer))

    # Predict the classes for each point in the mesh grid
    Z = classifier.predict(grid_data_transformed)
    Z = Z.reshape(xx.shape)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ['#FF0000', '#00FF00']

    plt.figure(figsize=(18, 5))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)

    # Plot theing points
    scatter = plt.scatter(X[feature1], X[feature2], c=y, edgecolors='k', marker='o', cmap=ListedColormap(cmap_bold))

    # Adjust y-axis limits
    plt.ylim(y_min, y_max)

    # Add more tick labels
    plt.yticks(np.arange(y_min, y_max + 1, 5))  # Adjust the increment as needed

    # Label the y-axis
    plt.ylabel(feature2)

    plt.xlabel(feature1)
    plt.title(f'{title} Decision Boundary')

    # Create a legend
    handles = scatter.legend_elements()[0]
    labels = ['Class 0', 'Class 1']
    plt.legend(handles, labels)

    plt.show()