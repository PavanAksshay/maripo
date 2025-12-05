"""
Interactive Data Analysis Notebook
Author: 24f2002734@ds.study.iitm.ac.in

This notebook demonstrates the relationship between variables in a dataset
using interactive widgets and reactive cell dependencies.
"""

import marimo

__generated_with = "0.1.0"
app = marimo.App()


# ============================================================================
# CELL 1: Import Libraries and Generate Sample Dataset
# ============================================================================
# This cell sets up the environment and creates a synthetic dataset
# for demonstration purposes. The dataset contains two correlated variables.
@app.cell
def __():
    import marimo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic dataset with correlated variables
    n_samples = 500
    x = np.random.normal(50, 15, n_samples)
    # Create y with a linear relationship to x plus noise
    noise = np.random.normal(0, 10, n_samples)
    y = 2 * x + 30 + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Variable_X': x,
        'Variable_Y': y,
        'Category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Variable_X range: [{df['Variable_X'].min():.2f}, {df['Variable_X'].max():.2f}]")
    print(f"Variable_Y range: [{df['Variable_Y'].min():.2f}, {df['Variable_Y'].max():.2f}]")
    return marimo, np, pd, plt, sns, stats, n_samples, x, noise, y, df


# ============================================================================
# CELL 2: Interactive Slider Widget
# ============================================================================
# This cell creates an interactive slider that controls the sample size
# for analysis. The value from this widget will be used by dependent cells.
@app.cell
def __(marimo):
    # Create slider widget for sample size selection
    # This widget value will be used by dependent cells below
    sample_size = marimo.ui.slider(
        50,  # minimum value
        500,  # maximum value
        200,  # default value
        step=10,  # step size
        label="Sample Size for Analysis"
    )
    return sample_size,


# ============================================================================
# CELL 3: Data Processing (Depends on slider)
# ============================================================================
# This cell depends on the sample_size widget from Cell 2.
# It samples the dataset based on the slider value and calculates statistics.
@app.cell
def __(df, sample_size):
    # DATA FLOW: This cell receives sample_size from Cell 2 (slider widget)
    # When the slider changes, this cell automatically re-executes
    
    # Sample the dataset based on slider value
    sampled_df = df.sample(n=min(sample_size.value, len(df)), random_state=42)
    
    # Calculate correlation coefficient
    correlation = sampled_df['Variable_X'].corr(sampled_df['Variable_Y'])
    
    # Calculate linear regression statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sampled_df['Variable_X'], 
        sampled_df['Variable_Y']
    )
    
    # Calculate summary statistics
    x_mean = sampled_df['Variable_X'].mean()
    y_mean = sampled_df['Variable_Y'].mean()
    x_std = sampled_df['Variable_X'].std()
    y_std = sampled_df['Variable_Y'].std()
    
    return sampled_df, correlation, slope, intercept, r_value, p_value, std_err, x_mean, y_mean, x_std, y_std


# ============================================================================
# CELL 4: Visualization (Depends on processed data)
# ============================================================================
# This cell depends on sampled_df from Cell 3, which in turn depends on 
# sample_size from Cell 2. This creates a chain of dependencies.
@app.cell
def __(sampled_df, slope, intercept, plt, sns, np):
    # DATA FLOW: This cell receives sampled_df, slope, and intercept from Cell 3
    # When Cell 3 updates (due to slider change), this cell automatically updates
    
    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(
        data=sampled_df,
        x='Variable_X',
        y='Variable_Y',
        hue='Category',
        alpha=0.6,
        ax=ax
    )
    
    # Add regression line
    x_line = np.linspace(sampled_df['Variable_X'].min(), sampled_df['Variable_X'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression Line')
    
    ax.set_xlabel('Variable X', fontsize=12)
    ax.set_ylabel('Variable Y', fontsize=12)
    ax.set_title('Relationship Between Variable X and Variable Y', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_figure = plt.gcf()
    return fig, ax, plot_figure, x_line, y_line


# ============================================================================
# CELL 5: Dynamic Markdown Output (Depends on statistics)
# ============================================================================
# This cell creates dynamic markdown output that updates based on widget state
# and calculated statistics from dependent cells.
@app.cell
def __(marimo, sample_size, correlation, r_value, p_value, slope, intercept, 
       x_mean, y_mean, x_std, y_std, sampled_df):
    # DATA FLOW: This cell receives multiple values from Cells 2 and 3
    # It dynamically generates markdown content based on current widget state
    
    # Determine relationship strength based on correlation
    if abs(correlation) > 0.7:
        strength = "strong"
    elif abs(correlation) > 0.4:
        strength = "moderate"
    else:
        strength = "weak"
    
    # Determine direction
    direction = "positive" if correlation > 0 else "negative"
    
    # Statistical significance interpretation
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not statistically significant (p ≥ 0.05)"
    
    # Create dynamic markdown output
    markdown_output = marimo.md(f"""
    ## 📊 Data Analysis Results
    
    ### Dataset Information
    - **Sample Size:** {sample_size.value} observations
    - **Total Available:** {len(sampled_df)} samples analyzed
    
    ### Statistical Summary
    
    **Variable X:**
    - Mean: {x_mean:.2f}
    - Standard Deviation: {x_std:.2f}
    
    **Variable Y:**
    - Mean: {y_mean:.2f}
    - Standard Deviation: {y_std:.2f}
    
    ### Relationship Analysis
    
    - **Correlation Coefficient (r):** {correlation:.4f}
    - **R-squared (r²):** {r_value**2:.4f}
    - **P-value:** {p_value:.6f}
    
    **Interpretation:**
    The relationship between Variable X and Variable Y is **{strength}** and **{direction}**.
    The relationship is **{significance}**.
    
    ### Regression Equation
    
    The linear regression model is:
    
    ```
    Y = {slope:.4f} × X + {intercept:.4f}
    ```
    
    This means that for every unit increase in Variable X, Variable Y changes by {slope:.4f} units.
    
    ---
    *Analysis performed by: 24f2002734@ds.study.iitm.ac.in*
    """)
    
    return direction, markdown_output, significance, strength


# ============================================================================
# CELL 6: Additional Analysis - Distribution Comparison
# ============================================================================
# This cell creates additional visualizations showing distributions
@app.cell
def __(sampled_df, plt, sns):
    # Create distribution plots
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution of Variable X
    axes[0].hist(sampled_df['Variable_X'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Variable X', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Variable X')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of Variable Y
    axes[1].hist(sampled_df['Variable_Y'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Distribution of Variable Y', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Variable Y')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    distribution_plot = plt.gcf()
    return axes, distribution_plot, fig2


# ============================================================================
# CELL 7: Display Widget and Results
# ============================================================================
# This cell displays the interactive widget and the markdown output
@app.cell
def __(sample_size, markdown_output):
    # Display the slider widget
    sample_size
    
    # Display the dynamic markdown output
    markdown_output
    return


# ============================================================================
# CELL 8: Display Visualizations
# ============================================================================
@app.cell
def __(plot_figure, distribution_plot):
    # Display the scatter plot with regression line
    plot_figure
    
    # Display the distribution plots
    distribution_plot
    return


if __name__ == "__main__":
    app.run()

