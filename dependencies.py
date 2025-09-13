import pandas as pd
from scipy import stats
import numpy as np
import itertools

def analyze_dependencies(file_path):
    """
    Loads test score data and performs a comprehensive dependency analysis.

    This function conducts three main types of analysis:
    1. Chi-Squared Test: For pairs of categorical variables to test for independence.
       It also calculates Cramér's V to measure the strength of association.
    2. Relative Risk: Calculates the likelihood of achieving a high score (>150 and >160)
       based on different conditions (e.g., Cold vs. Hot months).
    3. ANOVA: Tests if the mean scores are significantly different across
       different categorical groups (e.g., Timed vs. Untimed tests).

    Args:
        file_path (str): The path to the CSV file containing the test score data.

    Returns:
        None: Prints the results of the analysis to the console.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Data Preparation ---
    # Create binary columns for high scores, which are needed for Relative Risk analysis.
    df['Score > 140'] = df['Score'] > 140 
    df['Score > 150'] = df['Score'] > 150
    df['Score > 160'] = df['Score'] > 160

    # Define the categorical columns we want to analyze.
    # We exclude coded columns (like AuthorCode) and individual author flags
    # as we have a primary 'Author' and 'TestType' column.
    categorical_cols = [
        'TimedUntimed', 'ColdHot', 'TestType', 'Author', 'Recent', 'Cold'
    ]
    # Filter out any columns that might not be in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]


    # --- 1. Chi-Squared Test for Independence (Categorical vs. Categorical) ---
    print("="*80)
    print("1. Chi-Squared Test for Independence (p-value < 0.05 suggests dependence)")
    print("   Cramér's V measures strength of association (0=none, 1=perfect).")
    print("="*80)

    # Get all unique pairs of categorical columns
    cat_pairs = list(itertools.combinations(categorical_cols, 2))
    significant_chi2_results = []

    for col1, col2 in cat_pairs:
        # Create a contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        # Perform the Chi-Squared test
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for association strength
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))

            if p < 0.05:
                significant_chi2_results.append((col1, col2, p, cramers_v))
        except ValueError:
            # This can happen if a category has no observations
            print(f"Could not perform Chi-Squared test for '{col1}' vs '{col2}'. Skipping.")
            continue

    # Sort results by p-value for clarity
    significant_chi2_results.sort(key=lambda x: x[2])
    if significant_chi2_results:
        for col1, col2, p, cramers_v in significant_chi2_results:
            print(f"\n- Found SIGNIFICANT dependence between '{col1}' and '{col2}':")
            print(f"  p-value: {p:.4f}")
            print(f"  Cramér's V: {cramers_v:.4f}")
    else:
        print("\nNo significant dependencies found between the tested categorical variables.")


    # --- 2. Relative Risk (Categorical vs. Binned Numerical) ---
    print("\n" + "="*80)
    print("2. Relative Risk Analysis")
    print("   RR > 1: Event is more likely in the first group.")
    print("   RR < 1: Event is less likely in the first group.")
    print("="*80)

    def calculate_relative_risk(dataframe, group_col, outcome_col):
        """Helper function to calculate and print relative risk."""
        contingency_table = pd.crosstab(dataframe[group_col], dataframe[outcome_col])
        
        # Ensure the table has the expected shape (2x2)
        if contingency_table.shape != (2, 2):
            # print(f"\nSkipping Relative Risk for '{group_col}' vs '{outcome_col}' (not a 2x2 table).")
            return

        # contingency_table[True] is outcome occurred, [False] is outcome did not occur
        # contingency_table.loc['Group1'] are observations in the first group
        group1_name, group2_name = contingency_table.index
        
        # P(Outcome | Group1) = (exposed with outcome) / (total exposed)
        prob_group1 = contingency_table.loc[group1_name, True] / contingency_table.loc[group1_name].sum()
        # P(Outcome | Group2) = (unexposed with outcome) / (total unexposed)
        prob_group2 = contingency_table.loc[group2_name, True] / contingency_table.loc[group2_name].sum()

        if prob_group2 > 0:
            rr = prob_group1 / prob_group2
            print(f"\n- Analysis for '{outcome_col}' based on '{group_col}':")
            print(f"  Comparing '{group1_name}' to '{group2_name}':")
            print(f"  Relative Risk (RR): {rr:.2f}")
            if rr > 1:
                print(f"  Interpretation: The event is {rr:.2f} times MORE likely to occur in the '{group1_name}' group.")
            elif rr < 1:
                 print(f"  Interpretation: The event is {1/rr:.2f} times LESS likely to occur in the '{group1_name}' group.")
            else:
                print("  Interpretation: There is no difference in risk between the groups.")
        else:
            print(f"\n- Cannot calculate Relative Risk for '{outcome_col}' based on '{group_col}' (division by zero).")

    # Perform RR for key binary variables vs. high score outcomes
    for group in ['ColdHot', 'TimedUntimed', 'Recent']:
        if group in df.columns:
            calculate_relative_risk(df, group, 'Score > 140')
            calculate_relative_risk(df, group, 'Score > 150')
            calculate_relative_risk(df, group, 'Score > 160')


    # --- 3. ANOVA (Categorical vs. Continuous Numerical) ---
    print("\n" + "="*80)
    print("3. ANOVA Test (Analysis of Variance)")
    print("   Tests if the mean 'Score' is significantly different across groups.")
    print("   p-value < 0.05 suggests the mean scores are NOT all equal.")
    print("="*80)

    significant_anova_results = []
    for col in categorical_cols:
        # Group scores by each category in the column
        groups = [df['Score'][df[col] == category].dropna() for category in df[col].unique()]
        
        # Ensure there is more than one group to compare
        if len(groups) > 1:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                if p_value < 0.05:
                    significant_anova_results.append((col, p_value))
            except ValueError:
                print(f"Could not perform ANOVA for 'Score' by '{col}'. Skipping.")

    # Sort results by p-value
    significant_anova_results.sort(key=lambda x: x[1])
    if significant_anova_results:
        for col, p in significant_anova_results:
            print(f"\n- Found SIGNIFICANT difference in mean 'Score' based on '{col}':")
            print(f"  p-value: {p:.4f}")
            # Also print the means for context
            print("  Group Means:")
            print(df.groupby(col)['Score'].mean().round(2))
    else:
        print("\nNo significant differences in mean scores found across the tested groups.")


# --- Main Execution ---
if __name__ == "__main__":
    # The user uploaded this file, so we will reference it directly.
    file_path = 'scores.csv'
    analyze_dependencies(file_path)

