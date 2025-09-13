import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Define constants at the top for easy modification.
FILEPATH = "scores.csv"
AUTHOR_TO_ANALYZE = "IQexams"

def load_and_prepare_data(filepath):
    """
    Loads and prepares the test score data from a CSV file.

    This function reads the data, creates the 'Author_Time' feature,
    and cleans rows with missing essential values.

    Args:
        filepath (str): The path to the scores.csv file.

    Returns:
        pd.DataFrame: A cleaned and prepared DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please make sure the file is in the correct directory.")
        return None

    # Engineer the 'Author_Time' feature
    df["Date"] = pd.to_datetime(df["Date"])
    first_author_test_date = df.groupby('Author')['Date'].transform('min')
    df['Author_Time'] = (df['Date'] - first_author_test_date).dt.days

    # Define essential columns and drop rows with missing values in them
    vars_to_keep = ["Date", "Score", "Author", "Author_Time"]
    df.dropna(subset=vars_to_keep, inplace=True)
    
    return df

def plot_practice_effect(df, author_name):
    """
    Filters data for a specific author and plots the practice effect over time.

    Args:
        df (pd.DataFrame): The prepared DataFrame containing all author data.
        author_name (str): The name of the author to analyze (e.g., "IQexams").
    """
    # Filter the DataFrame for the specified author
    author_df = df[df["Author"] == author_name].copy()

    if author_df.empty:
        print(f"No data found for author: '{author_name}'. Cannot generate plot.")
        return

    # Create the scatter plot with a regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(data=author_df, x='Author_Time', y='Score',
                line_kws={"color": "red", "lw": 2},       # Style the regression line
                scatter_kws={"alpha": 0.6, "s": 50})     # Style the scatter points

    # Add titles and labels for clarity
    plt.title(f'Practice Effect for {author_name} Tests', fontsize=16)
    plt.xlabel('Days Since First Test by Author (Author_Time)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

def main():
    """
    Main function to execute the data loading and plotting pipeline.
    """
    # 1. Load and prepare the data
    prepared_df = load_and_prepare_data(FILEPATH)
    
    # 2. If data was loaded successfully, generate the plot
    if prepared_df is not None:
        plot_practice_effect(prepared_df, AUTHOR_TO_ANALYZE)

# Standard Python entry point
if __name__ == "__main__":
    main()