import statsmodels.api as sm
import statsmodels.stats.weightstats as smstats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def load_and_prepare_data(filepath):
    """
    Loads data from a CSV, selects necessary columns, and drops missing values.
    
    Args:
        filepath (str): The path to the scores.csv file.
        
    Returns:
        pd.DataFrame: A cleaned and prepared DataFrame.
    """
    da = pd.read_csv(filepath)
    
    vars_to_use = ["Date", "Score", "Timed", "M", "V", "N", "S", "L", "Cold", "Recent",
                   "Cooijmans", "Ivec", "Betts", "IQexams", "Backlund", "Joshi", "Dorsey",
                   "Predavec", "Jouve", "Kutle", "Prousalis", "Scillitani", "Udriste",
                   "OtherAuthor", "Author", "TestType", "TimedUntimed", "ColdHot",
                   "counter", "AuthorCode", "TestTypeCode"]
                   
    return da[vars_to_use].dropna()

def engineer_date_features(df):
    """
    Converts date columns and engineers numerical time-based features.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with new time-based columns.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df['Date_numerical'] = (df['Date'] - df['Date'].min()).dt.days
    
    first_author_test_date = df.groupby('Author')['Date'].transform('min')
    df['Author_Time'] = (df['Date'] - first_author_test_date).dt.days
    
    return df

def perform_statistical_tests(df):
    """
    Performs and prints the results of various descriptive stats and t-tests.
    
    Args:
        df (pd.DataFrame): The main DataFrame.
    """
    print("\n--- Descriptive Statistics and T-Tests ---")
    
    # Create subsets for comparison
    recent_avg = df.query("counter > 197")["Score"]
    early_avg = df.query("counter <= 197")["Score"]
    timed_scores = df[df["Timed"] == 1]["Score"]
    untimed_scores = df[df["Timed"] == 0]["Score"]
    cold_scores = df[df["Cold"] == 1]["Score"]
    hot_scores = df[df["Cold"] == 0]["Score"]
    l_scores = df[df["L"] == 1]["Score"]
    n_scores = df[df["N"] == 1]["Score"]

    # Print means, medians, etc.
    print(f"Timed mean, median, sd: {timed_scores.mean():.2f}, {timed_scores.median():.2f}, {timed_scores.std():.2f}")
    print(f"Untimed mean, median, sd: {untimed_scores.mean():.2f}, {untimed_scores.median():.2f}, {untimed_scores.std():.2f}")

    # Print t-test results
    print("\nIs the first group's mean larger than the second's?")
    print("Recent vs. Early t-test:", smstats.ttest_ind(recent_avg, early_avg, alternative="larger", usevar="unequal"))
    print("Untimed vs. Timed t-test:", smstats.ttest_ind(untimed_scores, timed_scores, alternative="larger", usevar="unequal"))
    print("Cold vs. Hot t-test:", smstats.ttest_ind(cold_scores, hot_scores, alternative="larger", usevar="unequal"))
    print("Logical vs. Numerical t-test:", smstats.ttest_ind(l_scores, n_scores, alternative="larger", usevar="unequal"))

def run_statistical_models(df):
    """
    Runs and prints the summaries of various GEE and OLS models.
    
    Args:
        df (pd.DataFrame): The main DataFrame with engineered features.
    """
    print("\n--- GEE Model (Score ~ Time variables) ---")
    model_gee_time = sm.GEE.from_formula(
        "Score ~ Date_numerical + Author_Time",
        groups=df["Author"],
        cov_struct=sm.cov_struct.Exchangeable(),
        data=df
    )
    result_gee_time = model_gee_time.fit()
    print(result_gee_time.summary())

    print("\n--- Individual Author Analysis (OLS Loop) ---")
    for author in df["Author"].unique():
        print(f"\n--- Analysis for Author: {author} ---")
        author_df = df[df["Author"] == author]
        if len(author_df) > 5:
            model_ols_author = sm.OLS.from_formula("Score ~ Author_Time", data=author_df)
            result_ols_author = model_ols_author.fit()
            print(result_ols_author.summary())
        else:
            print(f"Not enough data to run a model for {author}.")
            
def generate_visualizations(df):
    """
    Creates and displays all plots from the analysis.
    
    Args:
        df (pd.DataFrame): The main DataFrame.
    """
    print("\n--- Generating Visualizations ---")
    
    # Subsets for plotting
    timed_scores = df[df['Timed'] == 1]['Score']
    untimed_scores = df[df['Timed'] == 0]['Score']

    # Histogram
    sns.histplot(x=timed_scores, binwidth=1, kde=False, color="blue", label="Timed")
    sns.histplot(x=untimed_scores, binwidth=1, kde=False, color="orange", label="Untimed").set_title("Histograms of Test Scores")
    plt.legend()
    plt.show()

    # QQ Plots with a red standardized line
    print("Displaying QQ Plot for Timed Scores...")
    sm.qqplot(timed_scores, line='s')
    plt.title("QQ Plot - Timed Scores")
    plt.show()
    
    print("Displaying QQ Plot for Untimed Scores...")
    sm.qqplot(untimed_scores, line='s')
    plt.title("QQ Plot - Untimed Scores")
    plt.show()

    # Box Plots
    sns.boxplot(data=df, x="Score", y="Author", color=".8", linecolor="#137", linewidth=.75).set_title("Scores by Author")
    plt.show()
    sns.boxplot(data=df, x="Score", y="ColdHot", color=".8", linecolor="#137", linewidth=.75).set_title("Scores by Cold/Hot Start")
    plt.show()
    sns.boxplot(data=df, x="Score", y="TimedUntimed", color=".8", linecolor="#137", linewidth=.75).set_title("Scores by Timed/Untimed")
    plt.show()
    sns.boxplot(data=df, x="Score", y="TestType", color=".8", linecolor="#137", linewidth=.75).set_title("Scores by Test Type")
    plt.show()

    # Violin Plot
    plt.figure(figsize=(12, 4))
    sns.violinplot(x="TestType", y="Score", data=df, hue="TestType", legend=False).set_title("Score Distribution by Test Type")
    plt.show()

    # KDE Plot with stats
    sns.kdeplot(df["Score"], bw_adjust=2).set_title("Overall Score Density Distribution")
    plt.axvline(df["Score"].mean(), label="Mean")
    plt.axvline(df["Score"].median(), color="black", label="Median")
    plt.axvline(df["Score"].mode().squeeze(), color="green", label="Mode")
    plt.legend()
    plt.show()

    # Test Type Pie Chart
    test_type_counts = df['TestType'].value_counts()
    plt.pie(test_type_counts, labels=test_type_counts.index, autopct='%.0f%%')
    plt.title("Proportion of Test Types")
    plt.show()

    # Author Pie Chart - Made larger and exploded for readability
    plt.figure(figsize=(12, 12)) 
    author_counts = df['Author'].value_counts()
    # Explode all slices with a larger value
    explode_values = [0.1] * len(author_counts) 
    # Rotated 90 degrees clockwise by changing startangle from 140 to 50
    plt.pie(author_counts, labels=author_counts.index, autopct='%.1f%%', explode=explode_values, startangle=50)
    plt.title("Proportion of Tests by Author")
    plt.show()


def main():
    """
    Main function to run the entire data analysis pipeline.
    """
    # 1. Load and clean the data
    df = load_and_prepare_data("scores.csv")
    
    # 2. Engineer time-based features for modeling
    df = engineer_date_features(df)
    
    # 3. Run and print statistical tests and summaries
    perform_statistical_tests(df)
    
    # 4. Run and print statistical models
    run_statistical_models(df)
    
    # 5. Generate and display all plots
    generate_visualizations(df)
    
    print("\n--- Analysis Complete ---")


# This standard block ensures that the main() function is called when the script is executed
if __name__ == "__main__":
    main()