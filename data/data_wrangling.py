import pandas as pd


def describe_df(df):
    """
    Display various information about the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to describe.

    Returns:
    None
    """
    display(df.head())
    display(df.info())
    display(df.describe())
    display(df.shape)
    display(df.isnull().sum())
    display(df.columns)


def analyze_portfolio_data(portfolio_df):
    """
    Analyzes the portfolio data by calculating the channel distribution and performing analysis on offer duration and difficulty.

    Args:
            portfolio_df (pandas.DataFrame): The portfolio data as a pandas DataFrame.

    Returns:
            tuple: A tuple containing two objects:
                    - channel_distribution (pandas.Series): The channel distribution of the portfolio data.
                    - duration_difficulty_analysis (pandas.DataFrame): The analysis of offer duration and difficulty.
    """
    # Checking channel distribution
    portfolio_df['channels'] = portfolio_df['channels'].apply(
        lambda x: ','.join(x))
    channel_distribution = portfolio_df['channels'].value_counts()

    # Analyzing offer duration and difficulty
    duration_difficulty_analysis = portfolio_df[[
        'duration', 'difficulty']].describe()

    return channel_distribution, duration_difficulty_analysis


def analyze_profile_data(profile_df):
    """
    Analyzes the profile data.

    Parameters:
    profile_df (DataFrame): The profile data to be analyzed.

    Returns:
    tuple: A tuple containing the following:
            - missing_data (Series): The count of missing data for each column in the profile data.
            - age_anomalies (DataFrame): The rows in the profile data where age is 118.
            - first_became_member (Series): The first few values of the 'became_member_on' column in the profile data.
    """
    # Checking for missing data
    missing_data = profile_df.isnull().sum()

    # Identifying age anomalies
    age_anomalies = profile_df[profile_df['age'] == 118]

    # Converting 'became_member_on' to datetime
    profile_df['became_member_on'] = pd.to_datetime(
        profile_df['became_member_on'].astype(str), format='%Y%m%d')

    return missing_data, age_anomalies, profile_df['became_member_on'].head()


def analyze_transcript_data(transcript_df):
    """
    Analyzes the transcript data.

    Args:
            transcript_df (DataFrame): The transcript data as a pandas DataFrame.

    Returns:
            tuple: A tuple containing the unpacked 'value' field as a DataFrame and the time distribution as a Series.
    """
    # Unpacking nested data in 'value' field
    value_unpacked = transcript_df['value'].apply(pd.Series)

    # Analyzing time distribution
    time_distribution = transcript_df['time'].describe()

    return value_unpacked.head(), time_distribution
