from datetime import datetime
import pandas as pd
import numpy as np
import json
import os


def clean_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the portfolio DataFrame by creating dummy columns for channels and offer types,
    combining them with the original dataset, and removing the original channels and offer_type columns.
    Also, rename the id column to offer_id.

    Parameters:
    portfolio (pd.DataFrame): The original portfolio DataFrame.

    Returns:
    pd.DataFrame: The cleaned portfolio DataFrame.
    """
    # create dummy column for each channel
    channels = portfolio['channels'].apply(lambda x: pd.Series(
        [1] * len(x), index=x)).fillna(0, downcast='infer')
    channels.columns = ['channel_' + str(col) for col in channels.columns]

    # create dummy column for each offer type
    offer_type = pd.get_dummies(portfolio['offer_type'], prefix="offer_type")

    # combine the dummy channels and dummy offer types with the original dataset and remove the original channels & offer_type columns
    with pd.option_context('mode.chained_assignment', None):
        portfolio_clean = pd.concat(
            [portfolio, channels, offer_type], axis=1, sort=False)
        portfolio_clean.drop(columns=['channels', 'offer_type'], inplace=True)

    # rename the id column to offer_id
    portfolio_clean.rename(columns={'id': 'offer_id'}, inplace=True)
    return portfolio_clean


def clean_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the profile DataFrame by performing the following steps:
    1. Convert 'became_member_on' column to datetime format.
    2. Filter out rows where age is 118.
    3. Rename 'id' column to 'customer_id'.
    4. Create dummy variables for 'gender'.
    5. Calculate member tenure in days.
    6. Drop unnecessary columns.

    Parameters:
    - profile: pd.DataFrame: The input DataFrame to be cleaned.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Create a copy of the profile DataFrame
    profile_clean = profile.copy()

    # Convert 'became_member_on' to datetime format
    try:
        profile_clean['became_member_on'] = pd.to_datetime(
            profile_clean['became_member_on'].astype(str), format='%Y%m%d')
    except Exception as e:
        print(f"Error converting 'became_member_on' to datetime: {e}")
        return profile

    # Filter out rows where age is 118
    profile_clean = profile_clean[profile_clean['age'] != 118]

    # Rename 'id' column to 'customer_id'
    profile_clean.rename(columns={'id': 'customer_id'}, inplace=True)

    # Create dummy variables for 'gender'
    profile_clean = pd.get_dummies(profile_clean, columns=[
                                   'gender'], prefix='gender')

    # Calculate member tenure in days
    today = datetime.today().date()
    profile_clean['member_tenure'] = profile_clean['became_member_on'].apply(
        lambda x: (today - x.date()).days)

    # Drop unnecessary columns
    profile_clean.drop(columns=['became_member_on'], inplace=True)

    return profile_clean


def clean_transcript(transcript):
    """
    Cleans the transcript data by performing the following steps:
    1. Replaces the space in the event column with an underscore and creates dummy columns for each event.
    2. Combines the event dummy columns with the original transcript dataset and drops the original event column.
    3. Extracts the offer ID from the value column.
    4. Extracts the transaction amount from the value column.
    5. Drops the value column after extraction.
    6. Renames the person column to customer_id.

    Parameters:
    transcript (DataFrame): The transcript data to be cleaned.

    Returns:
    DataFrame: The cleaned transcript data.
    """
    transcript.event = transcript.event.str.replace(' ', '_')
    event = pd.get_dummies(transcript.event, prefix="event")
    transcript_clean = pd.concat([transcript, event], axis=1, sort=False)
    transcript_clean.drop(columns='event', inplace=True)
    transcript_clean['offer_id'] = [[*v.values()][0]
                                    if [*v.keys()][0] in ['offer id', 'offer_id']
                                    else None
                                    for v in transcript_clean.value]
    transcript_clean['amount'] = [np.round([*v.values()][0], decimals=2)
                                  if [*v.keys()][0] == 'amount'
                                  else None
                                  for v in transcript_clean.value]
    transcript_clean.drop(columns='value', inplace=True)
    transcript_clean.rename(columns={'person': 'customer_id'}, inplace=True)
    return transcript_clean
