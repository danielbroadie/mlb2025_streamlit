from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import SALibrary.SimpleRatingSystem as srs
import scipy
from scipy.stats import norm
import time
from operator import attrgetter
import matplotlib.pyplot as plt

def db_simulate_mlb(schedule, standings):
    '''function to get standings and schedule_temp based off of SRS
    Args:
        schedule: dataframe
            schedule_temp of games with these columns: Week, playoff_game, away team, home team, and home margin
        standings: dataframe
            current standings
    Return:
        a simulation of whether or not a given team made the playoffs (division winners and wild card winners)
    '''
    schedule_temp = schedule.copy()
    standings_temp = standings.copy()
    num_conferences = 2
    # number of games remaining
    games_remaining = len(schedule_temp)
    num_teams = len(standings_temp)
    # unique teams
    unique_teams = np.unique(list(schedule_temp['home team']) + list(schedule_temp['away team']))
    num_mlb_games = len(schedule) / len(unique_teams)
    # dataframe to store results
    results = pd.DataFrame(index=unique_teams, columns=['winDivisionPercentage',
                                                        'winWildCardPercentage',
                                                        'makePlayoffsPercentage'],
                           data=np.zeros((len(unique_teams), 3)))
    # generating a random value for each game
    rand_vals = np.random.random(games_remaining)
    # determining home / away wins
    # print(schedule_temp)
    not_played_yet = schedule_temp['away score'].isnull()
    schedule_temp.loc[not_played_yet, 'home_win'] = (schedule_temp['prob_home'] > rand_vals).loc[not_played_yet].astype(
        int)
    schedule_temp.loc[not_played_yet, 'away_win'] = 1 - schedule_temp['home_win'].loc[not_played_yet]
    # Adding wins / losses to standings
    standings_temp['W'] = schedule_temp.groupby('away team')['away_win'].apply(lambda x: sum(x.astype(int)))
    standings_temp['L'] = schedule_temp.groupby('away team')['home_win'].apply(lambda x: sum(x.astype(int)))
    standings_temp['W'] = standings_temp['W'] + schedule_temp.groupby('home team')['home_win'].apply(
        lambda x: sum(x.astype(int)))
    standings_temp['L'] = standings_temp['L'] + schedule_temp.groupby('home team')['away_win'].apply(
        lambda x: sum(x.astype(int)))
    # print(sum(standings_temp['W']))
    # calculating win percentage based off wins and ties
    standings_temp['win_percent'] = standings_temp['W'] / num_mlb_games * 100
    # adding epsilon to break ties randomly
    standings_temp['win_percent'] = standings_temp['win_percent'] + np.random.random(num_teams) * 0.001
    # calculating division rank
    standings_temp['divisionRank'] = standings_temp.groupby(['League', 'Conference'])[['win_percent']].rank(
        ascending=False)
    # indicator measuring who won division
    standings_temp['wonDivision'] = standings_temp['divisionRank'] == 1

    # Calculate wild card standings for non-division winners
    non_division_winners = standings_temp[standings_temp['wonDivision'] == False].copy()

    # Rank non-division winners by win percentage within each league
    non_division_winners['wildCardRank'] = non_division_winners.groupby('League')[['win_percent']].rank(ascending=False)

    # Top 3 non-division winners in each league get wild card spots
    non_division_winners['wonWildCard'] = non_division_winners['wildCardRank'] <= 3

    # Add wild card column to main standings (division winners get False for wild card)
    standings_temp['wonWildCard'] = False
    standings_temp.loc[non_division_winners.index, 'wonWildCard'] = non_division_winners['wonWildCard']

    # Calculate overall playoff qualification
    standings_temp['madePlayoffs'] = standings_temp['wonDivision'] | standings_temp['wonWildCard']

    # Return dictionary with all playoff results
    return {
        'wonDivision': standings_temp['wonDivision'].astype(int),
        'wonWildCard': standings_temp['wonWildCard'].astype(int),
        'madePlayoffs': standings_temp['madePlayoffs'].astype(int)
    }


def db_MonteCarlo(func, schedule, standings, trials=1000):
    '''function to get standings and schedule_temp based off of SRS
    Args:
        func: function
            simulation function (returns dict with 'wonDivision', 'wonWildCard', 'madePlayoffs')
        schedule: dataframe
            schedule_temp of games with these columns: Week, playoff_game, away team, home team, and home margin
        standings: dataframe
            current standings
        trials:
            number of trials
    Return:
        a dictionary of simulation summaries for division winners, wild card winners, playoff qualifiers, and W/L averages
    '''
    startT = time.time()

    # Run first simulation to get structure
    tmp = func(schedule, standings)

    # Initialize output DataFrames for each category
    divisionOutput = pd.DataFrame(index=tmp['wonDivision'].index)
    wildCardOutput = pd.DataFrame(index=tmp['wonWildCard'].index)
    playoffsOutput = pd.DataFrame(index=tmp['madePlayoffs'].index)

    # NEW: Initialize DataFrames for tracking wins and losses
    winsOutput = pd.DataFrame(index=tmp['wonDivision'].index)  # Use same index as other results
    lossesOutput = pd.DataFrame(index=tmp['wonDivision'].index)

    # Add first trial results
    tmp['wonDivision'].name = '0'
    tmp['wonWildCard'].name = '0'
    tmp['madePlayoffs'].name = '0'

    divisionOutput = divisionOutput.join(tmp['wonDivision'])
    wildCardOutput = wildCardOutput.join(tmp['wonWildCard'])
    playoffsOutput = playoffsOutput.join(tmp['madePlayoffs'])

    # NEW: Get wins and losses from first simulation
    # You'll need to modify db_simulate_mlb to also return final standings with W/L
    # For now, we'll run the simulation again to get the final standings
    sample_standings = db_simulate_mlb_with_standings(schedule, standings)
    wins_series = sample_standings['W'].copy()
    losses_series = sample_standings['L'].copy()
    wins_series.name = '0'
    losses_series.name = '0'
    winsOutput = winsOutput.join(wins_series)
    lossesOutput = lossesOutput.join(losses_series)

    # Run remaining trials
    for eachTrial in range(trials - 1):
        tmp = func(schedule, standings)

        # Name each trial
        tmp['wonDivision'].name = str(eachTrial + 1)
        tmp['wonWildCard'].name = str(eachTrial + 1)
        tmp['madePlayoffs'].name = str(eachTrial + 1)

        # Join results
        divisionOutput = divisionOutput.join(tmp['wonDivision'])
        wildCardOutput = wildCardOutput.join(tmp['wonWildCard'])
        playoffsOutput = playoffsOutput.join(tmp['madePlayoffs'])

        # NEW: Get wins and losses for this trial
        trial_standings = db_simulate_mlb_with_standings(schedule, standings)
        wins_series = trial_standings['W'].copy()
        losses_series = trial_standings['L'].copy()
        wins_series.name = str(eachTrial + 1)
        losses_series.name = str(eachTrial + 1)
        winsOutput = winsOutput.join(wins_series)
        lossesOutput = lossesOutput.join(losses_series)

        if (eachTrial + 1) % 50 == 0:
            print('Trials:', eachTrial + 1)

    # Convert to float
    divisionOutput = divisionOutput.astype(float)
    wildCardOutput = wildCardOutput.astype(float)
    playoffsOutput = playoffsOutput.astype(float)
    winsOutput = winsOutput.astype(float)
    lossesOutput = lossesOutput.astype(float)

    # Function to create summary statistics
    def create_summary(output_df):
        summary = pd.DataFrame(index=output_df.index)
        summary['average'] = output_df.mean(axis=1)
        summary['standard deviation'] = output_df.std(axis=1)
        summary['standard error'] = output_df.sem(axis=1)
        summary['minimum'] = output_df.min(axis=1)
        summary['maximum'] = output_df.max(axis=1)
        summary['1% percentile'] = output_df.quantile(q=0.01, axis=1)
        summary['5%'] = output_df.quantile(q=0.05, axis=1)
        summary['10%'] = output_df.quantile(q=0.1, axis=1)
        summary['50%'] = output_df.quantile(q=0.5, axis=1)
        summary['90%'] = output_df.quantile(q=0.9, axis=1)
        summary['95%'] = output_df.quantile(q=0.95, axis=1)
        summary['99%'] = output_df.quantile(q=0.99, axis=1)
        return summary

    # Create summaries for each category
    divisionSummary = create_summary(divisionOutput)
    wildCardSummary = create_summary(wildCardOutput)
    playoffsSummary = create_summary(playoffsOutput)
    winsSummary = create_summary(winsOutput)
    lossesSummary = create_summary(lossesOutput)

    print('CPU seconds ', time.time() - startT)

    return {
        'divisionWinners': divisionSummary,
        'wildCardWinners': wildCardSummary,
        'playoffQualifiers': playoffsSummary,
        'avgWins': winsSummary,
        'avgLosses': lossesSummary
    }


def db_simulate_mlb_with_standings(schedule, standings):
    '''Modified version of db_simulate_mlb that returns the final standings with W/L records
    Args:
        schedule: dataframe
        standings: dataframe
    Return:
        final standings with wins and losses
    '''
    schedule_temp = schedule.copy()
    standings_temp = standings.copy()

    # Get unique teams and number of games
    unique_teams = np.unique(list(schedule_temp['home team']) + list(schedule_temp['away team']))
    num_mlb_games = len(schedule) / len(unique_teams)
    games_remaining = len(schedule_temp)

    # Generate random values for each game
    rand_vals = np.random.random(games_remaining)

    # Determine home / away wins for games not yet played
    not_played_yet = schedule_temp['away score'].isnull()
    schedule_temp.loc[not_played_yet, 'home_win'] = (schedule_temp['prob_home'] > rand_vals).loc[not_played_yet].astype(
        int)
    schedule_temp.loc[not_played_yet, 'away_win'] = 1 - schedule_temp['home_win'].loc[not_played_yet]

    # Initialize wins and losses for all teams
    standings_temp = standings_temp.copy()

    # Calculate wins and losses from the schedule
    away_wins = schedule_temp.groupby('away team')['away_win'].sum()
    away_losses = schedule_temp.groupby('away team')['home_win'].sum()
    home_wins = schedule_temp.groupby('home team')['home_win'].sum()
    home_losses = schedule_temp.groupby('home team')['away_win'].sum()

    # Make sure we have all teams in the results
    all_teams = standings_temp.index

    # Initialize W and L columns with zeros
    standings_temp['W'] = 0
    standings_temp['L'] = 0

    # Add away wins and losses
    for team in all_teams:
        if team in away_wins.index:
            standings_temp.loc[team, 'W'] += away_wins.loc[team]
        if team in away_losses.index:
            standings_temp.loc[team, 'L'] += away_losses.loc[team]

    # Add home wins and losses
    for team in all_teams:
        if team in home_wins.index:
            standings_temp.loc[team, 'W'] += home_wins.loc[team]
        if team in home_losses.index:
            standings_temp.loc[team, 'L'] += home_losses.loc[team]

    return standings_temp