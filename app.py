import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

st.title("2025 MLB Game Data & Standings with Playoff Probabilities")

uploaded_file = st.file_uploader("Upload a CSV file with 2025 MLB games", type=["csv"])

if uploaded_file is not None:
    # --- Load CSV ---
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df.columns = df.columns.str.strip()
    df['Home Score'] = pd.to_numeric(df['Home Score'], errors='coerce')
    df['Away Score'] = pd.to_numeric(df['Away Score'], errors='coerce')

    # --- Played games ---
    df_played = df.dropna(subset=['Home Score', 'Away Score']).copy()
    df_played['Run Differential'] = df_played['Home Score'] - df_played['Away Score']

    # --- Vectorized team ratings ---
    teams = pd.concat([df['Home'], df['Away']]).unique()
    n_teams = len(teams)
    team_index = {team: i for i, team in enumerate(teams)}

    X = np.zeros((len(df_played), n_teams + 1))
    y = df_played['Run Differential'].to_numpy()
    X[np.arange(len(df_played)), df_played['Home'].map(team_index)] = 1
    X[np.arange(len(df_played)), df_played['Away'].map(team_index)] = -1
    X[:, -1] = 1  # home advantage
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ratings = params[:n_teams]
    home_adv = params[-1]
    team_ratings = {team: ratings[team_index[team]] for team in teams}

    # --- Predicted Run Differential & Home Win Prob ---
    df['Predicted Run Differential'] = (
            ratings[df['Home'].map(team_index).to_numpy()] -
            ratings[df['Away'].map(team_index).to_numpy()] +
            home_adv
    )
    residuals = df_played['Run Differential'] - (
            ratings[df_played['Home'].map(team_index).to_numpy()] -
            ratings[df_played['Away'].map(team_index).to_numpy()] + home_adv
    )
    sigma = np.std(residuals)
    df['Home Win Prob'] = norm.cdf(df['Predicted Run Differential'] / sigma)

    # --- Remaining games ---
    df_remaining = df[df['Home Score'].isna() | df['Away Score'].isna()].copy()

    # --- Full MLB team info ---
    team_info = {
        # NL East
        'Atlanta Braves': {'League': 'NL', 'Division': 'East'},
        'Miami Marlins': {'League': 'NL', 'Division': 'East'},
        'New York Mets': {'League': 'NL', 'Division': 'East'},
        'Philadelphia Phillies': {'League': 'NL', 'Division': 'East'},
        'Washington Nationals': {'League': 'NL', 'Division': 'East'},
        # NL Central
        'Chicago Cubs': {'League': 'NL', 'Division': 'Central'},
        'Cincinnati Reds': {'League': 'NL', 'Division': 'Central'},
        'Milwaukee Brewers': {'League': 'NL', 'Division': 'Central'},
        'Pittsburgh Pirates': {'League': 'NL', 'Division': 'Central'},
        'St. Louis Cardinals': {'League': 'NL', 'Division': 'Central'},
        # NL West
        'Arizona Diamondbacks': {'League': 'NL', 'Division': 'West'},
        'Colorado Rockies': {'League': 'NL', 'Division': 'West'},
        'Los Angeles Dodgers': {'League': 'NL', 'Division': 'West'},
        'San Diego Padres': {'League': 'NL', 'Division': 'West'},
        'San Francisco Giants': {'League': 'NL', 'Division': 'West'},
        # AL East
        'Baltimore Orioles': {'League': 'AL', 'Division': 'East'},
        'Boston Red Sox': {'League': 'AL', 'Division': 'East'},
        'New York Yankees': {'League': 'AL', 'Division': 'East'},
        'Tampa Bay Rays': {'League': 'AL', 'Division': 'East'},
        'Toronto Blue Jays': {'League': 'AL', 'Division': 'East'},
        # AL Central
        'Chicago White Sox': {'League': 'AL', 'Division': 'Central'},
        'Cleveland Guardians': {'League': 'AL', 'Division': 'Central'},
        'Detroit Tigers': {'League': 'AL', 'Division': 'Central'},
        'Kansas City Royals': {'League': 'AL', 'Division': 'Central'},
        'Minnesota Twins': {'League': 'AL', 'Division': 'Central'},
        # AL West
        'Houston Astros': {'League': 'AL', 'Division': 'West'},
        'Los Angeles Angels': {'League': 'AL', 'Division': 'West'},
        'Oakland Athletics': {'League': 'AL', 'Division': 'West'},
        'Seattle Mariners': {'League': 'AL', 'Division': 'West'},
        'Texas Rangers': {'League': 'AL', 'Division': 'West'},
    }

    # --- Current standings ---
    standings = {team: {'Wins': 0, 'Losses': 0, 'Runs Scored': 0, 'Runs Allowed': 0} for team in teams}
    for _, row in df_played.iterrows():
        home, away = row['Home'], row['Away']
        hs, as_ = row['Home Score'], row['Away Score']
        standings[home]['Runs Scored'] += hs
        standings[home]['Runs Allowed'] += as_
        standings[away]['Runs Scored'] += as_
        standings[away]['Runs Allowed'] += hs
        if hs > as_:
            standings[home]['Wins'] += 1
            standings[away]['Losses'] += 1
        elif as_ > hs:
            standings[away]['Wins'] += 1
            standings[home]['Losses'] += 1

    # --- Season simulation: Division + Wild Card + Playoff ---
    n_sims = 1000
    division_results = {team: 0 for team in teams}
    wildcard_results = {team: 0 for team in teams}
    playoff_results = {team: 0 for team in teams}

    for sim in range(n_sims):
        sim_wins = {team: standings[team]['Wins'] for team in teams}
        sim_rdiff = {team: standings[team]['Runs Scored'] - standings[team]['Runs Allowed'] for team in teams}

        # Simulate remaining games
        for _, row in df_remaining.iterrows():
            home, away = row['Home'], row['Away']
            pred_rd = row['Predicted Run Differential']
            simulated_rd = np.random.normal(pred_rd, sigma)
            if simulated_rd > 0:
                sim_wins[home] += 1
                sim_rdiff[home] += simulated_rd
                sim_rdiff[away] -= simulated_rd
            else:
                sim_wins[away] += 1
                sim_rdiff[away] += -simulated_rd
                sim_rdiff[home] -= -simulated_rd

        # Determine division winners
        div_winners = []
        for league in ['AL', 'NL']:
            for division in ['East', 'Central', 'West']:
                teams_in_div = [team for team in teams
                                if team_info.get(team, {}).get('League') == league
                                and team_info.get(team, {}).get('Division') == division]
                if teams_in_div:
                    winner = max(teams_in_div, key=lambda t: (sim_wins[t], sim_rdiff[t]))
                    div_winners.append(winner)
                    division_results[winner] += 1

        # Determine wild cards (top 3 non-division winners per league)
        for league in ['AL', 'NL']:
            league_teams = [team for team in teams if
                            team_info.get(team, {}).get('League') == league and team not in div_winners]
            league_sorted = sorted(league_teams, key=lambda t: (sim_wins[t], sim_rdiff[t]), reverse=True)
            wild_cards = league_sorted[:3]
            for team in wild_cards:
                wildcard_results[team] += 1

        # Playoff = division winners + wild cards
        for team in div_winners + wild_cards:
            playoff_results[team] += 1

    # --- Prepare standings for display ---
    standings_df = pd.DataFrame.from_dict(standings, orient='index')
    standings_df['Runs Scored'] = standings_df['Runs Scored'].astype(int)
    standings_df['Runs Allowed'] = standings_df['Runs Allowed'].astype(int)
    standings_df['Run Differential'] = (standings_df['Runs Scored'] - standings_df['Runs Allowed']).astype(int)
    standings_df['Team Rating'] = [round(team_ratings[t], 2) for t in standings_df.index]

    standings_df['Division Win %'] = [division_results[t] / n_sims * 100 for t in standings_df.index]
    standings_df['Wild Card %'] = [wildcard_results[t] / n_sims * 100 for t in standings_df.index]
    standings_df['Playoff %'] = [playoff_results[t] / n_sims * 100 for t in standings_df.index]

    standings_df = standings_df.sort_values(by=['Wins', 'Run Differential'], ascending=False)

    # --- Streamlit tabs ---
    tab1, tab2, tab3 = st.tabs(["Game Data", "Standings", "Upcoming Games"])

    with tab1:
        st.subheader("Game Data")
        st.dataframe(
            df[['Date', 'Home', 'Away', 'Home Score', 'Away Score', 'Predicted Run Differential', 'Home Win Prob']])

    with tab2:
        st.subheader("Current Standings with Team Ratings & Playoff Probabilities")
        st.dataframe(
            standings_df.style.format({
                'Division Win %': '{:.1f}%',
                'Wild Card %': '{:.1f}%',
                'Playoff %': '{:.1f}%'
            })
        )

    with tab3:
        st.subheader("Upcoming Games")
        if not df_remaining.empty:
            st.dataframe(df_remaining[['Date', 'Home', 'Away', 'Predicted Run Differential', 'Home Win Prob']])
        else:
            st.info("No upcoming games remaining.")
