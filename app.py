import streamlit as st
import SALibrary.scrape as sc
import SALibrary.simulate as sim
import SALibrary.SimpleRatingSystem as srs
import wildcard
import numpy as np
import pandas as pd
import time
from datetime import datetime, date

st.set_page_config(page_title="MLB Season Simulation", layout="wide")
st.title("MLB Season Simulation ⚾")

# Initialize session state
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = None
if 'schedule_probs' not in st.session_state:
    st.session_state['schedule_probs'] = None
if 'standings' not in st.session_state:
    st.session_state['standings'] = None
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None
if 'rmse' not in st.session_state:
    st.session_state['rmse'] = None
if 'home_advantage' not in st.session_state:
    st.session_state['home_advantage'] = None

# Fetch Ratings Button
if st.button("Fetch schedule and compute ratings"):
    st.write("Fetching MLB schedule and computing ratings...")

    # Fetch schedule and standings
    schedule, standings = sc.scrape_mlb(2025)

    # Prepare schedule
    schedule = srs.months_test_set_MLB(schedule, standings)
    schedule_so_far = schedule.dropna()

    # Filter future games to only include today and later
    today = date.today()

    # Convert date column to datetime if it's not already
    if 'Date' in schedule.columns:
        schedule['Date'] = pd.to_datetime(schedule['Date']).dt.date
        # Filter to keep only games from today onwards for future predictions
        future_games = schedule[schedule['Date'] >= today]
        # Keep all past games for training
        past_games = schedule[schedule['Date'] < today].dropna()
        # Combine: past games (completed) + future games (from today onwards)
        schedule = pd.concat([past_games, future_games], ignore_index=True)
    elif 'date' in schedule.columns:
        schedule['date'] = pd.to_datetime(schedule['date']).dt.date
        # Filter to keep only games from today onwards for future predictions
        future_games = schedule[schedule['date'] >= today]
        # Keep all past games for training
        past_games = schedule[schedule['date'] < today].dropna()
        # Combine: past games (completed) + future games (from today onwards)
        schedule = pd.concat([past_games, future_games], ignore_index=True)

    # Use completed games for training
    schedule_so_far = schedule.dropna()

    # Compute ratings
    ratings, home_advantage, rmse = srs.srs_teamtrain(
        schedule_so_far['home team'],
        schedule_so_far['away team'],
        schedule_so_far['home score'],
        schedule_so_far['away score']
    )

    # Compute probabilities for all games (including filtered future games)
    schedule_probs = srs.SRS_to_probabilities(
        ratings, rmse, schedule, home_advantage
    )

    # Store in session state
    st.session_state['ratings'] = ratings.sort_values("Rating", ascending=False)
    st.session_state['schedule_probs'] = schedule_probs
    st.session_state['standings'] = standings
    st.session_state['rmse'] = rmse
    st.session_state['home_advantage'] = home_advantage
    st.session_state['sim_results'] = None  # clear old simulation

# Simulation controls
trials = st.slider(
    "Number of simulation trials",
    min_value=10, max_value=1000, value=100, step=10
)

# Run Simulation Button
if st.button("Run Simulation"):
    if st.session_state['schedule_probs'] is None or st.session_state['standings'] is None:
        st.warning("Please fetch ratings first!")
    else:
        status_text = st.empty()
        status_text.write(f"Running simulations with {trials} trials...")
        sim_results = wildcard.db_MonteCarlo(
            wildcard.db_simulate_mlb,
            st.session_state['schedule_probs'],
            st.session_state['standings'],
            trials=trials
        )
        st.session_state['sim_results'] = sim_results
        status_text.write("Simulations complete ✅")

# Create tabs for Team Ratings and Simulation Results
if st.session_state['ratings'] is not None or st.session_state['sim_results'] is not None:
    tab1, tab2, tab3 = st.tabs(["Team Ratings", "Simulation Results", "Projected Standings"])

    # Team Ratings Tab
    with tab1:
        if st.session_state['ratings'] is not None:
            st.subheader("Team Ratings with Current Records")

            # Display RMSE and Home Advantage if available
            if st.session_state['rmse'] is not None and st.session_state['home_advantage'] is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{st.session_state['rmse']:.3f}")
                with col2:
                    st.metric("Home Advantage", f"{st.session_state['home_advantage']:.3f}")

            # Merge ratings with standings to get wins and losses
            if st.session_state['standings'] is not None:
                # Create a copy of ratings for display
                ratings_display = st.session_state['ratings'].copy()

                # Round Rating to 2 decimal places with fixed formatting
                if 'Rating' in ratings_display.columns:
                    ratings_display['Rating'] = pd.to_numeric(ratings_display['Rating'], errors='coerce').round(
                        2).apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

                # Merge with standings to get W/L data
                # Assuming standings has columns like 'Team', 'W', 'L' or similar
                # You may need to adjust column names based on your actual standings structure
                try:
                    # Reset index to make team names a column if they're currently the index
                    if ratings_display.index.name or not isinstance(ratings_display.index, pd.RangeIndex):
                        ratings_display = ratings_display.reset_index()
                        team_col = ratings_display.columns[0]  # First column should be team names

                    # Try to merge with standings - adjust column names as needed
                    standings_cols = st.session_state['standings'].columns.tolist()

                    # Common column name patterns for wins/losses
                    win_col = None
                    loss_col = None
                    team_col_standings = None

                    for col in standings_cols:
                        col_lower = col.lower()
                        if 'win' in col_lower or col_lower == 'w':
                            win_col = col
                        elif 'loss' in col_lower or col_lower == 'l':
                            loss_col = col
                        elif 'team' in col_lower:
                            team_col_standings = col

                    # If we found the necessary columns, merge the data
                    if win_col and loss_col:
                        if team_col_standings:
                            merge_key_standings = team_col_standings
                        else:
                            # Assume first column or index contains team names
                            standings_for_merge = st.session_state['standings'].reset_index()
                            merge_key_standings = standings_for_merge.columns[0]
                            standings_for_merge = standings_for_merge

                        # Perform the merge
                        ratings_with_record = pd.merge(
                            ratings_display,
                            st.session_state['standings'][
                                [merge_key_standings, win_col, loss_col]] if team_col_standings else
                            standings_for_merge[[merge_key_standings, win_col, loss_col]],
                            left_on=team_col if 'team_col' in locals() else ratings_display.columns[0],
                            right_on=merge_key_standings,
                            how='left'
                        )

                        # Remove "Tm" column if it exists
                        if 'Tm' in ratings_with_record.columns:
                            ratings_with_record = ratings_with_record.drop('Tm', axis=1)

                        # Reorder columns to show W/L after Rating
                        cols = ratings_with_record.columns.tolist()
                        if win_col in cols and loss_col in cols:
                            # Find Rating column position
                            rating_idx = next((i for i, col in enumerate(cols) if 'rating' in col.lower()), 1)

                            # Reorder: Team, Rating, W, L, other columns
                            new_order = []
                            new_order.append(cols[0])  # Team name
                            if rating_idx < len(cols):
                                new_order.append(cols[rating_idx])  # Rating
                            new_order.extend([win_col, loss_col])  # W, L

                            # Add remaining columns
                            for col in cols:
                                if col not in new_order:
                                    new_order.append(col)

                            ratings_with_record = ratings_with_record[new_order]

                        # Calculate win percentage with fixed 3 decimal formatting
                        if win_col in ratings_with_record.columns and loss_col in ratings_with_record.columns:
                            win_pct = (
                                    ratings_with_record[win_col] /
                                    (ratings_with_record[win_col] + ratings_with_record[loss_col])
                            )
                            ratings_with_record['Win%'] = win_pct.apply(lambda x: f"{x:.3f}" if pd.notna(x) else x)

                        # Add rank column and reset index
                        ratings_with_record = ratings_with_record.reset_index(drop=True)
                        ratings_with_record.insert(0, 'Rank', range(1, len(ratings_with_record) + 1))

                        # Configure column widths for numerical columns
                        column_config = {
                            'Rank': st.column_config.NumberColumn(width='small'),
                            'Rating': st.column_config.TextColumn(width='small'),
                            win_col: st.column_config.NumberColumn(width='small'),
                            loss_col: st.column_config.NumberColumn(width='small'),
                            'Win%': st.column_config.TextColumn(width='small')
                        }

                        st.dataframe(ratings_with_record, use_container_width=True, hide_index=True,
                                     column_config=column_config)

                    else:
                        st.info("Could not find wins/losses columns in standings data. Showing ratings only.")
                        # Still apply formatting to ratings-only display
                        ratings_display = ratings_display.reset_index(drop=True)
                        if 'Rating' in ratings_display.columns:
                            ratings_display['Rating'] = pd.to_numeric(ratings_display['Rating'], errors='coerce').round(
                                2).apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                        ratings_display.insert(0, 'Rank', range(1, len(ratings_display) + 1))

                        # Configure column widths for numerical columns
                        column_config = {
                            'Rank': st.column_config.NumberColumn(width='small'),
                            'Rating': st.column_config.TextColumn(width='small')
                        }

                        st.dataframe(ratings_display, use_container_width=True, hide_index=True,
                                     column_config=column_config)

                        # Show available columns for debugging
                        with st.expander("Debug: Available standings columns"):
                            st.write("Standings columns:", standings_cols)

                except Exception as e:
                    st.warning(f"Could not merge ratings with standings: {str(e)}")
                    # Still apply formatting to fallback display
                    ratings_display = ratings_display.reset_index(drop=True)
                    if 'Rating' in ratings_display.columns:
                        ratings_display['Rating'] = pd.to_numeric(ratings_display['Rating'], errors='coerce').round(
                            2).apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    ratings_display.insert(0, 'Rank', range(1, len(ratings_display) + 1))

                    # Configure column widths for numerical columns
                    column_config = {
                        'Rank': st.column_config.NumberColumn(width='small'),
                        'Rating': st.column_config.TextColumn(width='small')
                    }

                    st.dataframe(ratings_display, use_container_width=True, hide_index=True,
                                 column_config=column_config)

                    # Show debug info
                    with st.expander("Debug: Data structure info"):
                        st.write("Ratings columns:", st.session_state['ratings'].columns.tolist())
                        st.write("Ratings index:", st.session_state['ratings'].index.name)
                        if st.session_state['standings'] is not None:
                            st.write("Standings columns:", st.session_state['standings'].columns.tolist())
            else:
                # Apply formatting even when no standings data
                ratings_display = st.session_state['ratings'].copy()
                if 'Rating' in ratings_display.columns:
                    ratings_display['Rating'] = pd.to_numeric(ratings_display['Rating'], errors='coerce').round(
                        2).apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                ratings_display = ratings_display.reset_index(drop=True)
                ratings_display.insert(0, 'Rank', range(1, len(ratings_display) + 1))

                # Configure column widths for numerical columns
                column_config = {
                    'Rank': st.column_config.NumberColumn(width='small'),
                    'Rating': st.column_config.TextColumn(width='small')
                }

                st.dataframe(ratings_display, use_container_width=True, hide_index=True, column_config=column_config)
        else:
            st.info("Click 'Fetch Ratings' to load team ratings data.")

    # Simulation Results Tab
    with tab2:
        if st.session_state['sim_results'] is not None:
            st.subheader("Monte Carlo Simulation Results")


            def create_display_table(results_data, percentage_label):
                """Helper function to create formatted display table"""
                if results_data is not None and 'average' in results_data.columns:
                    # Reset index to make team names a column
                    display_data = results_data.reset_index()
                    team_col = display_data.columns[0]  # First column should be team names

                    # Select only team and average columns
                    display_data = display_data[[team_col, 'average']].copy()

                    # Rename columns
                    display_data = display_data.rename(columns={
                        team_col: 'Tm',
                        'average': percentage_label
                    })

                    # Sort by percentage descending
                    display_data = display_data.sort_values(percentage_label, ascending=False)

                    # Round to 3 decimals (no percentage conversion)
                    display_data[percentage_label] = display_data[percentage_label].round(3)

                    # Add rank column
                    display_data.insert(0, 'Rank', range(1, len(display_data) + 1))

                    return display_data
                return None


            # Get the simulation results
            division_results = st.session_state['sim_results'].get('divisionWinners')
            wildcard_results = st.session_state['sim_results'].get('wildCardWinners')
            playoff_results = st.session_state['sim_results'].get('playoffQualifiers')

            # Create individual tables
            division_display = create_display_table(division_results, 'Division')
            wildcard_display = create_display_table(wildcard_results, 'Wildcard')
            playoff_display = create_display_table(playoff_results, 'Playoff')

            # Create combined table if all data is available
            if all(data is not None for data in [division_display, wildcard_display, playoff_display]):
                # Create combined dataframe
                combined_data = pd.DataFrame()
                combined_data['Rank'] = division_display['Rank']
                combined_data['Team'] = division_display['Tm']
                combined_data['Division %'] = division_display['Division']
                combined_data['Wildcard %'] = wildcard_display.set_index('Tm').loc[division_display['Tm']][
                    'Wildcard'].values
                combined_data['Playoff %'] = playoff_display.set_index('Tm').loc[division_display['Tm']]['Playoff'].values

                # Sort by playoff probability
                combined_data = combined_data.sort_values('Playoff %', ascending=False)
                combined_data['Rank'] = range(1, len(combined_data) + 1)

                # Configure column widths
                column_config = {
                    'Rank': st.column_config.NumberColumn(width='small'),
                    'Team': st.column_config.TextColumn(width='medium'),
                    'Division %': st.column_config.NumberColumn(width='small', format="%.3f"),
                    'Wildcard %': st.column_config.NumberColumn(width='small', format="%.3f"),
                    'Playoff %': st.column_config.NumberColumn(width='small', format="%.3f")
                }

                st.dataframe(combined_data, use_container_width=True, hide_index=True,
                             column_config=column_config)
            else:
                st.warning("Simulation data not available - please run simulation.")
                # Debug information
                with st.expander("Debug: Available data"):
                    st.write("sim_results keys:", list(st.session_state['sim_results'].keys()) if st.session_state[
                        'sim_results'] else "None")
                    st.write("Division display:", division_display is not None)
                    st.write("Wildcard display:", wildcard_display is not None)
                    st.write("Playoff display:", playoff_display is not None)
        else:
            st.info("Click 'Run Simulation' to generate simulation results.")

    # Projected Standings Tab
    with tab3:
        if st.session_state['ratings'] is not None:
            st.subheader("Projected Season-End Standings")

            if st.session_state['schedule_probs'] is not None and st.session_state['standings'] is not None:
                # Get current standings
                current_standings = st.session_state['standings'].copy()
                schedule_probs = st.session_state['schedule_probs'].copy()

                # Find win/loss columns in current standings
                standings_cols = current_standings.columns.tolist()
                win_col = None
                loss_col = None
                team_col_standings = None

                for col in standings_cols:
                    col_lower = col.lower()
                    if 'win' in col_lower or col_lower == 'w':
                        win_col = col
                    elif 'loss' in col_lower or col_lower == 'l':
                        loss_col = col
                    elif 'team' in col_lower:
                        team_col_standings = col

                if win_col and loss_col:
                    # Get teams from standings
                    if team_col_standings:
                        teams = current_standings[team_col_standings].tolist()
                    else:
                        teams = current_standings.index.tolist()

                    # Calculate expected wins from remaining games
                    remaining_games = schedule_probs[schedule_probs['away score'].isnull()].copy()

                    projected_wins = {}
                    projected_losses = {}

                    for team in teams:
                        # Current wins and losses
                        if team_col_standings:
                            current_w = current_standings[current_standings[team_col_standings] == team][win_col].iloc[
                                0]
                            current_l = current_standings[current_standings[team_col_standings] == team][loss_col].iloc[
                                0]
                        else:
                            current_w = current_standings.loc[team, win_col]
                            current_l = current_standings.loc[team, loss_col]

                        # Expected wins from remaining games
                        home_games = remaining_games[remaining_games['home team'] == team]
                        away_games = remaining_games[remaining_games['away team'] == team]

                        expected_home_wins = home_games['prob_home'].sum() if len(home_games) > 0 else 0
                        expected_away_wins = (1 - away_games['prob_home']).sum() if len(away_games) > 0 else 0

                        total_remaining_games = len(home_games) + len(away_games)
                        expected_remaining_wins = expected_home_wins + expected_away_wins
                        expected_remaining_losses = total_remaining_games - expected_remaining_wins

                        # Projected season totals
                        projected_wins[team] = current_w + expected_remaining_wins
                        projected_losses[team] = current_l + expected_remaining_losses

                    # Create projected standings dataframe
                    projected_display = pd.DataFrame()
                    projected_display['Team'] = teams
                    projected_display['Proj W'] = [projected_wins[team] for team in teams]
                    projected_display['Proj L'] = [projected_losses[team] for team in teams]

                    # Calculate projected win percentage
                    projected_display['Proj Win%'] = (
                            projected_display['Proj W'] /
                            (projected_display['Proj W'] + projected_display['Proj L'])
                    ).round(3)

                    # Get current ratings for display
                    ratings_display = st.session_state['ratings'].copy()

                    # Reset index to make team names a column if they're currently the index
                    if ratings_display.index.name or not isinstance(ratings_display.index, pd.RangeIndex):
                        ratings_for_merge = ratings_display.reset_index()
                        team_col = ratings_for_merge.columns[0]
                    else:
                        ratings_for_merge = ratings_display.copy()
                        team_col = 'Team'

                    # Round Rating to 2 decimal places
                    if 'Rating' in ratings_for_merge.columns:
                        ratings_for_merge['Rating'] = pd.to_numeric(ratings_for_merge['Rating'], errors='coerce').round(
                            2)

                    # Merge ratings with projected standings
                    projected_display = pd.merge(
                        projected_display,
                        ratings_for_merge[[team_col, 'Rating']],
                        left_on='Team',
                        right_on=team_col,
                        how='left'
                    )

                    # Remove duplicate team column if exists
                    if team_col != 'Team' and team_col in projected_display.columns:
                        projected_display = projected_display.drop(team_col, axis=1)

                    # Sort by projected win percentage (highest first)
                    projected_display = projected_display.sort_values('Proj Win%', ascending=False).reset_index(
                        drop=True)

                    # Add rank column
                    projected_display.insert(0, 'Rank', range(1, len(projected_display) + 1))

                    # Reorder columns
                    column_order = ['Rank', 'Team', 'Rating', 'Proj W', 'Proj L', 'Proj Win%']
                    projected_display = projected_display[column_order]

                    # Configure column widths
                    column_config = {
                        'Rank': st.column_config.NumberColumn(width='small'),
                        'Team': st.column_config.TextColumn(width='medium'),
                        'Rating': st.column_config.NumberColumn(width='small', format="%.2f"),
                        'Proj W': st.column_config.NumberColumn(width='small', format="%.1f"),
                        'Proj L': st.column_config.NumberColumn(width='small', format="%.1f"),
                        'Proj Win%': st.column_config.NumberColumn(width='small', format="%.3f")
                    }

                    st.dataframe(projected_display, use_container_width=True, hide_index=True,
                                 column_config=column_config)

                    st.success(
                        "Showing projected season-end records based on current standings + expected wins from remaining games!")

                else:
                    st.warning("Could not find wins/losses columns in standings data.")

            else:
                st.warning("Schedule probabilities and standings data are required.")
        else:
            st.info("Team ratings are needed for projected standings.")