import streamlit as st
import SALibrary.scrape as sc
import SALibrary.simulate as sim
import SALibrary.SimpleRatingSystem as srs
import numpy as np
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title="MLB Season Simulation", layout="wide")
st.title("⚾ MLB Season Simulation")

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
        st.write(f"Running simulations with {trials} trials...")
        sim_results = sim.MonteCarlo(
            sim.simulate_mlb,
            st.session_state['schedule_probs'],
            st.session_state['standings'],
            trials=trials
        )
        st.session_state['sim_results'] = sim_results.sort_values("average", ascending=False)

# Create tabs for Team Ratings and Simulation Results
if st.session_state['ratings'] is not None or st.session_state['sim_results'] is not None:
    tab1, tab2 = st.tabs(["Team Ratings", "Simulation Results"])

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

            # Format numerical columns to 2 decimal places
            sim_results_display = st.session_state['sim_results'].copy()

            # Apply 2-decimal formatting to all numerical columns
            for col in sim_results_display.columns:
                if sim_results_display[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    try:
                        sim_results_display[col] = sim_results_display[col].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else x)
                    except:
                        # If formatting fails for any reason, leave the column as-is
                        pass

            st.dataframe(sim_results_display, use_container_width=True)
        else:
            st.info("Click 'Run Simulation' to generate simulation results.")