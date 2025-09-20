import SALibrary.scrape as sc
import SALibrary.simulate as sim
import SALibrary.SimpleRatingSystem as srs
import numpy as np

# Get schedule and standings from the website
# note: the date column is in the format mm/dd/yyyy
schedule, standings = sc.scrape_mlb(2025)
schedule.head()

standings

# Keeps only games that have been played to compute ratings
schedule = srs.months_test_set_MLB(schedule, standings)
schedule_so_far = schedule.dropna()
num_teams = len(np.unique(np.ndarray.flatten(np.array([schedule_so_far['home team'].unique(),
                schedule_so_far['away team'].unique()]))))

# Compute ratings with home advantage
ratings, home_advantage, rmse = srs.srs_teamtrain(schedule_so_far['home team'], schedule_so_far['away team'], schedule_so_far['home score'], schedule_so_far['away score'])
ratings

# Compute probabilities of winning each game based on ratings
# note: the date column is in the format /mm/dd/yyyy
schedule_probs = srs.SRS_to_probabilities(ratings, rmse, schedule, home_advantage)
print(rmse)
schedule_probs.head()

# Show the probabilities of winning each game for the first 'future games' that will be used in the simulation
schedule_probs[schedule_probs['away score'].isnull()].head()

# Simulate the end of the season using win probabilities
sim_results = sim.MonteCarlo(sim.simulate_mlb, schedule_probs, standings, trials=1000)

# Sort the results
sim_results = sim_results.sort_values('average', ascending = 0)
ratings = ratings.sort_values('Rating', ascending = 0)

# Probabilities for each team to win its division
sim_results

