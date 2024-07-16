import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where the CSV file is located
data_dir = r'C:\Users\alex.britton\Documents\Cursor\Python\Tunneling'

# Change the current working directory
os.chdir(data_dir)

# Print the current working directory to confirm the change
print("Current Working Directory:", os.getcwd())

# Load the 7_9_24SPMaster.csv dataset
file_path = os.path.join(data_dir, '7_9_24SPMaster.csv')
data = pd.read_csv(file_path)

# Debug: Print the number of unique pitchers in the original dataset
print(f"Number of unique pitchers in the original dataset: {data['player_name'].nunique()}")

# Function to calculate the position of the pitch at time t
# y is from plate towards pitcher's mound, x is horizontal across mound, and z is up and down
def calculate_position(release_pos_x, release_pos_z, release_extension, vx0, vy0, vz0, ax, ay, az, t):
    x = release_pos_x + vx0 * t + 0.5 * ax * t**2
    y = 60.5 + (release_extension + vy0 * t + 0.5 * ay * t**2)  # Adjust y to start from 60.5 - release_extension
    z = release_pos_z + vz0 * t + 0.5 * az * t**2
    return x, y, z

# Calculate the trajectory for each pitch at different time points
time_points = np.linspace(0, 0.5, 100)  # Time points from 0 to 0.5 seconds
trajectories = []

for index, row in data.iterrows():
    trajectory = [calculate_position(row['release_pos_x'], row['release_pos_z'], row['release_extension'], row['vx0'], row['vy0'], row['vz0'], row['ax'], row['ay'], row['az'], t) for t in time_points]
    trajectories.append(trajectory)

data['trajectory'] = trajectories

# Calculate the average trajectory for each pitcher's fastball
fastball_data = data[data['pitch_type'] == 'FF']  # Assuming 'FF' is the code for fastball
average_fastball_trajectories = fastball_data.groupby('player_name')['trajectory'].apply(lambda x: np.mean(np.array(x.tolist()), axis=0))

# Function to find the tunnel point between a pitch and the average fastball trajectory
def find_tunnel_point(trajectory, average_fastball_trajectory, deviation_threshold=0.10):
    for i in range(len(trajectory)):
        distance = np.sqrt((trajectory[i][0] - average_fastball_trajectory[i][0])**2 + 
                           (trajectory[i][2] - average_fastball_trajectory[i][2])**2)  # x and z coordinates
        previous_distance = np.sqrt(average_fastball_trajectory[i][0]**2 + average_fastball_trajectory[i][2]**2)
        if previous_distance > 0 and (distance / previous_distance) > deviation_threshold:  # Deviation threshold
            return time_points[i], trajectory[i], average_fastball_trajectory[i], 60.5 - trajectory[i][1]  # Return distance from home plate (y-coordinate)
    return None, None, None, None

# Calculate the tunnel point for each pitch in reference to the average fastball trajectory
tunnel_points = []

for index, row in data.iterrows():
    player_name = row['player_name']
    if player_name in average_fastball_trajectories.index:
        average_fastball_trajectory = average_fastball_trajectories[player_name]
        trajectory = row['trajectory']
        tunnel_point = find_tunnel_point(trajectory, average_fastball_trajectory)
        tunnel_points.append(tunnel_point)
    else:
        tunnel_points.append((None, None, None, None))

data['tunnel_point'] = tunnel_points

# Extract the distance from home plate (y-coordinate) from the tunnel points
data['tunnel_distance_feet'] = data['tunnel_point'].apply(lambda x: x[3] if x[3] is not None else np.nan)

# Drop rows with NaN values in the tunnel_distance_feet column
data = data.dropna(subset=['tunnel_distance_feet'])

# Calculate the number of unique pitch trajectories for each pitcher
unique_pitch_trajectories = data.groupby('player_name')['pitch_type'].nunique().reset_index()
unique_pitch_trajectories = unique_pitch_trajectories.rename(columns={'pitch_type': 'unique_pitch_trajectories'})

# Filter pitchers with 4 or more unique pitch trajectories
pitchers_with_4_or_more_trajectories = unique_pitch_trajectories[unique_pitch_trajectories['unique_pitch_trajectories'] >= 4]

# Calculate the average tunnel distance for each pitcher
average_tunnel_distances = data.groupby('player_name')['tunnel_distance_feet'].mean().reset_index()
average_tunnel_distances = average_tunnel_distances.rename(columns={'tunnel_distance_feet': 'average_tunnel_distance'})

# Merge the unique pitch trajectories and average tunnel distances
pitcher_stats = pd.merge(pitchers_with_4_or_more_trajectories, average_tunnel_distances, on='player_name')

# Calculate the TNL metric (higher value indicates better tunneling ability)
# TNL = ((60.5 - average_tunnel_distance) / 60.5) * 1000
pitcher_stats['TNL'] = ((60.5 - pitcher_stats['average_tunnel_distance']) / 60.5 * 1000).astype(int)

# Sort the pitchers by TNL metric in descending order
pitcher_stats = pitcher_stats.sort_values(by='TNL', ascending=False)

# Display the top 10 pitchers by TNL metric
top_20_pitchers = pitcher_stats.head(20)
print("\nTop 10 pitchers by TNL metric:")
print(top_20_pitchers)

# Visualization: Plot average trajectories for the top 10 pitchers
plt.figure(figsize=(12, 6))
for i in range(min(20, len(top_20_pitchers))):
    player_name = top_20_pitchers.iloc[i]['player_name']
    average_trajectory = average_fastball_trajectories.loc[player_name]
    x = [pos[0] for pos in average_trajectory]
    y = [pos[1] for pos in average_trajectory]
    z = [pos[2] for pos in average_trajectory]
    plt.plot(y, z, label=player_name)
plt.xlabel('Distance from Home Plate (feet)')
plt.ylabel('Vertical Position (feet)')
plt.title('Average Fastball Trajectories of the Top 10 Pitchers by TNL')
plt.legend()
plt.show()

# Visualization: Plot average tunnel distances for the top 10 pitchers
plt.figure(figsize=(12, 6))
plt.barh(top_20_pitchers['player_name'], top_20_pitchers['average_tunnel_distance'], color='skyblue')
plt.xlabel('Average Tunnel Distance (feet)')
plt.ylabel('Pitcher')
plt.title('Average Tunnel Distances for the Top 10 Pitchers by TNL')
plt.gca().invert_yaxis()
plt.show()