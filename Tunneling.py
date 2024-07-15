import os
import pandas as pd
import numpy as np

# Define the directory where the CSV file is located
data_dir = r'C:\Users\alex.britton\Documents\Cursor\Python\PitchOutcomeModel'

# Change the current working directory
os.chdir(data_dir)

# Print the current working directory to confirm the change
print("Current Working Directory:", os.getcwd())

# Load the 7_9_24SPMaster.csv dataset
file_path = os.path.join(data_dir, '7_9_24SPMaster.csv')
data = pd.read_csv(file_path)

# Ensure 'previous_pitch' column is created
data['previous_pitch'] = data.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number']).groupby(['game_pk', 'at_bat_number'])['pitch_type'].shift(1)
data['previous_pitch'] = data['previous_pitch'].fillna('NA')

# Filter out rows where the current pitch and the previous pitch are the same or previous pitch is NA
data = data[(data['pitch_type'] != data['previous_pitch']) & (data['previous_pitch'] != 'NA')]

# Function to calculate the position of the pitch at time t
# y is from plate towards pitcher's mound, x is horizontal across mound, and z is up and down
def calculate_position(release_pos_x, release_pos_z, release_extension, vx0, vy0, vz0, ax, ay, az, t):
    x = release_pos_x + vx0 * t + 0.5 * ax * t**2
    y = release_extension + vy0 * t + 0.5 * ay * t**2
    z = release_pos_z + vz0 * t + 0.5 * az * t**2
    return x, y, z

# Calculate the trajectory for each pitch at different time points
time_points = np.linspace(0, 0.5, 100)  # Time points from 0 to 0.5 seconds
trajectories = []

for index, row in data.iterrows():
    trajectory = [calculate_position(row['release_pos_x'], row['release_pos_z'], row['release_extension'], row['vx0'], row['vy0'], row['vz0'], row['ax'], row['ay'], row['az'], t) for t in time_points]
    trajectories.append(trajectory)

data['trajectory'] = trajectories

# Function to find the tunnel point between two trajectories and return the distance from home plate (y-coordinate)
# Only considering x (horizontal) and z (vertical) coordinates from the batter's perspective
def find_tunnel_point(trajectory1, trajectory2):
    for i in range(len(trajectory1)):
        distance = np.sqrt((trajectory1[i][0] - trajectory2[i][0])**2 + 
                           (trajectory1[i][2] - trajectory2[i][2])**2)  # Only x and z coordinates
        previous_distance = np.sqrt(trajectory1[i][0]**2 + trajectory1[i][2]**2)
        if previous_distance > 0 and (distance / previous_distance) > 0.02:  # 2% deviation
            return time_points[i], trajectory1[i], trajectory2[i], 60.5 - trajectory1[i][1]  # Return distance from home plate
    return None, None, None, None

# Find the tunnel point for each pair of consecutive pitches
tunnel_points = []

for index, row in data.iterrows():
    if index > 0 and index < len(data) and data.iloc[index]['game_pk'] == data.iloc[index - 1]['game_pk'] and data.iloc[index]['at_bat_number'] == data.iloc[index - 1]['at_bat_number']:
        trajectory1 = data.iloc[index - 1]['trajectory']
        trajectory2 = row['trajectory']
        tunnel_point = find_tunnel_point(trajectory1, trajectory2)
        tunnel_points.append(tunnel_point)
    else:
        tunnel_points.append((None, None, None, None))

data['tunnel_point'] = tunnel_points

# Extract the distance from home plate (y-coordinate) from the tunnel points
data['tunnel_distance_feet'] = data['tunnel_point'].apply(lambda x: x[3] if x[3] is not None else 0)

# Adjust tunnel distances by subtracting the release extension
data['adjusted_tunnel_distance_feet'] = data['tunnel_distance_feet'] - data['release_extension']

# Ensure adjusted tunnel distances are within the valid range and greater than zero
data = data[(data['adjusted_tunnel_distance_feet'] <= 60.5) & (data['adjusted_tunnel_distance_feet'] > 0)]

# Sort the data by adjusted tunnel distance in feet in ascending order to get the least tunnel distances
sorted_data = data.sort_values(by='adjusted_tunnel_distance_feet', ascending=True)

# Display the ten smallest adjusted tunnel points with player_name
print("Ten smallest adjusted tunnel points:")
print(sorted_data[['player_name', 'previous_pitch', 'pitch_type', 'adjusted_tunnel_distance_feet']].head(10))

# Calculate the average adjusted tunnel distance for each pitcher
average_tunnel_distances = data.groupby('player_name')['adjusted_tunnel_distance_feet'].mean().reset_index()
average_tunnel_distances = average_tunnel_distances.rename(columns={'adjusted_tunnel_distance_feet': 'average_adjusted_tunnel_distance'})

# Sort the average tunnel distances in ascending order to get the pitchers with the lowest average tunnel distance
average_tunnel_distances = average_tunnel_distances.sort_values(by='average_adjusted_tunnel_distance', ascending=True)

# Calculate the TNL metric (higher value indicates better tunneling ability)
# TNL = ((60.5 - average_adjusted_tunnel_distance) / 60.5) * 1000
average_tunnel_distances['TNL'] = ((60.5 - average_tunnel_distances['average_adjusted_tunnel_distance']) / 60.5 * 1000).astype(int)

# Display the average adjusted tunnel distances and TNL for each pitcher
print("\nAverage adjusted tunnel distances and TNL for each pitcher:")
print(average_tunnel_distances)