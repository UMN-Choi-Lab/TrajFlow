# import pandas as pd

# # Load the CSV
# df = pd.read_csv("08_tracks.csv")

# # Keep only the desired columns
# df_filtered = df[['frame', 'trackId', 'xCenter', 'yCenter']]

# # Sort by frame
# df_filtered = df_filtered.sort_values(by='frame')

# # Split into train and test (80% train, 20% test by frame)
# unique_frames = df_filtered['frame'].unique()
# split_index = int(0.8 * len(unique_frames))
# train_frames = set(unique_frames[:split_index])
# test_frames = set(unique_frames[split_index:])

# df_train = df_filtered[df_filtered['frame'].isin(train_frames)]
# df_test = df_filtered[df_filtered['frame'].isin(test_frames)]

# # Save as tab-delimited, without headers
# df_train.to_csv("ind_train.txt", sep='\t', index=False, header=False)
# df_test.to_csv("ind_test.txt", sep='\t', index=False, header=False)

import pandas as pd

# Load the main CSV
df = pd.read_csv("08_tracks.csv")

# Keep only the desired columns
df_filtered = df[['frame', 'trackId', 'xCenter', 'yCenter']]

# Sort by frame
df_filtered = df_filtered.sort_values(by='frame')

# Compute number of frames per trackId
index_df = df_filtered.groupby('trackId')['frame'].nunique().reset_index(name='numFrames')

# Filter trackIds with 200 < numFrames < 1000
target_track_ids = index_df[(index_df['numFrames'] > 200) & (index_df['numFrames'] < 1000)]['trackId']

# Filter the main dataframe to keep only valid trackIds
df_filtered = df_filtered[df_filtered['trackId'].isin(target_track_ids)]

# Split into train and test (80% train, 20% test by frame)
unique_frames = df_filtered['frame'].unique()
split_index = int(0.75 * len(unique_frames))
train_frames = set(unique_frames[:split_index])
test_frames = set(unique_frames[split_index:])

df_train = df_filtered[df_filtered['frame'].isin(train_frames)]
df_test = df_filtered[df_filtered['frame'].isin(test_frames)]

# Save as tab-delimited, without headers
df_train.to_csv("ind_train.txt", sep='\t', index=False, header=False)
df_test.to_csv("ind_test.txt", sep='\t', index=False, header=False)