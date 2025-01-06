import os
import pandas as pd

tracks = pd.read_csv(os.path.join(f'data', f'08_tracks.csv'))
tracks_metadata = pd.read_csv(os.path.join(f'data', f'08_tracksMeta.csv'))

mask = tracks_metadata['class'].isin(['car', 'truck_bus'])
index_df = tracks_metadata[mask]
numframes = index_df['numFrames']
target_track_ids = index_df[(numframes < 1000) & (numframes > 200)]['trackId']

filtered_tracks = tracks[tracks['trackId'].isin(target_track_ids)]
filtered_tracks = filtered_tracks[['frame', 'trackId', 'xCenter', 'yCenter']].sort_values(by=['frame', 'trackId'])

split_index = int(len(filtered_tracks) * 0.75)
train = filtered_tracks.iloc[:split_index]
test = filtered_tracks.iloc[split_index:]

train.to_csv('ind_train.csv', sep='\t', header=False, index=False)
test.to_csv('ind_test.csv', sep='\t', header=False, index=False)
