import pandas as pd
import numpy as np

main_df = pd.read_csv(
    'main_pred.csv').set_index("image_id")
# preds = main_df['label']
# IDs = main_df['image_id']
head_df = ['image_id', 'label']
write_freq_df = pd.DataFrame(columns=head_df)
write_comm_df = pd.DataFrame(columns=head_df)
write_rare_df = pd.DataFrame(columns=head_df)
freq_df = pd.read_csv(
    '../final-project-challenge-3-tami/food_data/testcase/sample_submission_freq_track.csv')
comm_df = pd.read_csv(
    '../final-project-challenge-3-tami/food_data/testcase/sample_submission_comm_track.csv')
rare_df = pd.read_csv(
    '../final-project-challenge-3-tami/food_data/testcase/sample_submission_rare_track.csv')

for i in range(len(freq_df)):
    freq_df['label'][i] = main_df.loc[freq_df['image_id'][i], 'label']
for i in range(len(comm_df)):
    comm_df['label'][i] = main_df.loc[comm_df['image_id'][i], 'label']
for i in range(len(rare_df)):
    rare_df['label'][i] = main_df.loc[rare_df['image_id'][i], 'label']

freq_df.to_csv('freq_pred.csv', index=False)
comm_df.to_csv('comm_pred.csv', index=False)
rare_df.to_csv('rare_pred.csv', index=False)
