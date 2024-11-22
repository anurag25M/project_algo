# import pandas as pd
#
# # Load the first and second CSV files
# df1 = pd.read_csv('color_map_with_values.csv')
# df2 = pd.read_csv('felzenszwalb_segment_colors.csv')
#
# # Merge based on RGB values (R, G, B) columns
# merged_df = pd.merge(df2, df1[['R', 'G', 'B', 'Assigned_Value']], on=['R', 'G', 'B'], how='left')
#
# # Fill NaN in 'Assigned_Value' for non-matching rows (optional)
# merged_df['Assigned_Value'].fillna('No Match', inplace=True)
#
# # Save the merged file to a new CSV
# merged_df.to_csv('merged_file.csv', index=False)
# import pandas as pd
# import numpy as np
#
# # Load the first and second CSV files
# df1 = pd.read_csv('color_map_with_values.csv')
# df2 = pd.read_csv('felzenszwalb_segment_colors.csv')
#
# # Merge based on RGB values (R, G, B) columns
# merged_df = pd.merge(df2, df1[['R', 'G', 'B', 'Assigned_Value']], on=['R', 'G', 'B'], how='left')
#
# # Function to find the closest RGB match
# def find_closest(row):
#     if pd.isna(row['Assigned_Value']):  # Check if there is no exact match
#         # Calculate the distance to each RGB in df1
#         distances = np.sqrt((df1['R'] - row['R']) ** 2 +
#                             (df1['G'] - row['G']) ** 2 +
#                             (df1['B'] - row['B']) ** 2)
#         # Find the index of the closest match
#         closest_index = distances.idxmin()
#         return df1.loc[closest_index, 'Assigned_Value']  # Return the closest Assigned_Value
#     return row['Assigned_Value']  # Return the original Assigned_Value if match found
#
# # Apply the function to fill Assigned_Value for non-matching rows
# merged_df['Assigned_Value'] = merged_df.apply(find_closest, axis=1)
#
# # Save the merged file to a new CSV
# merged_df.to_csv('merged_file.csv', index=False)
import pandas as pd
import numpy as np
from skimage.color import rgb2lab

# Load the first and second CSV files
df1 = pd.read_csv('color_map_with_values.csv')
df2 = pd.read_csv('felzenszwalb_segment_colors.csv')

# Convert RGB values in both dataframes to LAB color space
df1[['L', 'A', 'B']] = rgb2lab(df1[['R1', 'G1', 'B1']].values / 255.0)
df2[['L', 'A', 'B']] = rgb2lab(df2[['R1', 'G1', 'B1']].values / 255.0)

# Merge based on RGB values (R, G, B) columns to find exact matches
merged_df = pd.merge(df2, df1[['R1', 'G1', 'B1', 'Assigned_Value']], on=['R1', 'G1', 'B1'], how='left')

# Function to find the closest LAB match based on CIE76 distance
# def find_closest_lab(row):
#     if pd.isna(row['Assigned_Value']):  # Check if there is no exact match
#         # Calculate the CIE76 (Euclidean) distance to each LAB color in df1
#         distances = np.sqrt((df1['L'] - row['L']) ** 2 +
#                             (df1['A'] - row['A']) ** 2 +
#                             (df1['B'] - row['B']) ** 2)
#         # Find the index of the closest match
#         closest_index = distances.idxmin()
#         return df1.loc[closest_index, 'Assigned_Value']  # Return the closest Assigned_Value
#     return row['Assigned_Value']  # Return the original Assigned_Value if exact match found

def find_closest_lab(row):
    if pd.isna(row['Assigned_Value']):  # Check if there is no exact match
        # Calculate the CIE76 (Euclidean) distance to each LAB color in df1
        distances = np.sqrt((df1['L'] - row['L']) ** 2 +
                            (df1['A'] - row['A']) ** 2 +
                            (df1['B'] - row['B']) ** 2)
        # Sort distances to find the two closest matches
        sorted_indices = distances.sort_values().index
        closest_index = sorted_indices[0]

        # Handle interpolation between the two closest values
        if len(sorted_indices) > 1:
            second_closest_index = sorted_indices[1]
            closest_value = df1.loc[closest_index, 'Assigned_Value']
            second_closest_value = df1.loc[second_closest_index, 'Assigned_Value']
            # Assign a value midway between the two closest values
            return (closest_value + second_closest_value) / 2
        else:
            # If only one match, assign slightly above the closest value
            return df1.loc[closest_index, 'Assigned_Value'] + 0.001

    return row['Assigned_Value']  # Return the original Assigned_Value if exact match found


# Apply the function to fill Assigned_Value for non-matching rows
merged_df['Assigned_Value'] = merged_df.apply(find_closest_lab, axis=1)

# Drop the LAB columns before saving to CSV (optional)
merged_df = merged_df.drop(columns=['L', 'A', 'B'])

# Save the merged file to a new CSV
merged_df.to_csv('merged_file_2.csv', index=False)
