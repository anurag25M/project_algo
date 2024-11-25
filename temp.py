import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from scipy.ndimage import center_of_mass

def grid_segmentation(input_image_path, num_rows=10, num_cols=10):
    # Load the input image
    image = io.imread(input_image_path)

    # Resize the image (optional: for faster processing)
    image = resize(image, (512, 512))  # Resize to 512x512 (adjust as needed)

    # Get the dimensions of the image
    img_height, img_width, _ = image.shape

    # Calculate the size of each grid segment
    grid_height = img_height // num_rows
    grid_width = img_width // num_cols

    # Create an array to store segment labels
    segments = np.zeros((img_height, img_width), dtype=int)

    # Assign segment labels based on the grid
    segment_id = 0
    for row in range(num_rows):
        for col in range(num_cols):
            segments[row * grid_height:(row + 1) * grid_height, col * grid_width:(col + 1) * grid_width] = segment_id
            segment_id += 1

    # Display the segmented image with each grid cell shown in different colors
    segmented_image = color.label2rgb(segments, image, kind='avg')

    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.axis('off')  # Hide axes
    plt.show()

    return segments, image

def extract_grid_segment_colors(segments, image):
    segment_ids = np.unique(segments)
    segment_colors = {}

    for seg_id in segment_ids:
        # Create a mask for each segment
        mask = segments == seg_id
        # Compute the mean color of the segment
        mean_color = np.mean(image[mask], axis=0)
        # Scale the color values to 0-255 range
        mean_color = np.clip(mean_color * 255, 0, 255).astype(int)
        segment_colors[seg_id] = mean_color

    return segment_colors

# def export_segment_colors_to_csv(segment_colors, min_value, max_value, output_csv_path):
#     # Convert the segment_colors dictionary to a pandas DataFrame
#     df = pd.DataFrame.from_dict(segment_colors, orient='index', columns=['R', 'G', 'B'])
#     df.index.name = 'Segment'
#
#     # Generate values based on the range divided equally for each segment
#     num_segments = len(df)
#     df['Assigned_Value'] = np.linspace(min_value, max_value, num_segments)
#
#     # Export the DataFrame to CSV
#     df.to_csv(output_csv_path)
#     print(f"Segment colors and assigned values have been exported to {output_csv_path}")
def export_segment_colors_to_csv(segment_colors, min_value, max_value, output_csv_path):
    # Convert the segment_colors dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(segment_colors, orient='index', columns=['R', 'G', 'B'])
    df.index.name = 'Segment'

    # Exclude segment 0
    df = df[df.index != 0]

    # Generate values based on the range, with the max value at the top (first row) and min at the bottom
    num_segments = len(df)
    df['Assigned_Value'] = np.linspace(max_value, min_value, num_segments)

    # Export the DataFrame to CSV
    df.to_csv(output_csv_path)
    print(f"Segment colors and assigned values have been exported to {output_csv_path}")

def display_segments_with_ids(segments, image):
    # Get the segmented image with average colors
    segmented_image = color.label2rgb(segments, image, kind='avg')

    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)

    # Calculate and display the segment IDs at the centroids of the segments
    segment_ids = np.unique(segments)
    for seg_id in segment_ids:
        # Find the centroid of each segment
        mask = segments == seg_id
        if np.any(mask):  # Avoid potential empty segments
            centroid = center_of_mass(mask)
            plt.text(centroid[1], centroid[0], str(seg_id), color='white', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
input_image_path = 'color_map_crop.jpg'
output_csv_path = 'color_map_with_values.csv'
num_rows = 35  # Number of rows in the grid
num_cols = 1  # Number of columns in the grid
min_value = 1.000  # Minimum value in the range
max_value = 1.126  # Maximum value in the range

segments, image = grid_segmentation(input_image_path, num_rows, num_cols)
segment_colors = extract_grid_segment_colors(segments, image)
export_segment_colors_to_csv(segment_colors, min_value, max_value, output_csv_path)
display_segments_with_ids(segments, image)