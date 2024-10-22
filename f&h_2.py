import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from skimage.transform import resize
from scipy.ndimage import center_of_mass


def felzenszwalb_segmentation(input_image_path, scale=100, sigma=0.5, min_size=50):
    # Load the input image
    image = io.imread(input_image_path)

    # Resize the image (optional: for faster processing)
    image = resize(image, (512, 512))  # Resize to 512x512 (adjust as needed)

    # Perform Felzenszwalb segmentation
    segments = segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)

    # Display the segmented image with average segment colors
    segmented_image = color.label2rgb(segments, image, kind='avg')

    # plt.figure(figsize=(8, 8))
    # plt.imshow(segmented_image)
    # plt.axis('off')  # Hide axes
    # plt.show()

    return segments, image


def extract_segment_colors(segments, image):
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


def export_segment_colors_to_csv(segment_colors, output_csv_path):
    # Convert the segment_colors dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(segment_colors, orient='index', columns=['R', 'G', 'B'])
    df.index.name = 'Segment'

    # Export the DataFrame to CSV
    df.to_csv(output_csv_path)
    print(f"Segment colors have been exported to {output_csv_path}")


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


# Example usage:
input_image_path = 'check.png'
output_csv_path = 'felzenszwalb_segment_colors.csv'

# Parameters for Felzenszwalb's algorithm
scale = 100
sigma = 0.5
min_size = 50

# Perform segmentation and extract segment colors
segments, image = felzenszwalb_segmentation(input_image_path, scale=scale, sigma=sigma, min_size=min_size)
segment_colors = extract_segment_colors(segments, image)

# Export the segment colors to a CSV file
export_segment_colors_to_csv(segment_colors, output_csv_path)

# Display the segmented image with segment IDs
display_segments_with_ids(segments, image)
