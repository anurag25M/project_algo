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

    # Display the segmented image with average segment colors and contours
    segmented_image = color.label2rgb(segments, image, kind='avg')

    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)

    # Add contour for the Felzenszwalb segments
    plt.contour(segments, colors='red', linewidths=0.5)

    plt.axis('off')  # Hide axes
    plt.show()

    return segments, image


def slic_within_segments(segments, image, n_segments=9, compactness=10):
    slic_segment_colors = []

    for felzenszwalb_id in np.unique(segments):
        # Create a mask for each Felzenszwalb segment
        mask = segments == felzenszwalb_id
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]

        # Apply SLIC to each Felzenszwalb segment individually
        slic_segments = segmentation.slic(masked_image, n_segments=n_segments, compactness=compactness, mask=mask,
                                          start_label=1)

        # Add SLIC contours within each Felzenszwalb segment
        plt.contour(slic_segments, colors='blue', linewidths=0.3)

        # Loop through each SLIC segment and get average color
        for slic_id in np.unique(slic_segments[slic_segments > 0]):  # Skip background
            slic_mask = slic_segments == slic_id
            mean_color = np.mean(image[slic_mask], axis=0)
            mean_color = np.clip(mean_color * 255, 0, 255).astype(int)
            slic_segment_colors.append({
                'Felzenszwalb_Segment': felzenszwalb_id,
                'SLIC_Segment': slic_id,
                'R': mean_color[0],
                'G': mean_color[1],
                'B': mean_color[2]
            })

    plt.axis('off')
    plt.show()

    return slic_segment_colors


def export_segment_colors_to_csv(segment_colors, output_csv_path):
    # Convert the segment_colors list to a pandas DataFrame
    df = pd.DataFrame(segment_colors)

    # Export the DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Segment colors have been exported to {output_csv_path}")


# Example usage:
input_image_path = 'check.png'
output_csv_path = 'slic_within_felzenszwalb_segments.csv'

# Parameters for Felzenszwalb's algorithm
scale = 100
sigma = 0.5
min_size = 50

# Perform Felzenszwalb segmentation
segments, image = felzenszwalb_segmentation(input_image_path, scale=scale, sigma=sigma, min_size=min_size)

# Apply SLIC within each Felzenszwalb segment and extract colors with contours
slic_segment_colors = slic_within_segments(segments, image)

# Export the combined segment colors to a CSV file
export_segment_colors_to_csv(slic_segment_colors, output_csv_path)
