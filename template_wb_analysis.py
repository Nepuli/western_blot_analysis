'''WESTERN BLOT ANALYSIS'''
'''owner: Nathalie Zgoda'''
'''hours lost in code: 4,3h'''
'''purpose: modified script for GP2025'''
'''this version is not working due to different adjustments'''

# This script demonstrates how to analyze a Western blot image
# using Python. It includes:
#   - Manual lane and band selection
#   - Background subtraction (local average)
#   - Multi-color channel handling (RGB)
#   - Molecular weight calibration using a marker lane


# --- 1. IMPORT REQUIRED LIBRARIES -------------------------------------
import numpy as np                # for numerical operations
import matplotlib.pyplot as plt   # for plotting and visualization
from skimage import io, color     # for image loading and processing
import pandas as pd               # for storing and exporting results


# --- 2. LOAD IMAGE ----------------------------------------------------
# TODO: Change this to your own Western blot image path
img = io.imread("western.png")  #--> you can use other ways of load the image


# Check if image has color channels (3D array: height x width x 3)
if img.ndim == 3:
    # Split channels: assume Red = β-Actin (loading), Green = Target, Blue = unused or marker
    red = img[..., 0].astype(float)   # Red channel
    green = img[..., 1].astype(float) # Green channel
    blue = img[..., 2].astype(float)  # Blue channel (optional)
else:
    # If grayscale, just duplicate the single channel for demonstration
    red = green = blue = img.astype(float)


# Show the original image
plt.imshow(img)
plt.title("Original Image - Click to select lanes (including marker lane!)")
plt.show()


# --- 3. MANUAL LANE SELECTION -----------------------------------------
# click once on the center of each lane (including the marker)Right click to finish.
print("Left click the center of each lane (including the marker lane). Right click to finish.")
plt.imshow(img)
lane_points = plt.ginput(n=-1, timeout=0)
plt.close()

# Extract x positions (lane centers)
lane_centers = [int(p[0]) for p in lane_points]
print("Lane centers (pixels):", lane_centers)

# --- 4. DEFINE BAND REGIONS MANUALLY ---------------------------------
# define the y-range for each type of band.
plt.imshow(img)
plt.title("Click top and bottom of the TARGET (Green) band region")
target_band = plt.ginput(2)
plt.close()

plt.imshow(img)
plt.title("Click top and bottom of the β-Actin (Red) band region")
loading_band = plt.ginput(2)
plt.close()

# Convert clicks to integer y-coordinates
target_y = (int(target_band[0][1]), int(target_band[1][1]))
loading_y = (int(loading_band[0][1]), int(loading_band[1][1]))
print("Target band y-range:", target_y)
print("Loading band y-range:", loading_y)

# --- 5. MOLECULAR WEIGHT MARKER CALIBRATION ---------------------------
# Optional: define marker band positions (for MW calibration)
print("Click on the CENTER of visible marker bands (ladder) in order from TOP to BOTTOM.")
plt.imshow(img)
marker_points = plt.ginput(n=-1, timeout=0)
plt.close()

# Adjust the molecular weights of each marker band (example values)
known_MW = [100, 75, 50, 37, 25, 20]  # in kDa (adjust is needed!!)

# Extract marker pixel y-positions
marker_y = np.array([p[1] for p in marker_points])

# Fit a linear model between pixel position and log(MW)
# (band position roughly follows a log-linear relationship)
fit = np.polyfit(marker_y, np.log10(known_MW[:len(marker_y)]), 1)
print("Marker calibration fit coefficients:", fit)

def estimate_MW(y_pixel):
    """Convert a y-position (pixel) to approximate molecular weight."""
    logMW = np.polyval(fit, y_pixel)
    return 10 ** logMW


# --- 6. SIGNAL MEASUREMENT WITH LOCAL BACKGROUND ----------------------
lane_width = 20   # adjustable lane width (in pixels)
bg_margin = 15    # distance between band and background area
bg_thickness = 10 # thickness of background region

def measure_signal(channel, x_center, y0, y1):
    """
    Measures the integrated signal in a rectangular ROI,
    subtracting the local background estimated from
    regions above and below the band.
    """
    # Define ROI boundaries for the band
    x0 = int(max(0, x_center - lane_width / 2))
    x1 = int(min(channel.shape[1], x_center + lane_width / 2))
    y0 = int(max(0, y0))
    y1 = int(min(channel.shape[0], y1))

    # Band ROI
    roi_band = channel[y0:y1, x0:x1]

    # Background regions (above and below band)
    y_top0 = int(max(0, y0 - bg_margin - bg_thickness))
    y_top1 = int(max(0, y0 - bg_margin))
    y_bot0 = int(min(channel.shape[0], y1 + bg_margin))
    y_bot1 = int(min(channel.shape[0], y1 + bg_margin + bg_thickness))

    # Extract background pixel values
    bg_top = channel[y_top0:y_top1, x0:x1]
    bg_bottom = channel[y_bot0:y_bot1, x0:x1]

    # Combine and take mean background intensity
    bg_values = np.concatenate([bg_top.flatten(), bg_bottom.flatten()])
    bg_mean = np.mean(bg_values) if bg_values.size > 0 else 0

    # Subtract background (band area * background intensity)
    corrected_signal = roi_band.sum() - bg_mean * roi_band.size
    return max(corrected_signal, 0)

# --- 7. MEASURE EACH LANE --------------------------------------------
results = []
for i, xc in enumerate(lane_centers, start=1):
    target_int = measure_signal(green, xc, *target_y)
    loading_int = measure_signal(red, xc, *loading_y)
    ratio = target_int / loading_int if loading_int > 0 else np.nan

    # Estimate molecular weight for each band’s vertical position
    target_MW = estimate_MW(np.mean(target_y))
    loading_MW = estimate_MW(np.mean(loading_y))

    results.append({
        "Lane": i,
        "Target_Intensity": target_int,
        "Loading_Intensity": loading_int,
        "Normalized_Ratio": ratio,
        "Target_MW_kDa": target_MW,
        "Loading_MW_kDa": loading_MW
            })
    

# --- 8. CREATE TABLE AND VISUALIZE -----------------------------------
df = pd.DataFrame(results)
print("\nQuantification Results:\n", df)

# Bar plot of normalized expression
plt.bar(df["Lane"], df["Normalized_Ratio"])
plt.xlabel("Lane")
plt.ylabel("Target / β-Actin")
plt.title("Normalized Protein Expression")
plt.show()

# --- 9. SAVE RESULTS --------------------------------------------------
df.to_csv("western_quant_results.csv", index=False)
print("Results saved as 'western_quant_results.csv'")
