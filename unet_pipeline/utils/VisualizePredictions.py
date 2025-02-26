import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Path to the saved model predictions
RESULT_PATH = "/Users/fenggao/Library/CloudStorage/GoogleDrive-2023isanewjourney@gmail.com/My Drive/result_top3.pkl"  # Update this if needed
IMAGE_FOLDER = "/Users/fenggao/Library/CloudStorage/GoogleDrive-2023isanewjourney@gmail.com/My Drive/kaggle/img-classif/input/dataset1024/test"  # Update this to your local image folder

# Load the predicted masks from the pickle file
def load_predictions(result_path):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"File {result_path} not found!")

    with open(result_path, "rb") as handle:
        mask_dict = pickle.load(handle)
    
    print(f"Loaded {len(mask_dict)} predicted masks.")
    return mask_dict

# Function to visualize an image and its predicted mask
def visualize_prediction(image_path, mask, alpha=0.5):
    """Overlay mask on the original image for visualization."""
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Skipping...")
        return

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    mask = (mask * 255).astype(np.uint8)  # Normalize mask to [0, 255]

    # Convert grayscale mask to color
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  
    
    # Blend the image and mask
    overlay = cv2.addWeighted(img, 1 - alpha, mask_colored, alpha, 0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()

def main():
    # Load masks
    mask_dict = load_predictions(RESULT_PATH)

    # Display a few sample predictions
    for i, (image_id, mask) in enumerate(mask_dict.items()):
        image_path = os.path.join(IMAGE_FOLDER, f"{image_id}.png")  # Update extension if needed
        visualize_prediction(image_path, mask)

        if i == 4:  # Show only first 5 images
            break

if __name__ == "__main__":
    main()
