import os
from PIL import Image
import numpy as np
import cv2

def read_image_as_array(image_path):
    """Read an image from the given path and convert it to a numpy array."""
    with Image.open(image_path) as img:
        return np.asarray(img)

def save_image_from_array(image_array, output_path):
    """Save a numpy array as an image to the given path."""
    img = Image.fromarray(image_array)
    img.save(output_path)

def compute_abs_difference(image1, image2):
    """Compute the absolute difference between two images."""
    return np.abs(image1.astype(np.float) - image2.astype(np.float))

def normalize_image(image):
    """Normalize image data to 0-255 scale."""
    image_min = np.min(image)
    image_max = np.max(image)
    return ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

def process_folders(folder1, folder2, output_folder):
    """Process all matching images in two folders and save the differences."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List files in both folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # Find intersection of both file sets
    common_files = files1 & files2

    # Process each file
    for filename in common_files:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        output_path = os.path.join(output_folder, filename)

        # Load images
        image1 = read_image_as_array(path1)
        #resize image1 to 384x384
        image1 = cv2.resize(image1, (384, 384))
        image2 = read_image_as_array(path2)
        image2 = normalize_image(image2)
        image1 = normalize_image(image1)
        # Compute absolute difference
        difference = compute_abs_difference(image1, image2)

        # Normalize the result
        normalized_difference = normalize_image(difference)

        # Save the normalized image
        save_image_from_array(normalized_difference, output_path)
        print(f"Processed {filename}")

# Set your folder paths and output folder path
folder1_path = '/root/RawSense/flatnet/data/flatnet_val/gts'
folder2_path = '/root/RawSense/flatnet/data/flatnet_val/decoded_sim_captures'
output_folder_path = '/root/RawSense/flatnet/data/flatnet_val/null_space'
process_folders(folder1_path, folder2_path, output_folder_path)


folder1_path = "/root/RawSense/LenslessPiCam/outputs/2024-08-12/11-43-17/SimPhlatCam_raw_1518x2012"
folder2_path = '/root/RawSense/flatnet/data/flatnet_val/decoded_sim_captures_disfa'
output_folder_path = '/root/RawSense/flatnet/data/flatnet_val/null_space_disfa'
process_folders(folder1_path, folder2_path, output_folder_path)

folder1_path = "/root/StableSR/data/flatnet_sim_output_384_val/inputs"
folder2_path = '/root/StableSR/data/flatnet_sim_output_384_val/decoded_sim_captures'
output_folder_path = '/root/StableSR/data/flatnet_sim_output_384_val/null_space'
# Process folders and save results
process_folders(folder1_path, folder2_path, output_folder_path)