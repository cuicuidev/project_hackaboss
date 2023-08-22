import os
import shutil
import re

def sort_images_into_folders(root_path, categories):
    for category in categories:  # loop over 'train', 'test', 'val'
        category_path = os.path.join(root_path, category)
        
        # Get the list of image files in the category directory
        image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        
        for image_file in image_files:
            # Use a regular expression to find the car make in the filename
            match = re.search(r'_([a-zA-Z]+)', image_file)
            if match:
                car_make = match.group(1)
                
                # Create a directory for this car make if it doesn't already exist
                car_make_dir = os.path.join(category_path, car_make)
                if not os.path.exists(car_make_dir):
                    os.makedirs(car_make_dir)
                    
                # Move the image into the corresponding car make directory
                src_path = os.path.join(category_path, image_file)
                dest_path = os.path.join(car_make_dir, image_file)
                shutil.move(src_path, dest_path)
            else:
                print(f"Could not extract car make from {image_file}")

# Define the root path and categories
root_path = 'notebooks/car_make_images'
categories = ['train', 'test', 'val']

# Call the function
sort_images_into_folders(root_path, categories)