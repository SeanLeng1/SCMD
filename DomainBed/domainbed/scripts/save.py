import os
import shutil

# Replace these with the paths of the input and output directories
input_dir = "sweep/PACS/outputs"
output_dir = "sweep/PACS/log"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Walk through the directory structure
for root, dirs, files in os.walk(input_dir):
    for file in files:
        # Check if the current file is 'result.json'
        if file == "results.jsonl" or file == 'out.txt':
            # Construct the full path to the source and destination files
            src_file = os.path.join(root, file)

            # Create a new directory for each file in the output directory
            new_folder = os.path.join(output_dir, os.path.basename(root))
            os.makedirs(new_folder, exist_ok=True)

            dst_file = os.path.join(new_folder, file)

            # Copy the 'result.json' file to the output directory
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

