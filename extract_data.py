import tarfile
import os

# Define paths
input_folder = r"C:\Users\Mike\Desktop\CLASSES\Data Analytics and Projects II\Week 6\FIDO\Stanford"
output_folder = r"C:\Users\Mike\Desktop\CLASSES\Data Analytics and Projects II\Week 6\FIDO\Dogdata"

# Files to extract
tar_files = ["Images.tar", "Annotation.tar"]

# Extract files
for tar_name in tar_files:
    tar_path = os.path.join(input_folder, tar_name)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_folder)
        print(f"Extracted {tar_name} to {output_folder}")