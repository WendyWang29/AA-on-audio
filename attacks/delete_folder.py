import os
import shutil
from tqdm import tqdm

def delete_folder_with_progress(folder_path):
    # List all files and folders in the directory
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))
        for dir in dirs:
            all_files.append(os.path.join(root, dir))
    # Show progress while deleting
    for item in tqdm(all_files, desc="Deleting files", unit="file"):
        try:
            if os.path.isfile(item):
                os.remove(item)  # Remove file
            elif os.path.isdir(item):
                shutil.rmtree(item)  # Remove directory and its contents
        except Exception as e:
            print(f"Error deleting {item}: {e}")

    # After deleting all files and subfolders, delete the main folder
    shutil.rmtree(folder_path, ignore_errors=True)
    print(f"Folder {folder_path} deleted successfully.")

if __name__ == "__main__":
    folder_path = 'FGSM_3s_ResNet'
    delete_folder_with_progress(folder_path)