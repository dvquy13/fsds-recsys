import os


def list_files_in_folder(folder_path):
    # List all files in the given folder
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return files
