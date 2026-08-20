import os
import glob

# Function to list all .obj files in the folder
def get_obj_files(folder, uploaded_files_dict):
    preloaded_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.obj'))]
    uploaded_files = list(uploaded_files_dict.keys())
    return preloaded_files + uploaded_files