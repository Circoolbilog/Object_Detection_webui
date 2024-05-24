try:
    from tkinter import filedialog, Tk
except ImportError:
    pass
import os
import sys
import shutil

ENV_EXCLUSION = ["COLAB_GPU", "RUNPOD_POD_ID"]

def get_folder_path(folder_path: str = "") -> str:
    """
    OPens a folder dialog to select a folder, allowing the user to navigate and choose a folder.
    If no folder is selected, returns the initially provided fol;der path or an empty string if not provided.
    This functon is conditioned to skip the folder dialog on macOS or if specific environment variables are present,
    indicating the possible automated environment where the dialog cannot be displayed.

    Parameters:
    - folder_path (str): The initial folder path or an empty string by default.
    Used as the fallback if no folder is selected.

    Returns:
    - str: The path of the folder selected by the user, or the initial 'folder_path' if no selection is made.

    Raises:
    - TypeError: If `folder_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the folder dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the folder dialog
    should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """

    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        # Check for environment variable conditions
        if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
            return folder_path or ""
        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory()
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}")


