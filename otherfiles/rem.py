import os

def remove_png_files():
    """Remove all .png files in the current directory."""
    files = os.listdir('.')
    for file in files:
        if file.endswith('.png'):
            os.remove(file)
            print(f"Removed: {file}")


