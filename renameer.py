import os

def rename_images(folder_path, image_extension):
    image_files = os.listdir(folder_path)
    for i, image_file in enumerate(image_files, start=1):
        old_path = os.path.join(folder_path, image_file)
        new_name = f"image{i}"
        new_path = os.path.join(folder_path, f"{new_name}{image_extension}")
        os.rename(old_path, new_path)

folder_path = "C:\\Users\\BENI\\Desktop\\sports_bike_recognition\\train\\r15"
image_extension = ".jpg"
rename_images(folder_path, image_extension)
