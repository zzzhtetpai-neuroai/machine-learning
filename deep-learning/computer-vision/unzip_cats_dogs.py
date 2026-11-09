import zipfile
import os

train_zip = os.path.join("data", "train.zip")
train_folder = os.path.join("data", "train")

os.makedirs(train_folder, exist_ok=True)

with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(train_folder)

print("Train images extracted to:", train_folder)

