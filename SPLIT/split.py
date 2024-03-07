import os
from random import randint
import shutil
from pathlib import Path


for entry in os.scandir("train"):
    print(entry.path)
    os.remove(entry)
for entry in os.scandir("valid"):
    print(entry.path)
    os.remove(entry)

for entry in os.scandir("input"):
    if entry.name.endswith(".jpg") and entry.is_file():
        number = randint(1, 100)  # Stores the random number between 1 and 100 in the variable 'number'.
        if number > 20:
            print("train", number, entry.path)
            shutil.copy(entry.path, "train")
            shutil.copy(os.path.join(Path(entry.path).parent, Path(entry.path).stem + ".txt"), "train")
        else:
            print("val", number, entry.path)
            shutil.copy(entry.path, "valid")
            shutil.copy(os.path.join(Path(entry.path).parent, Path(entry.path).stem + ".txt"), "valid")
