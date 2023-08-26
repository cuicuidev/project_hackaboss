import os
import shutil

def run(hard: bool = False):
    # Check if 'history.csv' exists and delete it
    if os.path.exists('history.csv'):
        os.remove('history.csv')
        print("Deleted file: history.csv")
    else:
        print("File history.csv does not exist")

    # Check if 'models' folder exists and delete it
    if os.path.exists('models'):
        shutil.rmtree('models')
        print("Deleted folder: models")
    else:
        print("Folder models does not exist")
