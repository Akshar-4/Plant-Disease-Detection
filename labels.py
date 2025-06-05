import os

# This should point to the path where your training images are stored
dataset_path = "dataset"
class_labels = sorted(os.listdir(dataset_path))
print(class_labels)