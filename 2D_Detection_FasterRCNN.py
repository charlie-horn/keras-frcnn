import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import subprocess

tf.enable_eager_execution()

from waymo_od.waymo_open_dataset.utils import range_image_utils
from waymo_od.waymo_open_dataset.utils import transform_utils
from waymo_od.waymo_open_dataset.utils import  frame_utils
from waymo_od.waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gcloud import storage

def get_file_names(data_directory):
    list_command = "gsutil ls " + data_directory
    file_names_string_output = subprocess.check_output(list_command, shell=True).decode("utf-8")
    file_names_list = file_names_string_output.split("\n")[:-1]
    return file_names_list

def main(mode="train"):
    """ Main function """

    # Build network

    # Get input data
    data_directories = {
        "train" : "gs://waymo_open_dataset_v_1_2_0_individual_files/training/",
        "test" : "gs://waymo_open_dataset_v_1_2_0_individual_files/testing/",
        "validate" : "gs://waymo_open_dataset_v_1_2_0_individual_files/validation/"
    }
    data_directory = data_directories[mode]
    remote_file_names = get_file_names(data_directory)
    for remote_file in remote_file_names:
        # Download File
        file_name = os.path.basename(remote_file)
        local_path = "/content/input_files/" + file_name
        cp_command = "gsutil cp " + remote_file + " " + local_path
        subprocess.call(cp_command, shell=True)
        # Create Frame Objects
        dataset = tf.data.TFRecordDataset(local_path, compression_type='')
        input_data = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            print(frame.camera_labels)
            for image in frame.images:
                print(image)
                break
            break
        # print(frame.context)
        # Define mini batches

        # Loop through frames
        #for frame_batch in frame_batches:
            # Generate outputs

            # Train network

        # Remove Frames
        subprocess.call("rm " + local_path, shell=True)
        break

if __name__ == "__main__":
    main()
