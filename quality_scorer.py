"""
Uses the Pyiqa toolkit (and specifically maniqa-koniq which can be changed) to give an image a quality score
(good or bad).
"""

import pyiqa
import os
from tqdm import tqdm
import shelve
import torch
import custom_preprocess_a
import re

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("using cuda gpu")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("mps gpu used")
else:
    DEVICE = torch.device("cpu")
    print("mps is not available, using CPU instead")

def run_pyiqa_for_all_files(img_directory_path, shelve_filepath, metric="maniqa-koniq", filename_pattern=False):
    """
    Takes path to a directory containing images to score. Returns a shelf object whereby keys are individual image
    filepaths and values are quality scores according to the selected PYIQA metric.
    :param img_directory_path: [string] path to directory containing images to score
    :param shelve_filepath: [string] path to shelve file
    :param metric: [string] For available metrics, see https://github.com/chaofengc/IQA-PyTorch and
    https://iqa-pytorch.readthedocs.io/
    :param filename_pattern: [string] Preprocess only image files which include this regex pattern in the name.
    """

    if os.path.exists(shelve_filepath):
        # remove file created during previous runs if exists.
        os.remove(shelve_filepath)

    shelve_file = shelve.open(shelve_filepath, writeback=True)

    # Iterate through all files in the given directory:
    file_count = 0
    for root, dirs, filenames in os.walk(img_directory_path):
        for filename in filenames:
            if filename.lower().endswith(tuple(custom_preprocess_a.IMAGE_EXTENSIONS)):
                if filename_pattern:
                    if re.search(filename_pattern, filename):
                        file_count += 1
                else:
                    file_count += 1

    # for item in progress_bar:
    with tqdm(total=file_count, desc="Quality-scoring images", unit="file") as pbar:

        for root, dirs, filenames in os.walk(img_directory_path):

            for filename in filenames:

                file_path = os.path.join(root, filename)

                if file_path == os.path.join(root, '.DS_Store'):
                    pass

                # If there is a filename regex pattern to identify specific image files to preprocess (and leave others
                # preprocessed just to meet basic Transkribus upload requirements), preprocess as follows. Otherwise, leave images.
                elif filename_pattern:
                    # Check if pattern is in the filename:
                    if re.search(filename_pattern, filename):

                        pbar.set_description(f"Scoring {filename}")

                        try:
                            # create metric with default setting
                            iqa_metric = pyiqa.create_metric(metric_name=metric, device=DEVICE)

                            # img path as inputs.
                            score_nr = float(iqa_metric(file_path))
                            print(f"score for {filename} is: {score_nr}")

                            # Update shelve object so we save the scores as we go along in case of midway errors
                            shelve_file[file_path] = score_nr
                            shelve_file.sync()

                            # Update tqdm progress bar
                            pbar.update(1)

                        except Exception as scoring_shelving_error:
                            print(f"Error: Something went wrong running metric {metric}", scoring_shelving_error)

                            # Update shelve object so we save the scores as we go along in case of midway errors
                            shelve_file[file_path] = "NA"
                            shelve_file.sync()

                            # Update tqdm progress bar
                            pbar.update(1)
                else:

                    pbar.set_description(f"Scoring {filename}")

                    try:
                        # create metric with default setting
                        iqa_metric = pyiqa.create_metric(metric_name=metric, device=DEVICE)

                        # img path as inputs.
                        score_nr = float(iqa_metric(file_path))
                        print(f"score for {filename} is: {score_nr}")

                        # Update shelve object so we save the scores as we go along in case of midway errors
                        shelve_file[file_path] = score_nr
                        shelve_file.sync()

                        # Update tqdm progress bar
                        pbar.update(1)

                    except Exception as scoring_shelving_error:
                        print(f"Error: Something went wrong running metric {metric}", scoring_shelving_error)

                        # Update shelve object so we save the scores as we go along in case of midway errors
                        shelve_file[file_path] = "NA"
                        shelve_file.sync()

                        # Update tqdm progress bar
                        pbar.update(1)

    shelve_file.close()

def map_to_qualityclass(shelvefilepath, goodbad_threshold, lower_better=False):
    """
    Takes a shelve file whereby keys are filepaths to images and values are quality scores. Returns a dictionary
    whereby keys are filepaths to images and a given values is one of two classes - 'good' or 'bad' depending on
    whether the metric used for scoring returns a higher or lower score when it deems an image to be better quality,
    and what threshold to use to decide between 'good' and 'bad' classes.
    :param shelvefilepath: [python shelve obj] Path to stored shelve file object.
    :param goodbad_threshold: [float] Threshold used as score boundary between 'good' and 'bad' classes.
    :param lower_better: [bool] True when lower score indicates better quality image for scoring metric being used.
    :return: [dict] Dictionary mapping 'good' or 'bad' classification to each image filepath based on quality score.
    """
    with shelve.open(shelvefilepath) as db:
        goodbad_dict = {}
        for filepath, score in db.items():
            if not lower_better: # i.e. if higher score is better
                if score == "NA":
                    goodbad_dict[filepath] = "bad" # take bad quality as default as, in general, the non-machine
                    # learning approach results in better OCR accuracy for images from this source (dark, microfiche),
                    # and we will use the non-machine learning approach for bad quality images

                elif score >= goodbad_threshold:
                    goodbad_dict[filepath] = "good"

                else:
                    goodbad_dict[filepath] = "bad"

            elif lower_better:
                if score == "NA":
                    goodbad_dict[filepath] = "bad"

                elif score <= goodbad_threshold:
                    goodbad_dict[filepath] = "good"

                else:
                    goodbad_dict[filepath] = "bad"

        db.close()

    print(f"\nImage quality assessment: {goodbad_dict}\n")
    return goodbad_dict


def map_to_treatment(score_dict):
    """
    Takes a dictionary whereby keys are filepaths to images and a given value is a 'good' or 'bad' quality class for
    that image. Returns a dictionary whereby keys are filepaths to treatment images and value is the required treatment
    for that image based on the 'good'/'bad' quality class. Maps 'good' quality images to the preprocessing pipeline
    using the SBB (machine learning) binarisation approach. Maps 'bad' quality images to the preprocessing pipeline
    using Sauvola (non-machine learning) binarisation.
    :param score_dict: [dict] Dictionary mapping 'good' or 'bad' classification to each image filepath.
    :return: treatment_dict: [dict] Dictionary mapping one of two preprocessing treatments to each image filepath based
    on quality class.
    """

    treatment_map = {'good': 'sbb', 'bad': 'sauvola'}
    return {filepath: treatment_map[quality] for filepath, quality in score_dict.items()}