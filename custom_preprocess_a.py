"""
Image processing script which determines which one of two preprocessing pipelines to use according to an image's
quality.

To run this code, you need to download the sbb_binarization model from here:
https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2020_01_16.zip

Unzip the 'saved_model_2020_01_16' directory.

Move the entire directory into the dd_custom_preprocess folder.

N.B. pyiqa must be version 0.1.11 for this script to work. The maniqa-koniq quality scoring method is not available in
later versions.

"""
import dd_preprocess # image preprocessing pipeline (mostly contains code for non-machine learning approach)
import quality_scorer # scores images using pyiqa toolkit to determine whether images are good or bad quality.
import pyiqa
import dd_preprocessor # helper functions to perform the image preprocessing
import os
import argparse
from tqdm import tqdm
import pickle


# according to https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html
SCORING_METRIC = "maniqa-koniq"

# Common image file extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.ico', '.svg']

# The threshold used to determine whether an image is good or bad quality. Update as appropriate for the metric and
# source materials you are using.
DEFAULT_GOODBAD_THRESHOLD = 0.335 # maniqa-koniq

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image processing script which determines which one of two "
                                                 "preprocessing pipelines to use according to an image's quality.")
    parser.add_argument("source_folder", type=str, help="Path to the source folder")
    parser.add_argument("destination_folder", type=str, help="Path to the destination folder")
    parser.add_argument("--sauv_k_val", "-k", type=float, default=0.24,
                        help="K-value when Sauvola binarisation is used (default: 0.24)")
    parser.add_argument("--sauv_window_size", "-w", type=int, default=11,
                        help="Window size when Sauvola binarisation is used. Should not be an even value (default: 11)")
    parser.add_argument("--contrast_enhance", "-ce",
                        help="Use flag if you want to contrast stretch and enhance contrast of images within "
                             "pipeline. Default is not to use this.",
                        action="store_true")
    parser.add_argument("--regex", "-re", nargs='?', type=str,
                        help="Only preprocess image files which include this regex pattern in the name. "
                             "Remaining images will meet basic Transkribus upload requirements. If not specified, "
                             "will preprocess all images in same way. Do not wrap with r\" and \".")
    parser.add_argument("--goodbad_threshold", "-gb", type=float, default=DEFAULT_GOODBAD_THRESHOLD,
                        help=f"Image quality score to use as threshold between 'good' and 'bad' quality determination "
                             f"(default: {DEFAULT_GOODBAD_THRESHOLD})")

    args = parser.parse_args()

    filename_pattern = args.regex
    goodbad_threshold = args.goodbad_threshold

    shelvefilepath = os.path.join(args.source_folder, "image_scores")

    # (saved to source folder rather than destination folder because code iterates through dest folder again later
    # and .db file may complicate this)

    # Preprocess images using command-line arguments to meet Transkribus upload requirements
    print("Performing initial preprocessing")
    file_count = dd_preprocess.count_files_in_directory_tree(args.source_folder, IMAGE_EXTENSIONS)

    with tqdm(total=file_count, desc="Preprocessing images", unit="image") as pbar:
        # Iterate through the source folder and its sub-folders
        # root = current directory, dirs = subdirectories within current directory,
        # files = filenames in current directory
        for root, dirs, files in os.walk(args.source_folder):
            for file in files:
                try:
                    # Check if the file is an image (any file with an image extension)
                    if any(file.lower().endswith(image_ext) for image_ext in IMAGE_EXTENSIONS):

                        # Build the full path for the source and destination images.
                        # relpath is used to replicate source directory structure.
                        src_image_path = os.path.join(root, file)
                        dest_image_path = os.path.join(args.destination_folder,
                                                       os.path.relpath(src_image_path, args.source_folder))

                        # Create the destination folder if it doesn't exist
                        os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                        # exist_ok ensures function doesn't raise error if directory already exists

                        # Determine destination file path/name after conversion to jpg
                        dest_image_path = os.path.splitext(dest_image_path)[0] + '.jpg'

                        pbar.set_description(f"Preprocessing image: {os.path.basename(src_image_path)}")

                        dd_preprocess.meet_upload_reqs(src_image_path, dest_image_path, basic_only=False)

                        pbar.update(1)

                except Exception as meet_upload_reqs_error:
                    print(f"error preparing image: {file}", meet_upload_reqs_error)
                    pbar.update(1)

    # score to determine which preprocessing pipeline to use (non-ML, ML) - use pyiqa, maniqa-koniq
    quality_scorer.run_pyiqa_for_all_files(args.destination_folder,
                                           shelve_filepath=shelvefilepath,
                                           metric=SCORING_METRIC,
                                           filename_pattern=filename_pattern)

    lower_better = pyiqa.create_metric(metric_name=SCORING_METRIC, device=quality_scorer.DEVICE).lower_better

    # map image-wise scores to 'good'/'bad' quality class. If score is NA, assume image quality is 'bad'.
    good_bad_dict = quality_scorer.map_to_qualityclass(shelvefilepath, goodbad_threshold, lower_better)

    # map image-wise 'good'/'bad' quality classes to required preprocessing treatment. Keys: image filepaths, values:
    # 'sbb' if 'good', 'sauvola' if 'bad'.
    treatment_dict = quality_scorer.map_to_treatment(good_bad_dict)

    # Map scores to method required. Create 2 lists of filepaths - one of images to be processed w/ pipeline 1, one
    # of images to be processed w/ pipeline 2 (dd_preprocessor.process_images does this)

    # preprocessing of both pipelines' (SBB/ML and Sauvola/non-ML)

    # preprocesses poor quality images using non-ML pipeline
    # places images into new folder with same structure as had previously
    sbb_filepaths = dd_preprocessor.process_images(treatment_dict,
                                   sauvola_k_val=args.sauv_k_val,
                                   sauvola_window_size=args.sauv_window_size,
                                   contrast_enhance=args.contrast_enhance)

    # output list of good quality images to pass to next script custom_preprocess_b.py (different virtual environment
    # needed to use SBB binarisation code).

    with open('sbb_filepath_list.pkl', 'wb') as f:
        pickle.dump(sbb_filepaths, f)

    # custom_preprocess_b.py performs the following remaining steps:
    # - Preprocess good quality images using ML pipeline via OCRD's SBB-binarisation script.
    # - Complete further preprocessing steps.
    # - Place images into new folder with same structure as had previously



