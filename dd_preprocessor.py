"""Helper functions to custom_preprocess_a.py. These functions perform the image preprocessing."""

import os  # Deals with path names
import dd_preprocess
from tqdm import tqdm  # For progress loading bar
import cv2  # For image preprocessing


# Common image file extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.ico', '.svg']
TRANSKRIBUS_IMAGE_EXTENSIONS = ['.pdf', '.jpg', '.png']

def process_images(treatment_map, sauvola_k_val, sauvola_window_size,
                   contrast_enhance):
    """
    Processes images based on their treatment type specified in treatment_map.
    :param treatment_map: [dict]
        Dictionary mapping image filepaths to their corresponding preprocessing treatments ('sbb' or 'sauvola').
    :param sauvola_k_val: [int]
        K-value when Sauvola binarisation is used (default: 0.24)
    :param sauvola_window_size: [int]
        Window size when Sauvola binarisation is used. Should not be an even value (default: 11)
    :param contrast_enhance: [bool]
        True if user wishes to contrast stretch and enhance contrast of images within pipeline. Default is False,
        determined through argparse.
    :return: [list]
        List of good quality image filepaths to pass to SBB binarisation pipeline.
    """

    sauvola_filepaths = [filepath for filepath, treatment in treatment_map.items() if treatment == 'sauvola']
    sbb_filepaths = [filepath for filepath, treatment in treatment_map.items() if treatment == 'sbb']

    # Process files with Sauvola pipeline
    process_sauvola(sauvola_filepaths, sauvola_k_val, sauvola_window_size, contrast_enhance)

    # Prepare files for preprocessing with SBB pipeline
    process_before_sbb(sbb_filepaths, contrast_enhance)

    # Return list of good quality image filepaths to pass to SBB binarisation pipeline.
    return sbb_filepaths


def process_sauvola(filepaths, sauvola_k_val, sauvola_window_size, contrast_enhance):
    """
    Function for processing images with Sauvola (non-machine learning) pipeline.

    :param filepaths: [list]
        List of filepaths to images to be processed with Sauvola pipeline.
    :param sauvola_k_val: [int]
        K-value when Sauvola binarisation is used (default: 0.24)
    :param sauvola_window_size: [int]
        Window size when Sauvola binarisation is used. Should not be an even value (default: 11)
    :param contrast_enhance: [bool]
        True if user wishes to contrast stretch and enhance contrast of images within pipeline. Default is False,
        determined through argparse.
    """
    print("Preprocessing bad quality images")
    file_count = len(filepaths)

    with tqdm(total=file_count, desc="Preprocessing bad quality images", unit="image") as pbar:
        for file in filepaths:
            try:
                # Check if the file is an image (any file with an image extension)
                if any(file.lower().endswith(image_ext) for image_ext in TRANSKRIBUS_IMAGE_EXTENSIONS):
                    pbar.set_description(f"Preprocessing image: {os.path.basename(file)}")

                    # Pre-process images to prepare them for OCR/HTR (we use destination_folder as the source
                    # folder as images in destination folder have already been partially prepared by
                    # meet_upload_reqs).
                    dd_preprocess.preprocess_image(src_image_path = file,
                                     dest_image_path = file,
                                     contrast_enhance = contrast_enhance,
                                     k_val = sauvola_k_val,
                                     window_size = sauvola_window_size)

                    # Update tqdm progress bar
                    pbar.update(1)
            except Exception as sauvola_processing_error:
                print(f"Error preprocessing image {file}: {sauvola_processing_error}")
                pbar.update(1)

    print("Preprocessing of bad quality images completed")


def process_before_sbb(filepaths, contrast_enhance=None):
    """
    Function for completing preprocessing of good quality images BEFORE SBB binarisation (machine learning).
    Includes: Greyscale, denoise - non-local means [(2.5) contrast enhance)]

    Pipeline: (1) Greyscale (2) denoise - non-local means [(2.5) contrast enhance)]
    *** (3) SBB *** (4) deskew - projection profiling (5) compression

    :param filepaths: [list]
        List of filepaths to images to be processed with SBB pipeline.
    :param contrast_enhance: [bool]
        True if user wishes to contrast stretch and enhance contrast of images within pipeline. Default is False,
        determined through argparse.
    """
    file_count = len(filepaths)

    if file_count == 0:
        print("No good quality images found to be binarised using SBB binarisation")
    else:
        print("Preparing good quality images for SBB binarisation")

    if file_count == 0:
        pass
    else:
        with tqdm(total=file_count, desc="Preprocessing images", unit="image") as pbar:
            for file in filepaths:
                try:
                    # Check if the file is an image (any file with an image extension)
                    if any(file.lower().endswith(image_ext) for image_ext in TRANSKRIBUS_IMAGE_EXTENSIONS):
                        pbar.set_description(f"Preparing image: {os.path.basename(file)}")

                        # Read the image
                        image = cv2.imread(file)
                        if image is None:
                            print(f"Error: Image not found or cannot be read: {os.path.basename(file)}")
                            pass

                        # Pre-process images to prepare them for OCR/HTR (we use destination_folder as the source
                        # folder as images in destination folder have already been partially prepared by
                        # meet_upload_reqs).

                        # Greyscale the image
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # Denoise image: Fast non-local means denoising (method for greyscale images):
                        image = cv2.fastNlMeansDenoising(image, None, h=10,
                                                         templateWindowSize=7,
                                                         searchWindowSize=21)

                        # Enhance contrast (optional): Normalisation (contrast stretching) +
                        # adaptive histogram equalization (CLAHE)
                        if contrast_enhance:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            image = clahe.apply(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))

                        # Write image to path
                        cv2.imwrite(file, image)

                        # Update tqdm progress bar
                        pbar.update(1)

                except Exception as sbb_processing_error:
                    print(f"Error preprocessing image {file}: {sbb_processing_error}")
                    pbar.update(1)

    if file_count == 0:
        pass
    else:
        print("Good quality images ready to be binarised with SBB pipeline")
