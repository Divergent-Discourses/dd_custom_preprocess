"""
Continuation of custom_preprocess_a.py driver script, resulting in 2 separate scripts requiring 2 separate environments
(custom_preprocess_a and custom_preprocess_b).

Requires different virtual environment to use OCRD's SBB binarisation script:
https://github.com/qurator-spk/sbb_binarization

Preprocesses good quality images using machine learning (SBB binarisation) pipeline ('good' quality according to
maniqa-koniq image quality assessment score).

Completes further preprocessing steps (deskew, compression).

Places images into new folder with same structure as had previously.

# N.B. cannot install tensorflow-macos and tensorflow-metal for Mac M-series users to use GPU - causes compatibility
errors between Keras 3 (upgraded) and legacy model files for binarisation

# N.B. SBB binarisation outputs images in .tif or .png format. We output .png images to interface with Transkribus
which takes images in .jpeg or .png format currently (September 2024).

"""
import pickle
from sbb_binarize.sbb_binarize import SbbBinarizer
import os
import dd_preprocess
import cv2

# Define your model directory, input image, and output image paths
model_dir = "./saved_model_2020_01_16"

# load list of good quality image filepaths - these should be preprocessed using sbb_binarisation
with open('sbb_filepath_list.pkl', 'rb') as f:
    sbb_filepaths = pickle.load(f)

# Instantiate the SbbBinarizer and run the binarization
if len(sbb_filepaths) == 0:
    pass
else:
    print("Preparing to binarise")
    binarizer = SbbBinarizer(model_dir)

    # Run the binarizer on each object in the filepath list

    new_output_filepaths = [] # filepaths with new .png extensions (SBB binarisation only outputs .tif or .png)

    for input_image_path in sbb_filepaths:
        output_image_path = input_image_path.replace(".jpg", ".png")
        new_output_filepaths.append(output_image_path)
        print(f"Binarising image: {os.path.basename(input_image_path)}")
        binarizer.run(image_path=input_image_path, save=output_image_path)

        # delete input image (has now been replaced with binarised image)
        os.remove(input_image_path)

        print("Final preprocessing steps - rotate, compress")
        image = cv2.imread(output_image_path)
        dd_preprocess.rotate_image(image, output_image_path)

        # If image is already smaller than target size, return the image
        img_size_bytes = os.path.getsize(output_image_path)
        if img_size_bytes <= dd_preprocess.max_image_bytes:
            pass
        else:
            # If image is larger than max allowed size, compress until allowable size
            dd_preprocess.compress_under_size(dd_preprocess.bytes_in_mb, output_image_path)

if len(sbb_filepaths) == 0:
    pass
else:
    print("Finished binarising good quality images")







