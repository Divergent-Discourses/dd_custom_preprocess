# Image Preprocessing Pipeline

Preprocesses images of written documents to prepare them for optical character recognition (OCR) or handwritten text recognition (HTR). Applies one of two pipelines to an image depending on its quality. 

Aims to obtain more accurate transcriptions by making text more machine-readable. 

Originally written for OCR/HTR of historical documents, predominantly Tibetan-language newspaper pages from 1950-60s.

## About

#### Step 1: Prepares images to meet Transkribus upload requirements

- Converts all images to JPEG format
- Ensures each image is maximum 300 DPI
- Increases size of images so that at least one dimension is 2500 pixels
- Ensures each image is maximum 10 MB

#### Step 2: Preprocesses images to be more accurately recognised by an optical character recognition (OCR) model

Pipeline originally designed to preprocess and improve readability of old scans/microfiche images of historical newspaper articles.

Generates a quality score for each image using **maniqa-koniq** image quality assessment (IQA). Uses this score as threshold to decide whether image is bad or good quality.


If image is 'bad quality':

- Converts images to greyscale
- Applies fast non-local means denoising
- [optional: use -ce flag] Performs contrast stretching and the adaptive histogram equalisation (CLAHE) contrast enhancement method
- Performs **Sauvola binarisation** (local thresholding approach)
- Deskews images using projection profiling


If image is 'good quality':

- Converts images to greyscale
- Applies fast non-local means denoising
- [optional: use -ce flag] Performs contrast stretching and the adaptive histogram equalisation (CLAHE) contrast enhancement method
- Performs **SBB binarisation** (machine learning approach): https://github.com/qurator-spk/sbb_binarization/
- Deskews images using projection profiling


## Installation

Using the command line, navigate to the location in which you wish to install the code. Then, download the code.

```bash
git clone https://github.com/Divergent-Discourses/dd_custom_preprocess.git
```

Download the sbb_binarization model from here: https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2020_01_16.zip

Unzip the 'saved_model_2020_01_16' directory.

**Move the entire directory into the dd_custom_preprocess folder.**


Using the command line, navigate to the location of this repository.

```bash
cd dd_custom_preprocess
```

Running the bash driver script for the first time will install required packages (into two separate conda environments: custom_preprocess_a and custom_preprocess_b).


## Usage

Place all images you want to preprocess in a directory. The directory can contain sub-directories if you want to keep image sub-groups.


Run the script from the command line once you have navigated to the location of the dd_custom_preprocess directory like this:

```bash
bash preprocess_driver.sh path/to/source/directory path/to/destination/directory
```

- **Source directory path:** The path to the folder which stores the images you want to preprocess

- **Destination directory path:** The path to the folder which will store the preprocessed images - this doesn't have to exist yet. It just needs to include the desired path to/name for the folder


After you've typed that (before pressing enter), you can optionally include the following flags:

- **--k_val /  -k [float] :** Modify the K-value used during Sauvola binarisation (default: 0.24)

- **--window_size / -w [int] :** Modify the window size used during Sauvola binarisation. Should not be an even value (default: 11)

- **--contrast_enhance / -ce :** Use flag if you want to contrast stretch and enhance contrast of images within pipeline. Default is not to use this as it tends to introduce speckling.

- **--regex / -r [str] :** Only preprocess image files which include this regex pattern in the name. Remaining images will meet basic Transkribus upload requirements. If not specified, will preprocess all images. Do not wrap with r\" and \".

- **--goodbad_threshold / -gb [float] :** Image quality score to use as threshold between 'good' and 'bad' quality determination (default: 0.335)


For best results, you will need to tune **k_val** and **window_size** to values which work best for your 'bad quality' materials. Default values were found to work the best on relatively noisy images with black text, some staining and bleedthrough.


N.B. The code saves a shelf file, **image_scores.db**, to the source directory folder. This holds all image quality scores for future reference (key=filepath, value=quality score). You can delete this after you've run the code if you don't need it.


#### For example...

You could use this command to adjust the k-value and window size used during Sauvola binarisation to alter the quality of images outputted:

```bash
bash preprocess_driver.sh path/to/source/directory path/to/destination/directory --k_val 0.22 --window_size 301
```

You could use this command to only preprocess images with '_archive_' in the filename, while ensuring all remaining images meet basic Transkribus image upload requirements
(e.g. file size, image format):

```bash
bash preprocess_driver.sh path/to/source/directory path/to/destination/directory --regex _archive_
```


## Why this approach?

Our initial tests showed that some images were transcribed most accurately by our OCR model when preprocessed in one way and others when preprocessed in the other way.

In general, our OCR model most accurately transcribed heavily stained, noisy images featuring dark patches when preprocessed with our own dd_preprocess pipeline. While dd_preprocess introduces more noise/speckling than sbb_binarization, it results in little less information loss.

We use IQA quality scoring to roughly classify images into 'poor quality' and 'good quality' buckets so we can decide which preprocessing method is likely to result in the most accurate possible OCR prediction for each image. These quality ratings act as a rough proxy for the class labels we really want, the preprocessing pipeline that will likely result in the most accurate OCR prediction for a given image (pipeline using sbb_binarization for 'good quality' images, or the pipeline using Sauvola binarisation for 'bad quality' images). The proxy classifications are not perfect but generally work for our sources.

For your own images, you should **tune the score threshold** using the --goodbad_threshold flag. We use the threshold to decide whether an image is good or bad quality based on its score and average quality scores differ between image sets.


## Acknowledgements

### MANIQA (the image quality assessment method used)

We used MANIQA via the PYIQA module (https://github.com/chaofengc/IQA-PyTorch).

Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J. and Yang, Y., 2022. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1191-1200).
[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)](https://arxiv.org/abs/2204.08958)
[![code](https://img.shields.io/badge/code-github-red.svg)](https://github.com/IIGROUP/MANIQA)

### SBB_BINARIZATION (machine learning binarisation method used)
Qurator Project. 2023. Document image binarisation.
[![code](https://img.shields.io/badge/code-github-red.svg)](https://github.com/qurator-spk/sbb_binarization/)


## Copyright

**dd_custom_preprocess** was developed by Christina Sabbagh of SOAS University of London for the Divergent Discourses project. The project is a joint study involving SOAS University of London and Leipzig University, funded by the AHRC in the UK and the DFG in Germany.

Please acknowledge the project in any use of these materials. Copyright for the project resides with the two universities.
