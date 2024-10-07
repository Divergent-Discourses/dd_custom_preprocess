#!/bin/bash

# bash script takes same arguments as custom_preprocess_a.py and supplies them
# to the python script

usage() {
    echo "Usage: $0 [options] <source_folder> <destination_folder>"
    echo ""
    echo "Options:"
    echo "  -k, --sauv_k_val             K-value for Sauvola binarization (default: 0.24)"
    echo "  -w, --sauv_window_size       Window size for Sauvola binarization (default: 11)"
    echo "  -ce, --contrast_enhance      Enable contrast enhancement (flag)"
    echo "  -re, --regex                 Regex pattern used to select which image files to preprocess (default: False)"
    echo "  -gb, --goodbad_threshold     Image quality score to use as threshold between 'good' and 'bad' quality determination (default: 0.335)"
    echo ""
    echo "Example: $0 source_folder destination_folder -k 0.5 -w 15 --contrast_enhance"
    exit 1
  }

# Set default values for command line arguments
  sauv_k_val=0.24
  sauv_window_size=11
  countrast_enhance=''
  goodbad_threshold=0.335

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -k|--sauv_k_val)
            sauv_k_val="$2"
            shift 2
            ;;
        -w|--sauv_window_size)
            sauv_window_size="$2"
            shift 2
            ;;
        -ce|--contrast_enhance)
            contrast_enhance_flag="--contrast_enhance"
            shift 1
            ;;
        -re|--regex)
            regex="$2"
            shift 2
            ;;
        -gb|--goodbad_threshold)
            goodbad_threshold="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$source_folder" ]]; then
                source_folder="$1"
            elif [[ -z "$destination_folder" ]]; then
                destination_folder="$1"
            else
                echo "Unknown argument: $1"
                usage
            fi
            shift 1
            ;;
    esac
done


# Validate required arguments
if [[ -z "$source_folder" || -z "$destination_folder" ]]; then
    echo "Error: Source folder and destination folder are required."
    usage
fi

# Check if required environments already exist. If not, create them and
# install required packages

# Function to check if conda environment exists
function check_env() {
    conda env list | grep -q "^$1\s"
}


# Set up environment for custom_preprocess_a
if ! check_env "custom_preprocess_a"; then
    echo "Creating environment 'custom_preprocess_a'..."
    conda create -n custom_preprocess_a "python==3.10.9"
    source ~/.zshrc
    conda activate custom_preprocess_a
    pip install -r requirements_a.txt
    source ~/.zshrc
    conda deactivate
else
    echo "Environment 'custom_preprocess_a' already exists. Skipping creation."
fi

# Set up environment for custom_preprocess_b
if ! check_env "custom_preprocess_b"; then
    echo "Creating environment 'custom_preprocess_b'..."
    conda create -n custom_preprocess_b "python<=3.10"
    source ~/.zshrc
    conda activate custom_preprocess_b
    pip install -r requirements_b.txt
    conda install "tensorflow<=2.11.1"
    conda install "readline==8.2"
    source ~/.zshrc
    conda deactivate
else
    echo "Environment 'custom_preprocess_b' already exists. Skipping creation."
fi

source ~/.zshrc
conda activate custom_preprocess_a
python custom_preprocess_a.py "$source_folder" "$destination_folder" \
    --sauv_k_val "$sauv_k_val" \
    --sauv_window_size "$sauv_window_size" \
    $contrast_enhance_flag \
    --regex "$regex" \
    --goodbad_threshold "$goodbad_threshold"
source ~/.zshrc
conda deactivate

source ~/.zshrc
conda activate custom_preprocess_b
python custom_preprocess_b.py
source ~/.zshrc
conda deactivate
