#!/bin/bash

Define your base paths
BASE_PATH_BLUR="/home/zy3724/git_repos/flatten/video_samples/talkman/pis_enhanced"
BASE_PATH_BLEND="/home/zy3724/git_repos/flatten/video_samples/talkman/mag_enhanced"
RESULTS_PATH="/home/zy3724/git_repos/flatten/video_samples/talkman/output"
PROMPT_INPUT="a side view of real talking man face, open mouth and show teech a little bit, high quality, 4K"
IDX=60




# Number of images to process
N=100  # Change this value to process first N images

# Initialize a counter
counter=0

# Create the results folder if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Loop through all images in BASE_PATH_BLUR
for image_file in "$BASE_PATH_BLUR"/*.png; do
  # Increment the counter
  ((counter++))

  # Break the loop if counter exceeds N
  if [ "$counter" -gt "$N" ]; then
    break
  fi

  # Extract the base filename without the path and generate the new padded filename
  # Example: "000008_512.png" -> "00000008.png"
  filename=$(basename "$image_file")
  filename_number=$(printf "%08d" "$counter")  # Pad the counter with leading zeros
  new_filename="${filename_number}.png"  # New filename in the desired format

  # Set the blur file path
  FILE_PATH_BLUR="$image_file"

  # Set the blend file path, check if the enhanced version exists
  enhanced_file="$BASE_PATH_BLEND/${filename%.png}.png"
  if [ -f "$enhanced_file" ]; then
    FILE_PATH_BLEND="$enhanced_file"
  else
    # If no corresponding blend file exists, skip the file and output a message
    echo "Jumping: No corresponding blend file for $filename"
    continue
  fi

  # Set the save path with the new formatted filename
  SAVE_PATH="$RESULTS_PATH/${filename_number}_inversion.png"

  # Output the index and file paths
  echo "Processing index: $counter"
  echo "Blur file path: $FILE_PATH_BLUR"
  echo "Blend file path: $FILE_PATH_BLEND"
  echo "Save path: $SAVE_PATH"

  # Call the Python script with the defined parameters
  python avideo_script4bash.py --file_path_blur "$FILE_PATH_BLUR" --file_path_blend "$FILE_PATH_BLEND" --prompt_input "$PROMPT_INPUT" --idx "$IDX" --save_path "$SAVE_PATH"

  # Print the operation details
  echo "Processed $new_filename: Saved to $SAVE_PATH"
done
