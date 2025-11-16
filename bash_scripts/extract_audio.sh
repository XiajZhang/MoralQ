#!/bin/bash

# Source directory containing video files
SRC_DIR="/Users/mariyamohiuddin/Desktop/MIT Media Lab Projects/MoralQ Assets/Session 3/"
# Destination directory for audio files
DEST_DIR="/Users/mariyamohiuddin/Desktop/MIT Media Lab Projects/MoralQ Assets/Session 3 Audio/"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Processing videos from: $SRC_DIR"
echo "Saving audio to: $DEST_DIR"

cd "$SRC_DIR"

# Iterate through all MP4 files
for video in *.MP4; do
    if [ -f "$video" ]; then
        # Extract the first two parts before the first underscore
        prefix=$(echo "$video" | cut -d'_' -f1,2)
        # Create output filename
        audio_file="$DEST_DIR/${prefix}.wav"
        
        echo "Processing: $video"
        echo "Output: $audio_file"
        
        # Extract audio using ffmpeg
        ffmpeg -i "$video" -vn "$audio_file"
    fi
done

echo "Processing complete!"
