import os
import wave
import librosa
import logging

# Set up logging to write errors to a log file
logging.basicConfig(filename='wav_sanity_check_log.txt', level=logging.WARNING)
logger = logging.getLogger()

# Define the directory where the WAV files are located
wav_directory = '../fma_small'  # Update this to the correct path

# Expected sample rate (update if needed)
expected_sample_rate = 22050  # or another value like 44100

# Function to check if a WAV file is valid
def check_wav_file(wav_file):
    try:
        # Try to open the file using wave module
        with wave.open(wav_file, 'rb') as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            frame_rate = wav.getframerate()
            
            # Check for non-zero length
            if n_frames == 0:
                logger.warning(f"Empty file: {wav_file}")
                return False
            
            # Check for valid sample width (16-bit PCM format)
            if sample_width != 2:
                logger.warning(f"Invalid sample width in {wav_file}: {sample_width}")
                return False
            
            # Check for expected sample rate
            if frame_rate != expected_sample_rate:
                logger.warning(f"Unexpected sample rate in {wav_file}: {frame_rate}")
                return False
            
            # Check for number of channels (mono/stereo)
            if n_channels not in [1, 2]:
                logger.warning(f"Unexpected number of channels in {wav_file}: {n_channels}")
                return False
            
            return True
        
    except Exception as e:
        logger.warning(f"Error processing file {wav_file}: {str(e)}")
        return False

# Walk through all files in the directory and perform the sanity check
corrupted_files = []
for root, dirs, files in os.walk(wav_directory):
    for file in files:
        if file.endswith('.wav'):  # Only process WAV files
            wav_file = os.path.join(root, file)
            
            if not check_wav_file(wav_file):
                corrupted_files.append(wav_file)

# Print the results
if corrupted_files:
    print("The following WAV files failed the sanity check:")
    for file in corrupted_files:
        print(file)
else:
    print("All WAV files passed the sanity check!")

