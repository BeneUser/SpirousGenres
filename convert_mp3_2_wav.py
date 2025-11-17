# convert all mp3 files into wav files
# log which files are not converted


import os
import librosa
import soundfile as sf
import warnings


# Ignore warnings and treat them as errors
warnings.filterwarnings("error")

# Define the directory where the MP3 files are located
input_directory = '../fma_small'  # Path to your FMA_small dataset

# Define the output log file for corrupted files
log_file = 'corrupted_files.txt'

# Open the log file in append mode (this allows us to add to the file without overwriting)
with open(log_file, 'a') as log:

    # Write a header to the log file, indicating the start of the log for this run
    log.write("Corrupted files log:\n")
    log.write("="*40 + "\n")

    # Walk through all the directories and files in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.mp3'):  # Only process MP3 files
                mp3_file = os.path.join(root, file)

                # Check if the file is too small (e.g., less than 10 KB)
                if os.path.getsize(mp3_file) < 10000:
                    # Log the corrupted file and skip it
                    log.write(f"Skipping {mp3_file}, file is too small or corrupted.\n")
                    print(f"Skipping {mp3_file}, file is too small or corrupted.")
                    continue

                try:
                    # Try loading the MP3 file
                    y, sr = librosa.load(mp3_file, sr=22050)  # Load with a sample rate of 22,050 Hz
                    
                    # If the file is successfully loaded, process and save as WAV
                    wav_file = os.path.join(root, file.replace('.mp3', '.wav'))
                    sf.write(wav_file, y, sr)
                    print(f"Converted {mp3_file} to {wav_file}")
                
                except Exception as e:
                    # If there's an error (e.g., file is corrupted or unreadable), log it
                    log.write(f"Error converting {mp3_file}: {str(e)}\n")
                    log.flush()  # Ensure immediate writing to log
                    print(f"Error converting {mp3_file}: {str(e)}")
                log.flush()    

    # End of log file
    log.write("="*40 + "\n")

