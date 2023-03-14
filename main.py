########################################
#   Title: Voice Synth Training Data Prepper
#   Author: Tyler Wolf
#   Date: 11/11/2022
#   
#   Description:
#       Creating datasets for my machine learning project was extremely
#       tedious and error prone due to the large amount of files needed.
#       This script was made to automate the audio chunking, transcribing,
#       training data subset creation, and packaging process.
#
#   Use:
#       All you need to do is place all your source audio files in the "Recordings"
#       directory and run the script. Be sure to use as clean of audio as possible to
#       minimize the likelyhood of unusable chunks or transcriptions.
#
#   Notes:
#       Need to add logging and exception handling so the user knows where they are
#       in terms of script progress.
#       - Fix subset creation method, it's lacking.
#       - Add method for normalizing transcription (numbers, symbols, etc.)
#
#   My Project Stats:
#       After multi-thread/multi-process changes:
#           Script Runtime:                 26min 27sec
#           Total Source Audio Files:       18
#           Combined Source Audio Length:   13hr 10min 48sec
#           Total Useable Chunks Created:   12923
#           Combined Chunk Audio Length:    11hr 1min 12sec
#           Total Silence Cut:              2hr 7min 48sec
#           Total Chunks Lost:              211
#           Total Audio Duration Lost:      1min 44sec (Lost Chunks)
#       Before changes:
#           Script Runtime:                 2hr 47min
#           Total Source Audio Files:       18
#           Combined Source Audio Length:   13hr 10min 48sec
#           Total Useable Chunks Created:   12,909
#           Combined Chunk Audio Length:    11hr 0min 36sec
#           Total Silence Cut:              128.26 mins 2hr 7min 59sec
#           Total Chunks Lost:              225 chunks
#           Total Audio Duration Lost:      2min 32sec  (Lost Chunks)
#   
########################################

import os
import csv
import tarfile
import speech_recognition as sr
import pydub
import concurrent.futures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help = "Set file prefix for output")
args = parser.parse_args()

AUDIO_DIR = "./recordings/"
CHUNKS_DIR = "./output/chunks/"
TRANS_DIR = "./output/transcripts/"
FILELIST_DIR = './output/filelists/'

def verify_dirs():

    if not os.path.exists(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)
    if not os.path.exists(AUDIO_DIR):
        os.mkdir(AUDIO_DIR)
    if not os.path.exists(TRANS_DIR):
        os.mkdir(TRANS_DIR)
    if not os.path.exists(FILELIST_DIR):
        os.mkdir(FILELIST_DIR)

def transcribe_chunk(chunk):
    """
        Receives audio chunks and outputs transcriptions into CSV
    """

    path = CHUNKS_DIR + chunk
    r = sr.Recognizer()

    with sr.AudioFile(path) as source:
        audio_listened = r.listen(source)

        try:
            # total_audio_len = total_audio_len + pydub.AudioSegment.from_file(path).duration_seconds
            return f'{os.path.splitext(chunk)[0]}|{r.recognize_google(audio_listened)}'
        except Exception as e:
            print(e)
            os.remove(path)
            return None


def write_subset(file, data, x, y):
    temp = data[x:x+y]
    with open(file, 'w') as f:
            for line in temp:
                f.write(f"JPSpeech-1.0/wavs/{line[:10]}.wav{line[10:]}\n")

def create_training_subsets():
    """
        Create multiple testing, training, and validation subsets of gathered
        transcriptions for mel and audio training
    """
    data = []
    with open(f'{TRANS_DIR}/metadata.csv', 'r') as file:
        data = file.read().splitlines()
    data_length = len(data)
    
    write_subset('./output/filelists/jps_audio_text_test_filelist.txt', data, 0, 500)
    write_subset('./output/filelists/jps_audio_text_train_filelist.txt', data, 500, data_length)
    write_subset('./output/filelists/jps_audio_text_val_filelist.txt', data, 600, 100)
    # write_subset('./output/filelists/jps_audio_text_train_subset_64_filelist.txt', data, 700, 64)
    # write_subset('./output/filelists/jps_audio_text_train_subset_300_filelist.txt', data, 764, 300)
    # write_subset('./output/filelists/jps_audio_text_train_subset_625_filelist.txt', data, 1064, 625)
    # write_subset('./output/filelists/jps_audio_text_train_subset_1250_filelist.txt', data, 1689, 1250)
    # write_subset('./output/filelists/jps_audio_text_train_subset_2500_filelist.txt', data, 2939, 2500)
    write_subset('./output/filelists/jps_mel_text_filelist.txt', data, 0, data_length)
    write_subset('./output/filelists/jps_mel_text_test_filelist.txt', data, 0, 500)
    write_subset('./output/filelists/jps_mel_text_train_filelist.txt', data, 500, data_length)
    write_subset('./output/filelists/jps_mel_text_val_filelist.txt', data, 600, 100)
    # write_subset('./output/filelists/jps_mel_text_train_subset_64_filelist.txt', data, 700, 64)
    # write_subset('./output/filelists/jps_mel_text_train_subset_300_filelist.txt', data, 764, 300)
    # write_subset('./output/filelists/jps_mel_text_train_subset_625_filelist.txt', data, 1064, 625)
    # write_subset('./output/filelists/jps_mel_text_train_subset_1250_filelist.txt', data, 1689, 1250)
    # write_subset('./output/filelists/jps_mel_text_train_subset_2500_filelist.txt', data, 2939, 2500)

def package_data():
    """
        Place all output data into an organized compressed tar file
    """
    with tarfile.open("./output/JPSpeech-1.0.tar.bz2", "w:bz2") as tar:
        tar.add(CHUNKS_DIR, arcname=os.path.basename("./wavs"))
        tar.add(TRANS_DIR, arcname=os.path.basename("/"))
        tar.add(FILELIST_DIR, arcname=os.path.basename("./filelists"))
        tar.close()

def chunk_audio(audio_file, file_num, prefix):
    """
        Split all source audio into small chunks base don silence and export in proper format
    """
    audio = pydub.AudioSegment.from_file(audio_file)

    chunks = pydub.silence.split_on_silence(audio, min_silence_len=375, silence_thresh=-40)
    
    for i, chunk in enumerate(chunks):
        final = chunk + pydub.AudioSegment.silent(duration=200)
        final.export(f"{CHUNKS_DIR+prefix}"+f"{file_num}".zfill(3)+"-"+f"{i+1}".zfill(4)+".wav", format="wav")

def convert_audio_files(file):
    audio = pydub.AudioSegment.from_file(CHUNKS_DIR + file, format="wav")
    audio = audio.set_frame_rate(22050)
    audio = audio.set_channels(1)

    audio.export(CHUNKS_DIR + file, format="wav")

def create_expanded_filelist(src):
    """
        Joins the necessary path to the file names in a given directory to be used with other functions
    """
    init_list = os.listdir(src)
    final_list = []

    for filename in init_list:
        final_list.append(os.path.join(src,filename))
    
    return final_list

def main():
    global args
    prefix = args.prefix

    # Verify necessary DIRs exist or make them
    verify_dirs()

    # Chunk our source audio by silence and export to output DIR
    src_aud = create_expanded_filelist(AUDIO_DIR)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, audio_file in enumerate(src_aud):
            executor.submit(chunk_audio, audio_file, i, prefix)

    # Transcribe chunked audio and output to metadata.csv
    chunk_aud = os.listdir(CHUNKS_DIR)

    with open(TRANS_DIR+'metadata.csv', 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for transcription in executor.map(transcribe_chunk, chunk_aud):
                if transcription:
                    writer.writerow([transcription])

    chunk_aud = os.listdir(CHUNKS_DIR)

    # Convert sample rate and stereo->mono
    for file in chunk_aud:
        convert_audio_files(file)

    # Create training subset files
    create_training_subsets()

    # Package complete dataset to tar.bz2 file for portability
    package_data()

if __name__=="__main__":
    main()