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
#       - Also see about optimizing chunking and transription process (try multi-threading). 
#       - Add method for normalizing transcription (numbers, symbols, etc.)
#       - Change audio chunk output to 22050 sample rate
#
#   My Project Stats:
#       Script Runtime:                 2hr 47min
#       Total Source Audio Files:       18
#       Combined Source Audio Length:   13.18 hours
#       Total Useable Chunks Created:   12,909
#       Combined Chunk Audio Length:    11.01 hours
#       Total Silence Cut:              128.26 mins
#       Total Chunks Lost:              225 chunks
#       Total Audio Duration Lost:      2.54 mins (Lost Chunks)
#   
########################################

import os
import csv
import tarfile
import fnmatch
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

AUDIO_DIR = "./recordings/"
CHUNKS_DIR = "./output/chunks/"
TRANS_DIR = "./output/transcripts/"
FILELIST_DIR = './output/filelists/'
MEL_DIR = "./output/mel/"

def chunk_audio():
    """
        Chunks audio by pauses in speech pattern and outputs @@###-####.wav files
    """
    
    if not os.path.exists(AUDIO_DIR):
        os.mkdir(AUDIO_DIR)

    recordings = sorted(os.listdir(AUDIO_DIR))

    for i, filename in enumerate(recordings):
        print(f"Creating chunks from source file {i}....")
        chunks = []
        f = os.path.join(AUDIO_DIR, filename)
        source_aud = AudioSegment.from_file(f)
        
        chunks = chunks + split_on_silence(source_aud, min_silence_len=375, silence_thresh=-40)
        for j, chunk in enumerate(chunks):
            chunk_name = "JP" + f"{i+1}".zfill(3) + "-" + f"{j+1}".zfill(4) + ".wav"
            chunk.export(CHUNKS_DIR + chunk_name, format="wav")

def transcribe_chunks():
    """
        Receives audio chunks and outputs transcriptions into CSV
    """
    ##
    global files_lost
    global total_audio_len
    global audio_len_lost
    ##

    if not os.path.exists(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)

    with open(TRANS_DIR + 'metadata.csv', 'a') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, quotechar='')
        for filename in os.listdir(CHUNKS_DIR):
            f = os.path.join(CHUNKS_DIR, filename)
            recog = sr.Recognizer()
            with sr.AudioFile(f) as source:
                audio = recog.record(source)
            try:
                trans = recog.recognize_google(audio)
                writer.writerow([os.path.splitext(filename)[0] + '|' + trans])
                total_audio_len = total_audio_len + AudioSegment.from_file(f).duration_seconds
            except Exception:
                print(f"****File --{filename}-- caused an error****")
                os.remove(f)
                print(f"****File --{filename}-- Removed from dir****")

def write_subset(file, data, x, y):
    temp = data[x:x+y]
    with open(file, 'w') as f:
            for lines in temp:
                f.write(f"{lines}\n")

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
    write_subset('./output/filelists/jps_audio_text_train_subset_64_filelist.txt', data, 700, 64)
    write_subset('./output/filelists/jps_audio_text_train_subset_300_filelist.txt', data, 764, 300)
    write_subset('./output/filelists/jps_audio_text_train_subset_625_filelist.txt', data, 1064, 625)
    write_subset('./output/filelists/jps_audio_text_train_subset_1250_filelist.txt', data, 1689, 1250)
    write_subset('./output/filelists/jps_audio_text_train_subset_2500_filelist.txt', data, 2939, 2500)
    write_subset('./output/filelists/jps_mel_text_filelist.txt', data, 0, data_length)
    write_subset('./output/filelists/jps_mel_text_test_filelist.txt', data, 0, 500)
    write_subset('./output/filelists/jps_mel_text_train_filelist.txt', data, 500, data_length)
    write_subset('./output/filelists/jps_mel_text_val_filelist.txt', data, 600, 100)
    write_subset('./output/filelists/jps_mel_text_train_subset_64_filelist.txt', data, 700, 64)
    write_subset('./output/filelists/jps_mel_text_train_subset_300_filelist.txt', data, 764, 300)
    write_subset('./output/filelists/jps_mel_text_train_subset_625_filelist.txt', data, 1064, 625)
    write_subset('./output/filelists/jps_mel_text_train_subset_1250_filelist.txt', data, 1689, 1250)
    write_subset('./output/filelists/jps_mel_text_train_subset_2500_filelist.txt', data, 2939, 2500)

    data = []
    for filename in os.listdir('./output/filelists/'):
        if fnmatch.fnmatch(filename, '*audio*'):
            with open(FILELIST_DIR+filename, 'r+') as file:
                data = file.read().splitlines()
                file.seek(0)
                for line in data:
                    file.write(f"JPSpeech-1.0/wavs/{line[:10]}.wav{line[10:]}\n")
        elif fnmatch.fnmatch(filename, "*mels*"):
            with open(FILELIST_DIR+filename, 'r+') as file:
                data = file.read().splitlines()
                file.seek(0)
                for line in data:
                    file.write(f"JPSpeech-1.0/mels/{line[:10]}.pt{line[10:]}\n")

def package_data():
    """
        Place all output data into an organized compressed tar file
    """
    with tarfile.open("./output/JPSpeech-1.0.tar.bz2", "w:bz2") as tar:
        tar.add(CHUNKS_DIR, arcname=os.path.basename("./wavs"))
        tar.add(TRANS_DIR, arcname=os.path.basename("/"))
        tar.add(FILELIST_DIR, arcname=os.path.basename("./filelists"))
        tar.close()

def main():

    chunk_audio()
    transcribe_chunks()
    create_training_subsets()
    package_data()

if __name__=="__main__":
    main()