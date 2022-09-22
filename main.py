import os
import csv
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# imports for converting WAV to MEL Spectograms
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

list = []
AUDIO_DIR = "./recordings/"
CHUNKS_DIR = "./chunks/"
TRANS_DIR = "./transcripts/"
MEL_DIR = "./mel/"


def ChunkAudio():
    """
        Chunks audio by pauses in speech pattern and outputs ####.wav files
    """
    if not os.path.exists(AUDIO_DIR):
        os.mkdir(AUDIO_DIR)
    else:
        chunks = []
        for filename in os.listdir(AUDIO_DIR):
            f = os.path.join(AUDIO_DIR, filename)
            source_aud = AudioSegment.from_file(f)
            chunks = chunks + split_on_silence(source_aud, min_silence_len=875, silence_thresh=-60)
            
        for i, chunk in enumerate(chunks):
            chunk_name = "{0}".format(i)
            chunk_name = chunk_name.zfill(4)+".wav"
            chunk.export(CHUNKS_DIR + chunk_name, format="wav")

def Transcribe():
    """
        Receives audio chunks and outputs transcriptions into CSV
    """
    if not os.path.exists(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)
    else:
        for filename in sorted(os.listdir(CHUNKS_DIR)):
            f = os.path.join(CHUNKS_DIR, filename)
            recog = sr.Recognizer()
            with sr.AudioFile(f) as source:
                audio = recog.record(source)
        
            trans = recog.recognize_google(audio)
            list.append([f"{f}    {trans}"])
    CraftCSV(list)

def CraftCSV(data):
    """
        Take in transcription list and create formated CSV
    """
    if not os.path.exists(TRANS_DIR):
        os.mkdir(TRANS_DIR)
    else:
        with open(TRANS_DIR + 'transcript.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerows(data)


def Wav_to_MEL():
    for filename in sorted(os.listdir(CHUNKS_DIR)):
        f = os.path.join(CHUNKS_DIR, filename)
        #f = "./chunks/0013.wav"
        ipd.Audio(f)

        # load audio files with librosa
        scale, sr = librosa.load(f)

        filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
        filter_banks.shape

        plt.figure(figsize=(25, 10))
        librosa.display.specshow(filter_banks, 
                                sr=sr, 
                                x_axis="linear")
        plt.colorbar(format="%+2.f")
        #plt.show()

        mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
        mel_spectrogram.shape

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrogram.shape
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(log_mel_spectrogram, 
                                x_axis="time",
                                y_axis="mel", 
                                sr=sr)
        plt.colorbar(format="%+2.f")
        plt.savefig(MEL_DIR + os.path.splitext(filename)[0] + '.png')

def main():
    ChunkAudio()
    Transcribe()
    Wav_to_MEL()        

if __name__=="__main__":
    main()