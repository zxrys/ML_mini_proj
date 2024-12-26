"""
    This file is part of ML_mini_proj project
    Copyright (C) 2024 Yao Shu  <springrainyszxr@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import speech_recognition as sr
import pandas as pd
import wave
import numpy as np
import ssl
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context


# Encapsulate CMU Sphinx speech recognition functionality
def recognize_with_sphinx(recognizer, audio):
    return recognizer.recognize_sphinx(audio)


# Encapsulate Whisper speech recognition functionality
def recognize_with_whisper(model, file_path, language):
    result = model.transcribe(file_path, language=language)
    return result["text"]


# Encapsulate DeepSpeech speech recognition functionality
def recognize_with_deepspeech(model, file_path):
    with wave.open(file_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    return model.stt(audio)


# Encapsulate Google Cloud speech recognition functionality (as an example)
def recognize_with_google_cloud(recognizer, audio, language):
    return recognizer.recognize_google_cloud(audio, language=language)


def convert_speech_to_text(input_base_path, output_folder, csv_file, model_type="whisper", whisper_model_size="base"):
    # Initialize speech recognizer and model
    recognizer = sr.Recognizer()
    model = None

    # Adjust output folder name based on selected model
    if model_type == "whisper":
        output_folder += f"/whisper_{whisper_model_size}"
    elif model_type == "google_cloud":
        output_folder += "/google_cloud"
    elif model_type == "deepspeech":
        output_folder += "/deepspeech"
    else:
        raise ValueError(f"{model_type} is not supported, now supports [whisper, google_cloud, deepspeech]")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If using Whisper model
    if model_type == "whisper":
        import whisper
        model = whisper.load_model(whisper_model_size)  # Load the specified Whisper model
    # If using DeepSpeech model
    elif model_type == "deepspeech":
        import deepspeech
        model = deepspeech.Model("deepspeech-0.9.3-models.pbmm")  # Load DeepSpeech model
    # If using Google Cloud API
    elif model_type == "google_cloud":
        pass  # Google Cloud uses the default recognizer, no extra model loading needed

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Initialize a list to store paths of the saved text files
    saved_file_paths = []

    # Iterate over each row in the CSV file with tqdm for progress
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio Files", position=0):
        filename = row['filename']
        language = row['Language']  # e.g., 'zh-CN', 'en-US'
        story_type = row['Story_type']

        # Construct the full path to the audio file
        file_path = os.path.join(input_base_path, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_file_path = os.path.join(output_folder, output_filename)

        # Check if the WAV file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        if os.path.exists(output_file_path):
            print(f"Processed file {output_file_path} already exists, skip preprocessing")
            saved_file_paths.append(output_file_path)
            continue

        try:
            # Open the audio file
            with sr.AudioFile(file_path) as source:
                # print(f"Processing: {filename} (Language: {language})")
                audio = recognizer.record(source)

            # Perform recognition based on the selected model
            if model_type == "sphinx":
                text = recognize_with_sphinx(recognizer, audio)
            elif model_type == "whisper":
                text = recognize_with_whisper(model, file_path, language)
            elif model_type == "deepspeech":
                text = recognize_with_deepspeech(model, file_path)
            elif model_type == "google_cloud":
                text = recognize_with_google_cloud(recognizer, audio, language)

            # print(f"Result: {text}")

            # Save the result to a text file
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            saved_file_paths.append(output_file_path)  # Add the path to the list

        except sr.UnknownValueError:
            print(f"Unable to process {filename} (Unrecognized speech)")
        except sr.RequestError as e:
            print(f"Unable to request Google Cloud for {filename}, msg: {e}")
        except Exception as e:
            print(f"Error occurred when preprocessing {filename}: {e}")

    return saved_file_paths


if __name__ == "__main__":
    input_base_path = "../../data/raw/CBU0521DD_stories"
    output_folder = "../data/processed/text_feature"
    csv_file = "../../data/raw/CBU0521DD_stories_attributes.csv"

    # Call the function with the desired model ("sphinx", "whisper", "deepspeech", "google_cloud")
    saved_files = convert_speech_to_text(input_base_path, output_folder, csv_file, model_type="whisper",
                                         whisper_model_size="base")

    # Print final saved paths
    print(f"Files saved to the following paths: {saved_files}")
