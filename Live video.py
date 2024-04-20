import base64
import json
import collections
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
app = Flask(__name__)
model=load_model('model_weights_new-500/model_199.h5')
from keras.applications import ResNet50
model2=ResNet50(weights="imagenet",input_shape=(224,224,3))
from keras import Model
new_model=Model(model2.input,model2.layers[-2].output)
def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)#for reshaping
    #Normalization
    img=preprocess_input(img)
    return img
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def get_caption(photo,model):
    descriptions = None
    with open('descriptions_created-500.txt', 'r') as f:
        descriptions = f.read()
    json_string = descriptions.replace("'", "\"")
    descriptions = json.loads(json_string)
    total_words = []
    for key in descriptions.keys():
        [total_words.append(i) for des in descriptions[key] for i in des.split()]
    counter = collections.Counter(total_words)
    freq_count = dict(counter)
    sorted_freq_count = sorted(freq_count.items(), reverse=True, key=lambda x: x[1])
    threshold = 10
    sorted_freq_count = [x for x in sorted_freq_count if x[1] > threshold]
    total_words = [x[0] for x in sorted_freq_count]
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(total_words):
        word_to_idx[word] = i + 1
        idx_to_word[i + 1] = word
    idx_to_word[177] = "startseq"
    idx_to_word[178] = "endseq"
    word_to_idx["startseq"] = 177
    word_to_idx['endseq'] = 178
    vocab_size = len(word_to_idx) + 1
    maxlen=19

    in_text='startseq'
    for i in range(maxlen):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence],maxlen=maxlen,padding='post')
        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax()#gives word with maximum probability--greedy sampling
        word=idx_to_word[ypred]
        in_text+=' '+word
        if word=='endseq':
            break
    final_caption=in_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    return final_caption
import cv2
import pyttsx3
import threading
import time

# Function to perform video processing
# Function to capture video frames and process them
def process_video(video_capture, frames_with_captions, interval, output_folder, model):
    # Set the camera properties (you can adjust these as needed)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = video_capture.read(0)

        if not ret:
            print("Failed to read a frame.")
            break

        elapsed_time = time.time() - start_time

        if elapsed_time >= interval:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            photo_2048 = encode_image(frame_path).reshape((1, 2048))
            caption = get_caption(photo_2048, model)

            start_time = time.time()
            frame_count += 1

            frames_with_captions.append((frame, caption))


# Function to perform audio insertion
def insert_audio(frames_with_captions):
    engine = pyttsx3.init()

    while True:
        if frames_with_captions:
            frame, caption = frames_with_captions.pop(0)

            cv2.putText(frame, caption, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            engine.say(caption)
            engine.runAndWait()

            cv2.imshow("Live Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

4
if __name__ == "__main__":
    output_folder = "frames_output"
    os.makedirs(output_folder, exist_ok=True)
    frames_with_captions = []
    interval = 6 # Adjust the interval as needed for your desired frame rate

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Failed to open video capture.")
        exit()

    # Load your captioning model here (replace 'your_model.h5' with your actual model path)
    model = load_model('model_weights_new-500/model_199.h5')

    # Create and start the video processing thread
    video_thread = threading.Thread(target=process_video,
                                    args=(video_capture, frames_with_captions, interval, output_folder, model))
    video_thread.start()

    # Start the audio insertion function in the main thread
    insert_audio(frames_with_captions)

    # Wait for the video processing thread to finish
    video_thread.join()
