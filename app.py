import gradio as gr
from transformers import pipeline
import numpy as np
import pytesseract
import cv2
from PIL import Image
from evaluate import load
import librosa

asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
wer = load("wer")


def extract_text(image):
    result = pytesseract.image_to_data(image, output_type='dict')
    n_boxes = len(result['level'])
    data = {}
    k = 0
    for i in range(n_boxes):
        if result['conf'][i] >= 0.3 and result['text'][i] != '' and result['conf'][i] != -1:
            data[k] = {}
            (x, y, w, h) = (result['left'][i], result['top']
                            [i], result['width'][i], result['height'][i])
            data[k]["coordinates"] = (x, y, w, h)
            text, conf = result['text'][k], result['conf'][k]
            data[k]["text"] = text
            data[k]["conf"] = conf
            k += 1
    return data


def draw_rectangle(image, x, y, w, h, color=(0, 0, 255), thickness=2):
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, thickness)
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))


def transcribe(audio):
    if isinstance(audio, str):  # If audio is a file path
        y, sr = librosa.load(audio)
    elif isinstance(audio, tuple) and len(audio) == 2:  # If audio is (sampling_rate, raw_audio)
        sr, y = audio
        y = y.astype(np.float32)
    else:
        raise ValueError("Invalid input. Audio should be a file path or a tuple of (sampling_rate, raw_audio).")
    
    y /= np.max(np.abs(y))

    # Call your ASR (Automatic Speech Recognition) function here
    # For now, let's assume it's called 'asr'
    transcribed_text = asr({"sampling_rate": sr, "raw": y})["text"]
    
    return transcribed_text


def clean_transcription(transcription):
    text = transcription.lower()
    words = text.split()
    cleaned_words = [words[0]]
    for word in words[1:]:
        if word != cleaned_words[-1]:
            cleaned_words.append(word)
    return ' '.join(cleaned_words)


def match(refence, spoken):
    wer_score = wer.compute(references=[refence], predictions=[spoken])
    score = 1 - wer_score
    return score


def split_to_l(text, answer):
    l = len(answer.split(" "))
    text_words = text.split(" ")
    chunks = []
    indices = []
    for i in range(0, len(text_words), l):
        chunk = " ".join(text_words[i: i + l])
        chunks.append(chunk)
        indices.append(i)
    return chunks, indices, l


def reindex_data(data, index, l):
    reindexed_data = {}
    for i in range(l):
        original_index = index + i
        reindexed_data[i] = data[original_index]
    return reindexed_data


def process_image(im, data):
    im_array = np.array(im)
    hg, wg, _ = im_array.shape
    text_y = np.max([data[i]["coordinates"][1]
                    for i in range(len(data))])
    text_x = np.max([data[i]["coordinates"][0]
                    for i in range(len(data))])
    text_start_x = np.min([data[i]["coordinates"][0]
                           for i in range(len(data))])
    text_start_y = np.min([data[i]["coordinates"][1]
                           for i in range(len(data))])
    max_height = int(np.mean([data[i]["coordinates"][3]
                              for i in range(len(data))]))
    max_width = int(np.mean([data[i]["coordinates"][2]
                    for i in range(len(data))]))
    text = [data[i]["text"] for i in range(len(data))]
    wall = np.zeros((hg, wg, 3), np.uint8)

    wall[text_start_y:text_y + max_height, text_start_x:text_x + max_width] = \
        im_array[text_start_y:text_y + max_height,
                 text_start_x:text_x + max_width, :]

    for i in range(1, len(data)):
        x, y, w, h = data[i]["coordinates"]
        wall = draw_rectangle(wall, x, y, w, h)
    return wall


def run(stream, image):
    data = extract_text(image)
    im_text_ = [data[i]["text"] for i in range(len(data))]
    im_text = " ".join(im_text_)
    trns_text = transcribe(stream)
    chunks, index, l = split_to_l(im_text, trns_text)
    im_array = np.array(Image.open(image))
    data2 = None
    for i in range(len(chunks)):
        if match(chunks[i], trns_text) > 0.1:
            data2 = reindex_data(data, index[i], l)
            break
    if data2 is not None:
        return process_image(im_array, data2)
    else:
        return im_array

demo = gr.Blocks()


demo1 = gr.Interface(
    run,
    [gr.Audio(sources=["microphone"] , type="numpy"), gr.Image(
        type="filepath", label="Image")],
    gr.Image(type="pil", label="output Image"),
)
demo2 = gr.Interface(
    run,
    [gr.Audio(sources=["upload"]), gr.Image(
        type="filepath", label="Image")],
    gr.Image(type="pil", label="output Image")
)
with demo:
    gr.TabbedInterface([demo1, demo2],
                       ["Microphone", "Audio File"])

demo.launch()
"""
data = extract_text(im)
im_text_ = [data[i]["text"] for i in range(len(data))]
im_text = " ".join(im_text_)
trns_text = transcribe_wav("tmpmucht0kh.wav")
chunks, index, l = split_to_l(im_text, trns_text)
im_array = np.array(Image.open(im))
for i in range(len(chunks)):
    if match(chunks[i], trns_text) > 0.5:
        print(chunks[i])
        print(match(chunks[i], trns_text))
        print(index[i])
        print(l)
        print(im_array.shape)
        print(fuse_rectangles(im_array, data, index[i], l))

strem = "tmpq0eha4we.wav"
im = "the-king-and-three-sisters-around-the-world-stories-for-children.png"
text = "A KING AND THREE SISTERS"
che_text = "A KING AND THREE SISTERS"
print(match(text, che_text))
data = extract_text(im)
text_transcript = transcribe_wav(strem)
print(text_transcript)
im_text_ = [data[i]["text"] for i in range(len(data))]
im_text = " ".join(im_text_)
print(im_text)
wall = run(strem, im)
wall.show()"""

