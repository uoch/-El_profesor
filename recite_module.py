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
    """
    Extracts text from an image using OCR.
    Args:
        image (PIL.Image.Image): Input image.
    Returns:
        dict: Extracted text with confidence and coordinates.
    Raises:
        ValueError: If the input image is not a PIL Image object.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid input. Image should be a PIL Image object.")

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
    """
    Draws a rectangle on the given image.
    Args:
        image (PIL.Image.Image): Input image.
        x (int): x-coordinate of the top-left corner of the rectangle.
        y (int): y-coordinate of the top-left corner of the rectangle.
        w (int): Width of the rectangle.
        h (int): Height of the rectangle.
        color (tuple, optional): Color of the rectangle in RGB format.
        thickness (int, optional): Thickness of the rectangle's border.
    Returns:
        PIL.Image.Image: Image with the rectangle drawn on it.
    Raises:
        ValueError: If the input image is not a PIL Image object.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid input. Image should be a PIL Image object.")

    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, thickness)
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))


def transcribe(audio):
    """
    Transcribes audio into text using ASR.
    Parameters:
        audio (str or tuple): Audio source.
    Returns:
        str: Transcribed text.
    Raises:
        ValueError: If the input audio is not valid.
    """
    if not isinstance(audio, (str, tuple)):
        raise ValueError(
            "Invalid input. Audio should be either a file path or a tuple of (sampling_rate, raw_audio).")

    if isinstance(audio, str):  # If audio is a file path
        y, sr = librosa.load(audio)
    # If audio is (sampling_rate, raw_audio)
    elif isinstance(audio, tuple) and len(audio) == 2:
        sr, y = audio
        y = y.astype(np.float32)
    else:
        raise ValueError(
            "Invalid input. Audio should be a file path or a tuple of (sampling_rate, raw_audio).")

    y /= np.max(np.abs(y))

    transcribed_text = asr(
        {"sampling_rate": sr, "raw": y}, language="en")["text"]

    return transcribed_text


def clean_transcription(transcription):
    """
    Cleans the transcription by removing consecutive duplicate words.
    Args:
        transcription (str): Input transcription.
    Returns:
        str: Cleaned transcription.
    Raises:
        ValueError: If the input transcription is not a string.
    """
    if not isinstance(transcription, str):
        raise ValueError("Invalid input. Transcription should be a string.")

    text = transcription.lower()
    words = text.split()
    cleaned_words = [words[0]]
    for word in words[1:]:
        if word != cleaned_words[-1]:
            cleaned_words.append(word)
    return ' '.join(cleaned_words)


def match(refence, spoken):
    """
    Calculates the match score between a reference and spoken string.
    Args:
        reference (str): Reference string.
        spoken (str): Spoken string.
    Returns:
        float: Match score between 0 and 1.
    Raises:
        ValueError: If either reference or spoken is not a string.
    """
    if not isinstance(refence, str) or not isinstance(spoken, str):
        raise ValueError(
            "Invalid input. Reference and spoken should be strings.")

    if spoken == "":
        return 0
    wer_score = wer.compute(references=[refence], predictions=[spoken])
    score = 1 - wer_score
    return score


def split_to_l(text, answer):
    """
    Splits the given text into chunks of length 'l' based on the answer.
    Args:
        text (str): The input text to be split.
        answer (str): The answer used to determine the chunk size.
    Returns:
        tuple: A tuple containing the chunks of text, the indices of the chunks, and the length of each chunk.
    """
    if not isinstance(text, str) or not isinstance(answer, str):
        raise ValueError("Invalid input. Text and answer should be strings.")

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
    """
    Reindexes a dictionary with keys ranging from 0 to l-1.
    Args:
        data (dict): Original dictionary.
        index (int): Starting index for reindexing.
        l (int): Length of the reindexed dictionary.
    Returns:
        dict: Reindexed dictionary.
    Raises:
        ValueError: If the input data is not a dictionary, or if index or l are not integers.
    """
    if not isinstance(data, dict) or not isinstance(index, int) or not isinstance(l, int):
        raise ValueError(
            "Invalid input. Data should be a dictionary, index and l should be integers.")

    reindexed_data = {}
    for i in range(l):
        original_index = index + i
        reindexed_data[i] = data[original_index]
    return reindexed_data


def process_image(im, data):
    """
    Processes an image by extracting text regions.
    Args:
        im (PIL.Image.Image): Input image.
        data (dict): Data containing information about text regions.
    Returns:
        numpy.ndarray: Processed image with text regions highlighted.
    Raises:
        ValueError: If the input image is not a PIL Image object or if the data is not a dictionary.
    """
    if not isinstance(im, Image.Image) or not isinstance(data, dict):
        raise ValueError(
            "Invalid input. Image should be a PIL Image object and data should be a dictionary.")

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
    wall = np.zeros((hg, wg, 3), np.uint8)

    wall[text_start_y:text_y + max_height, text_start_x:text_x + max_width] = \
        im_array[text_start_y:text_y + max_height,
                 text_start_x:text_x + max_width, :]

    for i in range(1, len(data)):
        x, y, w, h = data[i]["coordinates"]
        wall = draw_rectangle(wall, x, y, w, h)
    return wall


def run(stream, image):
    """
    Processes an image and transcribes audio.
    Args:
        stream (str or tuple): Audio source.
        image (PIL.Image.Image): Input image.
    Returns:
        numpy.ndarray or PIL.Image.Image: Processed image data.
    Raises:
        ValueError: If the input stream is not a valid type or if the input image is not a PIL Image object.
    """
    if not isinstance(stream, (str, tuple)):
        raise ValueError(
            "Invalid input. Stream should be either a file path or a tuple of (sampling_rate, raw_audio).")

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid input. Image should be a PIL Image object.")

    data = extract_text(image)
    im_text_ = [data[i]["text"] for i in range(len(data))]
    im_text = " ".join(im_text_)
    trns_text = transcribe(stream)
    chunks, index, l = split_to_l(im_text, trns_text)
    im_array = np.array(Image.open(image))
    data2 = None
    for i in range(len(chunks)):
        if match(chunks[i], trns_text) > 0.5:
            data2 = reindex_data(data, index[i], l)
            break
    if data2 is not None:
        return process_image(im_array, data2)
    else:
        return im_array
