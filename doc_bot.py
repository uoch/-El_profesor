import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
import tqdm as t
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER")
summarizer = pipeline(
    "summarization", model="facebook/bart-large-cnn", device=device)

qa = pipeline("question-answering",
              model="deepset/roberta-base-squad2", device=device)


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


def strong_entities(question):
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(question)
    search_terms = []
    current_term = ""
    for token in ner_results:
        if token["score"] >= 0.99:
            current_term += " " + token["word"]
        else:
            if current_term:
                search_terms.append(current_term.strip())
                current_term = ""
            search_terms.append(token["word"])
    if current_term:
        search_terms.append(current_term.strip())
    print(search_terms[0].split())
    return search_terms[0].split()


def wiki_search(question):
    search_terms = strong_entities(question)
    URL = "https://en.wikipedia.org/w/api.php"
    corpus = []

    for term in set(search_terms):  # Removing duplicates
        SEARCHPAGE = term
        params = {
            "action": "query",
            "format": "json",
            "titles": SEARCHPAGE,
            "prop": "extracts",
            "explaintext": True
        }

        response = requests.get(URL, params=params)
        try:
            if response.status_code == 200:
                data = response.json()
                for page_id, page_data in t.tqdm(data["query"]["pages"].items()):
                    if "extract" in page_data:  # Check if extract exists
                        corpus.append(page_data["extract"])
            else:
                print("Failed to retrieve data:", response.status_code)
        except Exception as e:
            print("Failed to retrieve data:", e)

    final_corpus = []
    for text in corpus:
        sections = re.split("\n\n\n== |==\n\n", text)
        for section in sections:
            if len(section.split()) >= 5:
                final_corpus.append(section)
    return " ".join(final_corpus[0:1])


def semantic_search(corpus, question):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode(question)

    max_similarity = -1
    most_similar_doc = None
    print(type(corpus[0]))
    print(corpus)
    for doc in t.tqdm(corpus):
        if len(doc.split()) >= 130:
            doc_summary = summarizer(
                doc, max_length=130, min_length=30, do_sample=False)
            if len(doc_summary) > 0 and "summary_text" in doc_summary[0]:
                summarized_doc = doc_summary[0]["summary_text"]
            else:
                summarized_doc = doc
        else:
            summarized_doc = doc

        doc_embedding = model.encode(summarized_doc)
        similarity = cosine_similarity(
            [question_embedding], [doc_embedding])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_doc = summarized_doc

    return most_similar_doc, similarity


def dm(q, a, corpus, new_q, max_history_size=5):

    history = deque(maxlen=max_history_size)
    history.append({"question": q, "answer": a, "corpus": corpus})

    best_corpus_index = None
    max_similarity = -1

    for i in range(len(history)):
        _, q_similarity = semantic_search([history[i]["corpus"]], new_q)
        _, a_similarity = semantic_search(
            [history[i]["corpus"]], history[i]["answer"])
        similarity = max(q_similarity, a_similarity)
        if similarity > max_similarity:
            max_similarity = similarity
            best_corpus_index = i

    if best_corpus_index is not None:
        return history[best_corpus_index]["corpus"]
    else:
        return corpus


def first_corp(data, question, botton=False):

    if botton:
        corpus = wiki_search(question)
        texts = [data[i]["text"] for i in range(len(data))]
        text = " ".join(texts)
        corpus = [cp + " " + text for cp in corpus]
    else:
        texts = [data[i]["text"] for i in range(len(data))]
        text = " ".join(texts)
        corpus = [text]
    return " ".join(corpus)


def Qa(image, new_q, internet_access=False):
    old_q = ["how are you?"]
    old_a = ["I am fine, thank you."]
    im_text = extract_text(image)
    if im_text:  # Check if text is extracted
        old_corpus = [first_corp(im_text, old_q[-1], botton=internet_access)]
    else:
        old_corpus = None

    if internet_access:
        if not old_corpus:
            # Pass None as corpus to trigger internet access
            corpus = dm(old_q[-1], old_a[-1], None, new_q)
        else:
            # Pass old_corpus for internet access
            corpus = dm(old_q[-1], old_a[-1], old_corpus, new_q)
    else:
        corpus = old_corpus[0] if old_corpus else None

    a = qa(question=new_q, context=corpus)
    old_q.append(new_q)
    old_a.append(a["answer"])
    old_corpus.append(corpus)

    old_conversations = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(old_q, old_a))

    return a["answer"], old_conversations
