import gradio as gr
from recite_module import run
from chatbot_module import respond
from doc_bot import Qa
demo = gr.Blocks()


title = "El_Professor"
description = """
Demo for cascaded speech-to-speech translation (STST), mapping from source speech in any language to target speech in English. Demo uses OpenAI's [Whisper Base](https://huggingface.co/openai/whisper-base) model for speech translation, and Microsoft's
[SpeechT5 TTS](https://huggingface.co/microsoft/speecht5_tts) model for text-to-speech:
![Cascaded STST](https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png "Digram of cascaded speech to speech translation")
"""

demo1 = gr.Interface(
    run,
    [gr.Audio(sources=["microphone"], type="numpy"), gr.Image(
        type="filepath", label="Image")],
    gr.Image(type="pil", label="output Image"),
    title=title,
    description=description
)
demo2 = gr.Interface(
    run,
    [gr.Audio(sources=["upload"]), gr.Image(
        type="filepath", label="Image")],
    [gr.Image(type="pil", label="output Image")],
    title=title,
    description=description
)
demo3 = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.",
                   label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512,
                  step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7,
                  step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)
demo4 = gr.Interface(fn=Qa,
                     inputs=[gr.Image(
                         type="filepath", label="Upload Image"),
                         gr.Textbox(label="Question"),
                         gr.Checkbox(label="Internet access")],
                     outputs=[gr.Textbox(label="Answer"),
                              gr.Textbox(label="Conversations", type="text")],
                     title="Chatbot",
                     description="")
with demo:
    gr.TabbedInterface([demo1, demo2, demo3, demo4], [
                       "Microphone", "Audio File", "general_Chatbot", "Document_Chatbot"])
if __name__ == "__main__":
    demo.launch()
