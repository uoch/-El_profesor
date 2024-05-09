import gradio as gr
from recite_module import run
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
    [gr.Image(type="pil", label="output Image")]
)
with demo:
    gr.TabbedInterface([demo1, demo2],
                       ["Microphone", "Audio File"])

demo.launch()
