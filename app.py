import gradio as gr
from recite_module import run
from chatbot_module import respond
demo = gr.Blocks()


demo1 = gr.Interface(
    run,
    [gr.Audio(sources=["microphone"], type="numpy"), gr.Image(
        type="filepath", label="Image")],
    gr.Image(type="pil", label="output Image"),
)
demo2 = gr.Interface(
    run,
    [gr.Audio(sources=["upload"]), gr.Image(
        type="filepath", label="Image")],
    [gr.Image(type="pil", label="output Image")]
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
with demo:
    gr.TabbedInterface([demo1, demo2, demo3], [
                       "Microphone", "Audio File", "Chatbot"])
if __name__ == "__main__":
    demo.launch()
