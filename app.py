import gradio as gr
from recite_module import run
from chatbot_module import respond
from doc_bot import Qa
demo = gr.Blocks()


title = "El_Professor"
des = """
**El_Professor: Enhance Text Extraction from Images with Audio Transcription**

**How to Use:**

1. **Record Yourself**: Begin by recording yourself speaking the content that corresponds to the text in the image. 

2. **Upload Recorded Audio**: After recording, upload the audio file containing your speech. This audio will be used to enhance text extraction from the image.

3. **Upload Image**: Next, upload the image containing the text you want to extract. Ensure the text in the image is visible and clear.

4. **Check Your Advancement**: Once both the audio and image are uploaded, the application processes them to enhance text extraction. The output will display the processed image with highlighted text regions, showing your advancement in aligning spoken words with written text.

**Note:** This application aims to assist you in improving your ability to accurately transcribe spoken words from images. It may not provide perfect results in all cases, but it can help you track your progress and refine your transcription skills over time.
"""
im = "exemples/the-king-and-three-sisters-around-the-world-stories-for-children.png"

demo1 = gr.Interface(
    run,
    [gr.Audio(sources=["microphone"], type="numpy"), gr.Image(
        type="filepath", label="Image")],
    gr.Image(type="pil", label="output Image"),
    title=title,
    description=des
)
demo2 = gr.Interface(
    run,
    [gr.Audio(sources=["upload"]), gr.Image(
        type="filepath", label="Image")],
    [gr.Image(type="pil", label="output Image")],
    title=title,
    description=des,
    examples=[["exemples/Beginning.wav", im], ["exemples/Middel.wav", im]]
)
demo3 = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friend Chatbot.",
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

demo4 = gr.Interface(
    fn=Qa,
    inputs=[
        gr.Image(type="filepath", label="Upload Document"),
        gr.Textbox(label="Question"),
        gr.Checkbox(label="Enable Internet Access")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Conversations", type="text")
    ],
    title="Document-based Chatbot",
    examples=[[im, "how many sisters in the story", False]],
    description="This chatbot allows you to upload a document and ask questions. It can provide answers based on the content of the document as well as access information from the internet if enabled."
)
with demo:
    gr.TabbedInterface([demo2, demo4, demo1, demo3], [
                       "Audio File", " Document_Chatbot", " Microphone", "general_Chatbot"])
if __name__ == "__main__":
    demo.launch()
