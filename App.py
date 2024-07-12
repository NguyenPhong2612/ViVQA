import gradio as gr
from Predict import generate_caption

interface = gr.Interface(
    fn = generate_caption,
    inputs =[gr.components.Image(), gr.components.Textbox(label = "Question")],
    outputs=[gr.components.Textbox(label = "Answer", lines=3)]
)
interface.launch(share = True, debug = True)