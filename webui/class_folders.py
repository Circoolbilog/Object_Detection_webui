import gradio as gr
from .common_gui import get_folder_path

class Folders:
    """
    A class to handle folder operations in the GUI.
    """

    def __init__(
            self, dir_label: str = "", dir_hint: str = ""
    ):
        self.dir_label = dir_label
        self.dir_hint = dir_hint
        self.text_dir_input = None
        self.browse_dir = None
        self.dir = ""
        self.create_folders_gui()

    def update_dir(self, value):
        self.dir = value
        return value

    def create_folders_gui(self) -> None:
        with gr.Row():
            self.text_dir_input = gr.Textbox(label=self.dir_label, placeholder=self.dir_hint, interactive=True)
            self.browse_dir = gr.Button("📂", elem_id="open_folder_small")

            # Update the directory value whenever the textbox value changes
            self.text_dir_input.change(fn=self.update_dir, inputs=self.text_dir_input, outputs=self.text_dir_input)

            # Move the click method inside the gr.Blocks context
            self.browse_dir.click(get_folder_path, outputs=self.text_dir_input)

    def get_dir(self) -> str:
        return self.dir
