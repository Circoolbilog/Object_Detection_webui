import gradio as gr


# Function placeholders
def visualizer():
    pass


def parse():
    pass


def create_values():
    pass


def create_tfrecord_shards():
    pass


# Create tabs for each function with buttons
with gr.Blocks() as tabbed_interface:
    with gr.Row():
        cuda_directory = gr.File(
            label="CUDA Directory(Use only when cuda is not detected navigate to where the cuda directory is)",
            file_types=[".dll"], height=130)
        training_data_directory = gr.File(label="Training Data Directory", file_count='directory')
        validation_data_directory = gr.File(label="Validation Data Directory", file_count='directory')

    with gr.Tab("Data Prep"):
        visualize_button = gr.Button(value="Visualize" )
        parse_button = gr.Button(value="Parse")
        labels_text = gr.Textbox(label="values")

        create_values_button = gr.Button(value="Create values")

        labels_filename_text = gr.Textbox(label="values File Name")
        create_tfrecord_shards_button = gr.Button(value="Create/Convert to TFRecord/Shards")

    with gr.Tab("Training"):
        gr.Button("Placeholder")

    with gr.Tab("Testing"):
        gr.Button("Placeholder")

    with gr.Tab("Model"):
        gr.Button("Placeholder")



# Launch the tabbed interface
tabbed_interface.launch()
