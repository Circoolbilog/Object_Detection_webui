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

def train(num_train_steps):
    print(num_train_steps)
    return num_train_steps
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
        with gr.Row():
            model_dir = gr.File(label="Model Directory", file_count='directory')
            config_file = gr.File(label="Pipeline config file", file_count='single', file_types=['.config'])

        with gr.Row():
            num_train_steps = gr.Slider(label='Number of training steps',
                                        value=1000, minimum=500, maximum=50000,
                                        step=500, info="how many steps should the training take",
                                        interactive=True)
            train_button = gr.Button(value='Start training')
            train_output = gr.Textbox(label="Training Output")

            # Link the button click to the train function
            train_button.click(fn=train, inputs=num_train_steps, outputs=train_output)

        output_graph = gr.File(label='Graph Output Directory', file_count='directory')
        graph_button = gr.Button(value='Graph')
        populate_button = gr.Button(value="Populate Metadata")


    with gr.Tab("Testing"):
        get_ground_truth = gr.Button(value="Get Ground Truth")
        get_detections = gr.Button(value="Get Detections")
        get_map = gr.Button(value="Calulate the mAP")


    with gr.Tab("Model"):
        gr.Button("Placeholder")



# Launch the tabbed interface
tabbed_interface.launch()
