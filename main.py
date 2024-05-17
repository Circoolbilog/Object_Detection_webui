import logging
import os
import shutil
import gradio as gr
from Datakit import convert, visualizer
from webui import common_gui
from webui.class_folders import Folders
import argparse


css_path = "./assets/styles.css"
def visualize():
    train = training_data_directory.get_dir()
    val = validation_data_directory.get_dir()

    print(train)
    print(val)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    pass


def parse():
    train = training_data_directory.get_dir()
    val = validation_data_directory.get_dir()
    parsed_train_xml = convert.Converter(data_direct_directory=train,
        data_set='Train.csv', all_gen_dir='all_generated')
    parsed_train_xml.xml_write_to_csv()
    parsed_val_xml = convert.Converter(data_direct_directory=val,
        data_set='Val.csv', all_gen_dir='all_generated')
    parsed_train_xml.xml_write_to_csv()


def create_values():
    pass


def create_tfrecord_shards():
    pass

def train(num_train_steps):
    print(num_train_steps)
    return num_train_steps
    pass

# Create tabs for each function with buttons
with gr.Blocks(title="Object Detection webui", css=css_path) as tabbed_interface:
    with gr.Accordion("Cuda", open=False), gr.Group():
        cuda_directory = Folders(dir_label="Cuda Directory",
                              dir_hint="Use only when cuda is not detected navigate to where the cuda directory is")
    with gr.Row():
        training_data_directory = Folders(dir_label="Training Data Directory", dir_hint='directory')
        validation_data_directory = Folders(dir_label="Validation Data Directory", dir_hint='directory')

    with gr.Tab("Data Prep"):
        with gr.Row():
            parse_button = gr.Button(value="Parse")
            visualize_button = gr.Button(value="Visualize")
            visualize_button.click(fn=visualize)
            parse_button.click(fn=parse)
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

            # Link the button click to the train function
            train_button.click(fn=train, inputs=num_train_steps, outputs=None)

        output_graph = gr.File(label='Graph Output Directory', file_count='directory')
        graph_button = gr.Button(value='Graph')
        populate_button = gr.Button(value="Populate Metadata")


    with gr.Tab("Testing"):
        get_ground_truth = gr.Button(value="Get Ground Truth")
        get_detections = gr.Button(value="Get Detectios")
        get_map = gr.Button(value="Calulate the mAP")


    with gr.Tab("Model"):
        gr.Button("Placeholder")



# Launch the tabbed interface
tabbed_interface.launch()
