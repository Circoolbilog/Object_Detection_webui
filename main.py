import gradio as gr
from Datakit import convert, visualizer, create_labels, convert_to_tfrecord
from webui.class_folders import Folders
from od_model.train_model_tf2 import Config, run_model
from od_model.graph import export_tflite_graph
from od_model.quantization_utils import export_tflite_graph, is_quantized_model, convert_to_quantized_tflite
css_path = "./assets/styles.css"


def visualize():
    vis = visualizer.Visualizer(csv_file="Train_vs_Val",
                                train_csv="all_generated/Train.csv", val_csv="all_generated/Val.csv",
                                save_as_png="Train_vs_Val.png", html_file="vis_graph.html")
    vis.graph()
    pass

def graph():
    pipeline_config_path = config_file.get_dir()
    trained_checkpoint_dir = model_dir.get_dir()
    output_directory = "output"
    export_tflite_graph(pipeline_config_path, trained_checkpoint_dir, output_directory)
    pass

def parse():
    train = training_data_directory.get_dir()
    val = validation_data_directory.get_dir()
    parsed_train_xml = convert.Converter(data_direct_directory=train,
                                         data_set='Train.csv', all_gen_dir='all_generated')
    parsed_train_xml.xml_write_to_csv()
    parsed_val_xml = convert.Converter(data_direct_directory=val,
                                       data_set='Val.csv', all_gen_dir='all_generated')
    parsed_val_xml.xml_write_to_csv()


def create_values(label):
    labels = create_labels.Labels(labels_fname=f"{label}",
                                  train_vs_val_csv="Train_vs_Val")
    labels.create_txt()
    labels.create_pbtxt()


def create_tfrecord_shards(label: str, num_train: int, num_val: int):
    tfr = convert_to_tfrecord.shardTFRecord(csv_file_input="all_generated/Val.csv",
                                            pbtxt_labels=f"all_generated/{label}.pbtxt", data_type="Val",
                                            num_of_shards=num_val)
    tfr.convert()
    tfr_val = convert_to_tfrecord.shardTFRecord(csv_file_input="all_generated/Train.csv",
                                                pbtxt_labels=f"all_generated/{label}.pbtxt", num_of_shards=num_train)
    tfr_val.convert()


def train(num_steps):
    pipeline_config = config_file.get_dir()
    model = model_dir.get_dir()
    config = Config(pipeline_config_path=pipeline_config,
                    model_dir=model,
                    num_train_steps=num_steps,
    )
    run_model(config)
    print(config)

    return num_steps
    pass

def handle_tflite_model(file_obj):
    tflite_model_path = file_obj.name
    if is_quantized_model(tflite_model_path):
        return "The model is already quantized."
    else:
        # Specify paths for model export and conversion
        pipeline_config_path = "D:/Machine Learning/Abaca Disease Detection/models/research/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config"
        trained_checkpoint_dir = "D:/Machine Learning/Abaca Disease Detection/models/research/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/new20"
        output_directory = "D:/Machine Learning/Abaca Disease Detection/models/research/tflite18"

        # Export TFLite graph if not already done
        export_tflite_graph(pipeline_config_path, trained_checkpoint_dir, output_directory)

        # Convert exported model to quantized TFLite
        saved_model_dir = output_directory
        output_tflite_model_path = "D:/Machine Learning/Abaca Disease Detection/models/research/quantized_model.tflite"
        convert_to_quantized_tflite(saved_model_dir, output_tflite_model_path)

        return "The model has been quantized and saved to " + output_tflite_model_path

# Create tabs for each function with buttons
with gr.Blocks(title="Object Detection webui", css=css_path) as tabbed_interface:
    with gr.Accordion("Cuda", open=False), gr.Group():
        cuda_directory = Folders(dir_label="Cuda Directory",
                                 dir_hint="Use only when cuda is not detected navigate to where the cuda directory is")
    with gr.Row():
        training_data_directory = Folders(dir_label="Training Data Directory", dir_hint='directory')
        validation_data_directory = Folders(dir_label="Validation Data Directory", dir_hint='directory')
    val = validation_data_directory.get_dir()

    with gr.Tab("Data Prep"):
        with gr.Row():
            parse_button = gr.Button(value="Parse")
            visualize_button = gr.Button(value="Visualize")
            visualize_button.click(fn=visualize)
            parse_button.click(fn=parse)
        with gr.Row():
            val_shards = gr.Number(label="Val Shards", scale=1, value=1, minimum=1, maximum=10, interactive=True)
            train_shards = gr.Number(label="Train Shards", scale=1, value=3, minimum=1, maximum=10, interactive=True)
            labels_text = gr.Textbox(label="values", scale=8)

        create_values_button = gr.Button(value="Create values")
        create_tfrecord_shards_button = gr.Button(value="Create/Convert to TFRecord/Shards")

        create_values_button.click(fn=create_values, inputs=[labels_text])
        create_tfrecord_shards_button.click(fn=create_tfrecord_shards, inputs=[labels_text, train_shards, val_shards])

    with gr.Tab("Training"):
        with gr.Row():
            model_dir = Folders(dir_label="Model dir", dir_hint='Model')
            config_file = Folders(dir_label="Pipeline config", dir_hint='pipeline.config')

        with gr.Row():
            num_train_steps = gr.Slider(label='Number of training steps',
                                        value=1000, minimum=500, maximum=50000,
                                        step=500, info="how many steps should the training take",
                                        interactive=True)
            train_button = gr.Button(value='Start training')

            # Link the button click to the train function
            train_button.click(fn=train, inputs=num_train_steps)

        output_graph = Folders(dir_label="Graph output", dir_hint="Graph directory output")
        graph_button = gr.Button(value='Graph')
        graph_button.click(fn=graph)
        populate_button = gr.Button(value="Populate Metadata")
        populate_button.click()

    with gr.Tab("Testing"):
        get_ground_truth = gr.Button(value="Get Ground Truth")
        get_detections = gr.Button(value="Get Detection")
        get_map = gr.Button(value="Calculate the mAP")

    with gr.Tab("Model"):
        tflite_model = gr.File(label="Tflite Model")
        result = gr.Textbox(label="Result")
        tflite_model.upload(handle_tflite_model, inputs=tflite_model, outputs=result)
        gr.Button("Placeholder")

# Launch the tabbed interface
tabbed_interface.launch()
