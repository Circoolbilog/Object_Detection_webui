import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def export_tflite_graph(pipeline_config_path, trained_checkpoint_dir, output_directory):
    # load the pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # set the desired configurations
    config_override = None
    input_type = 'image_tensor'
    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_directory,
        use_side_inputs=False,
        side_input_types='',
        side_input_names='',
        side_input_shapes=''

    )