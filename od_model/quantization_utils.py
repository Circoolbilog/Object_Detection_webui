import tensorflow as tf
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def export_tflite_graph(pipeline_config_path, trained_checkpoint_dir, output_directory):
    # Load the pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Set the desired configurations
    config_override = None
    input_type = 'image_tensor'

    # Export the TFLite graph
    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_directory,
        use_side_inputs=False,
        side_input_shapes='',
        side_input_types='',
        side_input_names='',
    )


def is_quantized_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details + output_details:
        if detail['dtype'] in [tf.uint8, tf.int8]:
            return True
    return False


def convert_to_quantized_tflite(saved_model_dir, output_tflite_model_path):
    # Load the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Set the optimization flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    tflite_model = converter.convert()

    # Save the quantized model
    with open(output_tflite_model_path, 'wb') as f:
        f.write(tflite_model)


def convert_to_full_integer_quantized_tflite(saved_model_dir, output_tflite_model_path):
    def representative_data_gen():
        # Provide a representative dataset
        for _ in range(100):  # num_calibration_steps
            # Get sample input data as a numpy array in a method of your choosing
            yield [input_data]

    # Load the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Set the optimization flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide a representative dataset
    converter.representative_dataset = representative_data_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set the input and output tensors to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model
    tflite_model = converter.convert()

    # Save the quantized model
    with open(output_tflite_model_path, 'wb') as f:
        f.write(tflite_model)
