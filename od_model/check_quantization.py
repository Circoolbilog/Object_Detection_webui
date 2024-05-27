import tensorflow as tf
import numpy as np
import sys

def check_quantization(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()

    # Check if the model is quantized
    is_quantized = input_details[0]['dtype'] == np.uint8
    return is_quantized


def convert_to_quantized_tflite(model_path, quantization_method):
    # Initialize the TFLIteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # Apply the quantization method

    if quantization_method == 'dynamic_range':
        # Set optimization to dynamic range optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_method == 'full_integer':
        # Set the optimization to full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Provide representative dataset for calibration
        converter.representative_dataset = representative_dataset
    elif quantization_method == 'float16':
        # Set the optimization to float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_specs.supported_ops = [tf.float16]
    else:
        raise ValueError("Invalid quantization method. Supported methods: 'dynamic_range', 'full_integer', 'float16'")

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.TFLITE_BUILTINS # Enable TensorFlow ops.
    ]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    # Convert the model
    tflite_quant_model = converter.convert()

    # Save the quantized model
    tflite_model_path = model_path + "_quant.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_quant_model)

    return tflite_model_path


# Define a a representative dataset function for full integer quantization
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3)
        yield [data.astype(np.float32)]
    return