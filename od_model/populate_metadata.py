import tensorflow as tf
import os
from tflite_support import flatbuffers, metadata as _metadata, metadata_schema_py_generated as _metadata_fb


def verify_model(model):
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()

        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
        return False


def add_metadata(model, labelmap):
    print(model)
    verify_model(model)
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "SSD_Detector"
    model_meta.description = (
        "Identify which of a known set of objects might be present and provide "
        "information about their positions within the given image or a video "
        "stream.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5]
    input_normalization.options.std = [127.5]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255]
    input_stats.min = [0]
    input_meta.stats = input_stats

    # Creates outputs info.
    # location

    output_location_meta = _metadata_fb.TensorMetadataT()
    output_location_meta.name = "location"
    output_location_meta.description = "The locations of the detected boxes."
    output_location_meta.content = _metadata_fb.ContentT()
    output_location_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.BoundingBoxProperties)
    output_location_meta.content.contentProperties = (
        _metadata_fb.BoundingBoxPropertiesT())
    output_location_meta.content.contentProperties.index = [1, 0, 3, 2]
    output_location_meta.content.contentProperties.type = (
        _metadata_fb.BoundingBoxType.BOUNDARIES)
    output_location_meta.content.contentProperties.coordinateType = (
        _metadata_fb.CoordinateType.RATIO)
    output_location_meta.content.range = _metadata_fb.ValueRangeT()
    output_location_meta.content.range.min = 2
    output_location_meta.content.range.max = 2

    #category
    output_class_meta = _metadata_fb.TensorMetadataT()
    output_class_meta.name = "category"
    output_class_meta.description = "The categories of the detected boxes."
    output_class_meta.content = _metadata_fb.ContentT()
    output_class_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_class_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_class_meta.content.range = _metadata_fb.ValueRangeT()
    output_class_meta.content.range.min = 2
    output_class_meta.content.range.max = 2
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(labelmap)
    label_file.description = "Label of objects that this model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_class_meta.associatedFiles = [label_file]

    #score
    output_score_meta = _metadata_fb.TensorMetadataT()
    output_score_meta.name = "score"
    output_score_meta.description = "The scores of the detected boxes."
    output_score_meta.content = _metadata_fb.ContentT()
    output_score_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_score_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_score_meta.content.range = _metadata_fb.ValueRangeT()
    output_score_meta.content.range.min = 2
    output_score_meta.content.range.max = 2

    #num of detection
    output_number_meta = _metadata_fb.TensorMetadataT()
    output_number_meta.name = "number of detections"
    output_number_meta.description = "The number of the detected boxes."
    output_number_meta.content = _metadata_fb.ContentT()
    output_number_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_number_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())

    # Creates subgraph info.
    group = _metadata_fb.TensorGroupT()
    group.name = "detection result"
    group.tensorNames = [
        output_location_meta.name, output_class_meta.name,
        output_score_meta.name
    ]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [
        output_location_meta, output_class_meta, output_score_meta,
        output_number_meta
    ]
    subgraph.outputTensorGroups = [group]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator = _metadata.MetadataPopulator.with_model_file(model)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([labelmap])
    populator.populate()

    export_model_path = model

    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    export_json_file = os.path.join(os.path.dirname(export_model_path),  #"detect.json")
                                    os.path.splitext(os.path.basename(export_model_path))[0] + ".json")
    json_file = displayer.get_metadata_json()
    # Optional: write out the metadata as a json file
    with open(export_json_file, "w") as f:
        f.write(json_file)

    print('Populated {} with metadata'.format(export_model_path))


def add_metadata_quantized(model, labelmap):
    print(model)
    verify_model(model)
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "SSD_Detector"
    model_meta.description = (
        "Identify which of a known set of objects might be present and provide "
        "information about their positions within the given image or a video "
        "stream.")

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5]
    input_normalization.options.std = [127.5]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255]
    input_stats.min = [0]
    input_meta.stats = input_stats

    # Creates outputs info.
    # location

    output_location_meta = _metadata_fb.TensorMetadataT()
    output_location_meta.name = "location"
    output_location_meta.description = "The locations of the detected boxes."
    output_location_meta.content = _metadata_fb.ContentT()
    output_location_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.BoundingBoxProperties)
    output_location_meta.content.contentProperties = (
        _metadata_fb.BoundingBoxPropertiesT())
    output_location_meta.content.contentProperties.index = [1, 0, 3, 2]
    output_location_meta.content.contentProperties.type = (
        _metadata_fb.BoundingBoxType.BOUNDARIES)
    output_location_meta.content.contentProperties.coordinateType = (
        _metadata_fb.CoordinateType.RATIO)
    output_location_meta.content.range = _metadata_fb.ValueRangeT()
    output_location_meta.content.range.min = 2
    output_location_meta.content.range.max = 2

    # category
    output_class_meta = _metadata_fb.TensorMetadataT()
    output_class_meta.name = "category"
    output_class_meta.description = "The categories of the detected boxes."
    output_class_meta.content = _metadata_fb.ContentT()
    output_class_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_class_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_class_meta.content.range = _metadata_fb.ValueRangeT()
    output_class_meta.content.range.min = 2
    output_class_meta.content.range.max = 2
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(labelmap)
    label_file.description = "Label of objects that this model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_class_meta.associatedFiles = [label_file]

    # score
    output_score_meta = _metadata_fb.TensorMetadataT()
    output_score_meta.name = "score"
    output_score_meta.description = "The scores of the detected boxes."
    output_score_meta.content = _metadata_fb.ContentT()
    output_score_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_score_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())
    output_score_meta.content.range = _metadata_fb.ValueRangeT()
    output_score_meta.content.range.min = 2
    output_score_meta.content.range.max = 2

    # num of detection
    output_number_meta = _metadata_fb.TensorMetadataT()
    output_number_meta.name = "number of detections"
    output_number_meta.description = "The number of the detected boxes."
    output_number_meta.content = _metadata_fb.ContentT()
    output_number_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_number_meta.content.contentProperties = (
        _metadata_fb.FeaturePropertiesT())

    # Creates subgraph info.
    group = _metadata_fb.TensorGroupT()
    group.name = "detection result"
    group.tensorNames = [
        output_location_meta.name, output_class_meta.name,
        output_score_meta.name
    ]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [
        output_location_meta, output_class_meta, output_score_meta,
        output_number_meta
    ]
    subgraph.outputTensorGroups = [group]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator = _metadata.MetadataPopulator.with_model_file(model)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([labelmap])
    populator.populate()

    export_model_path = model

    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    export_json_file = os.path.join(os.path.dirname(export_model_path),  # "detect.json")
                                    os.path.splitext(os.path.basename(export_model_path))[0] + ".json")
    json_file = displayer.get_metadata_json()
    # Optional: write out the metadata as a json file
    with open(export_json_file, "w") as f:
        f.write(json_file)

    print('Populated {} with metadata'.format(export_model_path))
