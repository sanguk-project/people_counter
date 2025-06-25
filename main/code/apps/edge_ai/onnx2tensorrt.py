import tensorrt as trt
import yaml

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())

    # Enable fp16 mode
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to create engine")
    
    return engine

def save_engine(engine, engine_file_path):
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

def load_settings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    settings = load_settings('settings.yaml')

    # ONNX model path
    onnx_model_path = settings['onnx_model']

    # TensorRT engine path
    engine_file_path = settings['tensorrt_engine']

    # Build the engine
    engine = build_engine(onnx_model_path)

    # Save the engine
    save_engine(engine, engine_file_path)
    
if __name__ == '__main__':
    main()