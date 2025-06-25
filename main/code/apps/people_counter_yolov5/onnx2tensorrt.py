import tensorrt as trt
import yaml

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ERROR(오류메세지만 기록), WARNING(오류와 경고 메세지가 기록), INFO(오류, 경고, 그리고 정보 메세지 기록), VERBOSE(모든 종류의 메세지 기록) 

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER) # TensorRT의 Builder 객체를 생성 → TensorRT 엔진을 생성할 때 필요
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) # 네트워크 객체를 생성하여 ONNX 모델을 파싱할 때 사용
    config = builder.create_builder_config() # TensorRT 엔진의 구성 설정을 위한 객체를 생성

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            
            return None

    # Enable fp16 mode
    config.set_flag(trt.BuilderFlag.FP16) # FP16(반정밀도) 모드를 활성화
    config.set_flag(trt.BuilderFlag.STRICT_TYPES) # 모든 텐서와 레이어가 명시적으로 지정된 데이터 타입으로만 연산되도록 강제
    
    engine = builder.build_engine(network, config) # 주어진 네트워크와 구성을 사용하여 TensorRT 엔진을 생성
    
    if engine is None:
        print("Failed to create engine")
    
    return engine

def save_engine(engine, engine_file_path):
    if engine is not None:
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
    else:
        print("Engine is not saved since it's None")

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