import os
import yaml
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class SettingsLoader:
    @staticmethod
    def load_settings(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
        
class OnnxModelHelper:
    @staticmethod
    def get_output_node_name(onnx_file_path):
        onnx_model = onnx.load(onnx_file_path)
        return [node.name for node in onnx_model.graph.output]

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
            
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class TensorRTEngine:
    @staticmethod
    def get_engine(settings):
        def build_engine():
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()

            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            calibrator = PythonEntropyCalibrator(settings["cache_file"])
            config.int8_calibrator = calibrator

            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(settings["onnx_model"], "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            
            output_node_names = OnnxModelHelper.get_output_node_name(settings["onnx_model"])
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                for j in range(layer.num_outputs):
                    tensor = layer.get_output(j)
                    if tensor.name in output_node_names:
                        network.mark_output(tensor)
            
            engine = builder.build_engine(network, config)
            if engine is None:
                print("Failed to create TensorRT engine!")
            else:
                with open(settings["tensorrt_engine"], "wb") as f:
                    f.write(engine.serialize())
            return engine
        
        if os.path.exists(settings["tensorrt_engine"]):
            with open(settings["tensorrt_engine"], "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()
    
def main():
    settings = SettingsLoader.load_settings("settings.yaml")
    TensorRTEngine.get_engine(settings)

if __name__ == "__main__":
    main()
