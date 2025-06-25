import os
import yaml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import onnx
from tqdm import tqdm

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
    def __init__(self, calibration_data, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = calibration_data
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data.nbytes)

    def get_batch_size(self):
        return self.data.shape[0]
    
    def get_batch(self, names):
        if self.current_index + self.get_batch_size() > self.data.shape[0]:
            return None
        
        current_batch = self.data[self.current_index:self.current_index + self.get_batch_size()]
        self.current_index += self.get_batch_size()

        current_batch = current_batch.ravel()
        cuda.memcpy_htod(self.device_input, current_batch)

        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
            
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def __del__(self):
        del self.device_input
    
class TensorRTEngine:
    @staticmethod
    def provide_calibration_data(settings):
        image_dir = settings["dataset"]
        image_files = os.listdir(image_dir)
        images = []
        for image_file in tqdm(image_files):
            image_path = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue
            img = cv2.resize(img, (settings["img_size"], settings["img_size"]))
            img = img.astype(np.float32)
            img = img.transpose((2, 0, 1))
            img /= 255.0
            img = np.ascontiguousarray(img)
            images.append(img)
        return np.stack(images)
    
    @staticmethod
    def get_engine(settings):
        def build_engine():
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()

            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            calibration_data = TensorRTEngine.provide_calibration_data(settings)
            calibrator = PythonEntropyCalibrator(calibration_data, settings["cache_file"])
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
    engine = TensorRTEngine.get_engine(settings)

if __name__ == "__main__":
    main()