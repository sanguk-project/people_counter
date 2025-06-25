import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import yaml
import random

from utils.general import non_max_suppression, scale_boxes

def get_unique_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((b, g, r)) # Opencv uses BGR color
    return colors

def load_settings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_tensorrt_engine(runtime, engine_path):
    try:
        with open(engine_path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print("Error loading TensorRT Engine", e)
        exit()
    
def load_video_source(video_path):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise ValueError("Video could not be opened.")
    return vid

def initialize_cuda():
    cuda.init()
    return cuda.Device(0).make_context()

def main():
    # Initialize CUDA
    cuda_ctx = initialize_cuda()

    # Load settings from the YAML file
    settings = load_settings('settings.yaml')

    # Load TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = load_tensorrt_engine(runtime, settings['tensorrt_engine'])

    # Create TensorRT Context
    context = engine.create_execution_context()

    # Prepare input/output memory
    h_input = np.empty(shape=(1, 3, 512, 512), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)

    h_output = np.empty((1, 25200, 6), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Set Class name
    class_names = list(settings['names'].values())

    # Generate unique colors for each class
    class_colors = get_unique_colors(len(class_names))

    # Load video
    vid = load_video_source(settings['video_source'])

    # Define the codec using VidoeWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the corrrect fourcc code for video
    out = cv2.VideoWriter(settings['video_output'], fourcc, 30.0, (1920, 1080))

    # Create CUDA stream
    stream = cuda.Stream()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        # PreProcessing images
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (512, 512))
        input_image = input_image.transpose((2, 0, 1)).astype(np.float32)
        input_image /= 255.0
        input_image = np.expand_dims(input_image, axis=0)
        input_image = np.ascontiguousarray(input_image)

        # Copy input image to GPU memory
        cuda.memcpy_htod_async(d_input, input_image, stream)

        # Change the input shapes
        context.set_binding_shape(0, h_input.shape)

        # Inference
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

        # Synchronize CUDA stream
        stream.synchronize()

        # Copy result back to CPU memory
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

        # Convert h_output to numpy array
        h_output = h_output.astype(np.float32)

        # Apply NMS
        pred = non_max_suppression(h_output, conf_thres=settings['conf_thres_nms'], iou_thres=settings['iou_thres_nms'], classes=None, agnostic=None)

        for i, det in enumerate(pred): # Detectiions per image
            # Rescale boxes from img_size to img.shape
            det[:, :4] = scale_boxes((512, 512), det[:, :4], frame.shape)

            # Draw bounding boxes
            for *xyxy, conf, cls in reversed(det):
                if conf > settings['conf_thres_bbox']: #set confidence threshold
                    
                    # Get class name
                    class_name = class_names[int(cls)]

                    # Get class color
                    color = class_colors[int(cls)]

                    # Adjust the coordinates based on the ration
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # Calculate the center of the bounding box
                    x_center, y_center = (x1 + x2) // 2, (y1 + y2) //2

                    # Draw center point
                    cv2.circle(frame, (x_center, y_center), 8, color, -1,)

                    label_conf = f"{class_name} {conf:.2f}"
                    text_size = max(0.8, (y2 - y1) / frame.shape[0])
                    text_width, text_height = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, text_size, 3)[0]
                    
                    # Adjust text position to make sure it's inside the frame
                    x_text = max(0, min(x_center - text_width // 2, frame.shape[1] - text_width))
                    y_text = max(0, min(y_center - 10, frame.shape[0] - text_height))
                        
                    # Check if text would go beyond the frame and adjust accordingly
                    if x_text + text_width > frame.shape[1]:
                        x_text = frame.shape[1] - text_width

                    if y_text < text_height:
                        y_text = text_height + 2

                    cv2.rectangle(frame, (x_text, y_text - text_height - 3), (x_text + text_width, y_text), color, -1)
                    cv2.putText(frame, label_conf, (x_text, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, text_size, (230, 255, 255), 3)
                
        # Create a named window and resize it
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1280, 720)

        # Show video
        cv2.imshow('frame', frame)

        # Write the resulting frame to the output file
        out.write(frame)

        # Delay between frame
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    # Stop all process
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    # Clean up CUDA context
    cuda_ctx.pop()

if __name__ == "__main__":
    main()