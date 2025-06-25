import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import random
import yaml
import timeit

from utils.general import non_max_suppression, scale_boxes
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from shapely.geometry import Polygon, Point

# Get GPU name
def get_gpu_name():
    cuda.init()
    device_count = cuda.Device.count()
    gpu_names = []
    for i in range(device_count):
        device = cuda.Device(i)
        gpu_names.append(f"GPU {i} - {device.name()}")

    return gpu_names
    
gpu_names = get_gpu_name()
print("GPU name:", gpu_names)

# Initialize CUDA
cuda.init()

# Create CUDA context
cuda_ctx = cuda.Device(0).make_context()

# Load settings from the YAML file
with open('settings.yaml', 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# LoadTensorRT 
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(TRT_LOGGER)

try:
    with open(settings['tensorrt_engine'], 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
except Exception as e:
    print("Error loading TensorRT Engine:", e)
    exit()

# Create TensorRT Context
context = engine.create_execution_context()

# Prepare input/output memory
h_input = np.empty(shape=(1, 3, settings['img_size'], settings['img_size']), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)

h_output = np.empty((1, 16128, 6), dtype=np.float32) # Assuming n classes + 5 bounding box attributes, input_shape(512, 512) → 16,128
d_output = cuda.mem_alloc(h_output.nbytes)

def get_unique_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((b, g, r)) # OpenCV uses BGR color format
    
    return colors

# Set Class name
class_names = list(settings['names'].values())

# Generate unique colors for each class
class_colors = get_unique_colors(len(class_names))
print("The Number of colors:", len(class_colors))

# Load video 
try:
    vid = cv2.VideoCapture(settings['video_source'])
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Resolution: {} x {}'.format(width, height))

    if not vid.isOpened():
        raise ValueError("Video could not be opened.")
except Exception as e:
    print("Error opening video source:", e)
    exit()

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the correct fourcc code for video
out = cv2.VideoWriter(settings['video_output'], fourcc, 30.0, (1920, 1080))


# Set the variable for people counting
ct = CentroidTracker(maxDisappeared = settings['maxDisappeared'], maxDistance = settings['maxDistance'])

trackableObjects = {}

totalUp = 0
totalDown = 0

objects_moving_up = 0
objects_moving_down = 0

status = "Waiting"

latencies = []
fpses = []

# Create CUDA stream
stream = cuda.Stream()

print("\"Start loading video\"")

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    
    # Check time for loading video
    start_t = timeit.default_timer()

    # Define two points for the line
    p1 = Point(0, frame.shape[0] // 2)
    p2 = Point(frame.shape[1], frame.shape[0] // 2)

    # Create a Polygon from these points
    roi = Polygon([(p1.x, p1.y-120), (p2.x, p2.y-120), (p2.x, p2.y+120), (p1.x, p1.y+120)])

    # Preprocessing images
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (settings['img_size'], settings['img_size']))
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

    # Synchronize CUDA stream
    stream.synchronize()

    # Convert h_output to numpy array
    h_output = h_output.astype(np.float32)
    
    # Apply NMS
    pred = non_max_suppression(h_output, conf_thres=settings['conf_thres_nms'], iou_thres=settings['iou_thres_nms'], classes=None, agnostic=None)

    for i, det in enumerate(pred): # Detections per image
        if len(det) != 0:
            if status == "Waiting":
                status = "Tracking"
        else:
            if status != "Waiting":
                status = "Waiting"

        # Rescale boxes from img_size to img.shape
        det[:, :4] = scale_boxes((settings['img_size'], settings['img_size']), det[:, :4], frame.shape)
        
        # Update our centroid tracker using the computed set of bounding box rectangles
        objects = ct.update(det[:, :4])

        # Loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # Check if a trackable obejct exists for the current object ID
            to = trackableObjects.get(objectID, None)

            # If there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # Otherwise, there is a trackable object so we can utilize it
            else:
                # The difference between the x-coordinate of the *current* and *previous* centroid will tell us in which direction the object is moving
                y = [c[1] for c in to.centroids]
                direction_y = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # Check if the object has been counted or not
                if not to.counted:
                # Create a Point object from the centroid
                    point = Point(centroid[0], centroid[1])
                    within = point.within(roi)

                    if direction_y < 0 and within:
                        totalUp += 1
                        to.counted = True

                    elif direction_y > 0 and within:
                        totalDown += 1
                        to.counted = True

            # Store the trackable object in our dictionary
            trackableObjects[objectID] = to
        
            # Draw both the ID of the object and the centroid of the object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0], centroid[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Draw bounding boxes
            for *xyxy, conf, cls in reversed(det):
                if conf > settings['conf_thres_bbox']: # set confidence threshold
                    
                   # Get class name
                    class_name = class_names[int(cls)]

                    # Get class color
                    color = class_colors[int(cls)]

                    # Adjust the coordinates based on the ratio
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # Calculate the center of the bounding box
                    x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                    
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

        # Draw the vertical line
        cv2.line(frame, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 255), 2)
        # cv2.rectangle(frame, (int(p1.x-750), int(p1.y)), (int(p2.x + 750), int(p2.y)), (0,255, 255), 2)
        
        end_t = timeit.default_timer()

        elapsed_time = end_t - start_t
        latency = elapsed_time * 1000
        fps = 1 / elapsed_time

        latencies.append(latency)
        fpses.append(fps)

        # Calculate average latency ans fps
        avg_latency = np.mean(latencies)
        avg_fps = np.mean(fpses)

        # Calculate the total number of tracked objects
        total_tracked_objects = len(trackableObjects)

        # Calculate the total number of counts
        total_object_counts = totalUp + totalDown
        
        # Calculate the total number of loss counts
        object_counts_loss = total_tracked_objects - total_object_counts

        # Calculate the tracking accuracy for left and right
        tracking_accuracy =  total_object_counts / total_tracked_objects * 100 if total_tracked_objects != 0 else 0
        tracking_accuracy_up = totalUp / objects_moving_up * 100 if objects_moving_up != 0 else 0
        tracking_accuracy_down = totalDown / objects_moving_down * 100 if objects_moving_down != 0 else 0

        for to in trackableObjects.values():
            if len(to.centroids) >= 2:
                # Check the direction of object movement based on centroid coordinates
                if to.centroids[-1][1] < to.centroids[-2][1]: # Moving Up
                    if to.disappeared <= settings['maxDisappeared']:
                        objects_moving_up += 1
                        to.disappeared = settings['maxDisappeared'] + 1 # Set disappeared to a value greater than maxDisappeared 
                elif to.centroids[-1][1] > to.centroids[-2][1]: # Moving Down
                    if to.disappeared <= settings['maxDisappeared']:
                        objects_moving_down += 1
                        to.disappeared = settings['maxDisappeared'] + 1 # Set disappeared to a value greater than maxDisappeared 

        info = [
            ("Status", status),
            ("Average Latency", "{:.2f}ms".format(avg_latency)),
            ("Average FPS", "{:.2f}fps".format(avg_fps)),
            ("Total Tracked Objects", total_tracked_objects),
            ("Total Object Counts", total_object_counts),
            ("Objects Count loss", object_counts_loss),
            ("Objects Moving Up", objects_moving_up),
            ("Objects Moving Down", objects_moving_down),
            ("Counting Accuracy", "{:.2f}%".format(tracking_accuracy)),
            ("Counting Accuracy Up", "{:.2f}%".format(tracking_accuracy_up)),
            ("Counting Accuracy Down", "{:.2f}%".format(tracking_accuracy_down)),
            ("Up", totalUp),
            ("Down", totalDown),   
        ]

        # Loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            # Seperate key and value into different variables
            key_text = "{} : ".format(k)
            value_text = str(v)

            # For keys, use color: Green(0. 255, 0)
            # For values, use color: Red(0, 0, 255)
            key_text_width = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            value_text_width = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            
            # For "Status", place it in the top left corner
            if i == 0: # Status
                cv2.putText(frame, key_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Latency", place it in the top left corner
            if i == 1: # Latency
                cv2.putText(frame, key_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "FPS", place it in the top left corner
            if i == 2: # FPS
                cv2.putText(frame, key_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # For "Total Tracked Objects", place it in the top left corner
            if i == 3: # Total Tracked Objects
                cv2.putText(frame, key_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Total Object Counts", place it in the top left corner
            if i == 4: # Total Object Counts
                cv2.putText(frame, key_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    

            # For "Objects Count loss", place it in the top left corner
            if i == 5: # Objects Count loss
                cv2.putText(frame, key_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # For "Objects Moving Left", place it in the top left corner
            if i == 6: # Objects Moving Up
                cv2.putText(frame, key_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Objects Moving Right", place it in the top left corner
            if i == 7: # Objects Moving Down
                cv2.putText(frame, key_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Counting Accuracy", place it in the top left corner
            if i == 8: # Counting Accuracy
                cv2.putText(frame, key_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Counting Accuracy_Left", place it in the top left corner
            if i == 9: # Counting Accuracy Up
                cv2.putText(frame, key_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Counting Accuracy_Right", place it in the top left corner
            if i == 10: # Counting Accuracy Down
                cv2.putText(frame, key_text, (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (10 + key_text_width, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Up", we place it in the bottom left corner
            if i == 11: # Up
                cv2.putText(frame, key_text, (frame.shape[1] - key_text_width - value_text_width - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (frame.shape[1] - value_text_width - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # For "Down", we place it in the bottom left corner
            if i == 12: # Down
                cv2.putText(frame, key_text, (frame.shape[1] - key_text_width - value_text_width - 20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, value_text, (frame.shape[1] - value_text_width - 20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create a named window and resize it
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280, 720)

    # Show video
    cv2.imshow('frame', frame)

    # Write the resulting frame to the output file
    out.write(frame)

    # Save info to a text file
    with open(settings['result_info'], 'w') as f:
        f.write('Test Results for People Counting\n\n')
        f.write('GPU Information\n')
        f.write(f"- GPU name : {gpu_names}\n\n")
        f.write('Resolution: {} x {}\n\n'.format(width, height))
        f.write('maxDisappeared: {}, maxDistance: {}\n\n'.format(settings['maxDisappeared'], settings['maxDistance']))
        f.write('confidence_thres: {}, IoU_thres: {}\n\n'.format(settings['conf_thres_nms'], settings['iou_thres_nms']))
        for (k, v) in info[1:]:
            f.write(f"{k} : {v}\n")
     
    # Delay between frame
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

print("\"End Work\"")

# Stop all process
vid.release()

out.release()

cv2.destroyAllWindows()

# Clean up CUDA context
cuda_ctx.pop()