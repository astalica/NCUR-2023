from imutils.video import VideoStream
import time
import cv2
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
import numpy as np
from twilio.rest import Client

#Establish twilio client object
#personal info removed with '#'
client = Client("##################", "#################")

#Set up model path, establish a counter for later use
PATH_TO_SAVED_MODEL = "C:\\saved_model\\saved_model2\\saved_model"
counter = 0

# Load label map and obtain class names and ids
category_index=label_map_util.create_category_index_from_labelmap("C:\\saved_model\\label_map.pbtxt",use_display_name=True)

#function to draw bounding box on frame with label
def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

#Start camera stream
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#load model
print("Loading saved model ...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model Loaded!")

#Establish time variable
start_time = time.time()

while True:
    #read the frame
    frame = vs.read()

    frame = cv2.flip(frame, 1)
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    # The model expects a batch of images, so also add an axis with `tf.newaxis`.
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    # Pass frame through detector
    detections = detect_fn(input_tensor)

    #send the alert message if a mouse is detected
    #if its the first detection, send the message, set the time variable to current time
    #otherwise send the message in 3 min intervals
    if (detections['num_detections'] > 0):
        counter += 1
        if(counter == 1):
            now = time.time()
            client.messages.create(to="#######",
                                   from_="######",
                                   body="Mouse detected @: " + time.asctime() + "!!")
        if((now + 180) <= time.time()):
            now = time.time()
            client.messages.create(to="#########",
                                   from_="########",
                                   body="Mouse detected @: " + time.asctime() + "!!")

    # Set detection parameters
    score_thresh = 0.4  # Minimum threshold for object detection
    max_detections = 1

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # Only interested in the first num_detections.
    scores = detections['detection_scores'][0, :max_detections].numpy()
    bboxes = detections['detection_boxes'][0, :max_detections].numpy()
    labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
    labels = [category_index[n]['name'] for n in labels]

    # Display detections
    visualise_on_image(frame, bboxes, labels, scores, score_thresh)

    end_time = time.time()
    fps = int(1 / (end_time - start_time))
    start_time = end_time
    cv2.putText(frame, f"FPS: {fps}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('Frame',frame)

    #end program when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    # Write output video
    #result.write(frame)

cv2.destroyAllWindows()
vs.stop()
