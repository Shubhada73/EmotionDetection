
import cv2
import numpy as np
from keras.models import load_model

model = load_model('C:/Users/SHUBHADA/Documents/Data/model_file.h5')

faceDetect = cv2.CascadeClassifier('C:/Users/SHUBHADA/Documents/Data/haarcascade_frontalface_default (1).xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# List of image file paths to process
image_paths = [
    "C:/Users/SHUBHADA/Documents/Data/train/angry/im3983.png",
    "C:/Users/SHUBHADA/Documents/Data/train/angry/im3984.png",
    "C:/Users/SHUBHADA/Documents/Data/train/angry/im3985.png",
    "C:/Users/SHUBHADA/Documents/Data/train/disgusted/im0.png",
    "C:/Users\SHUBHADA/Documents/Data/train/fearful/im0.png",
    "C:/Users/SHUBHADA/Documents/Data/train/happy/im0.png",
    "C:/Users/SHUBHADA/Documents/Data/train/happy/im1.png",
    "C:/Users/SHUBHADA/Documents/Data/train/neutral/im0.png",
    "C:/Users/SHUBHADA/Documents/Data/train/sad/im0.png",
]


for image_path in image_paths:
    # Load an image for processing
    frame = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion_label = labels_dict.get(label, "Unknown")
        print(f"Predicted Label: {emotion_label}")

        # Draw rectangles and label the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
       # cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Adjust the text placement slightly below the face rectangle
        cv2.putText(frame, emotion_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #cv2.putText(frame, emotion_label, (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)



    # Display the image with detected faces and labels
    cv2.imshow("Frame", frame)

    # Wait for a key press to proceed to the next image
    cv2.waitKey(0)

# Close all windows at the end
cv2.destroyAllWindows()