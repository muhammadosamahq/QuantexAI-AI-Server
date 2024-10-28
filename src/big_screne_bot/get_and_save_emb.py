from imutils import paths
import face_recognition
import pickle
import cv2
import os
from PIL import Image
 
# Directory where you want to save the cropped faces
save_dir = "/home/xloop/Downloads/high_value_customer/cropped_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('/home/xloop/Downloads/high_value_customer/total_image'))
knownEncodings = []
knownNames = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (encoding, box) in zip(encodings, boxes):
        knownEncodings.append(encoding)
        knownNames.append(name)

        # Cropping and saving the face image
        top, right, bottom, left = box
        face_image = rgb[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        face_filename = f"{save_dir}/{name}_{i}.png"
        pil_image.save(face_filename)

#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("/home/xloop/Downloads/high_value_customer/encodings/combined_images.pkl", "wb")
f.write(pickle.dumps(data))
f.close()