from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow.compat.v1 as tf
import numpy as np
import facenet
from skimage.transform import resize
from align import detect_face
import imageio
from io import BytesIO

app = FastAPI()

# Global variables for the model and MTCNN
pnet = None
rnet = None
onet = None
sess = None

@app.on_event("startup")
async def startup_event():
    global pnet, rnet, onet, sess
    
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    facenet.load_model('20180402-114759/20180402-114759.pb')

def load_and_align_data(image, image_size, margin):
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    img = imageio.imread(BytesIO(image))
    if img.ndim < 2:
        print('Unable to align the image!')
        return None
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0:3]
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        for det in det_arr:
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin, 0)
            bb[1] = np.maximum(det[1] - margin, 0)
            bb[2] = np.minimum(det[2] + margin, img_size[1])
            bb[3] = np.minimum(det[3] + margin, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = resize(cropped, (image_size, image_size), anti_aliasing=True)
            prewhitened = facenet.prewhiten(aligned)
            return prewhitened
    print('Unable to align image!')
    return None

def calculate_embedding(image, image_size, margin):
    image_data = load_and_align_data(image, image_size, margin)
    if image_data is not None:
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        emb_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        feed_dict = {images_placeholder: [image_data], phase_train_placeholder: False}
        embedding = sess.run(emb_tensor, feed_dict=feed_dict)
        return embedding
    return None

@app.post("/recognize/")
async def recognize(train_image: UploadFile = File(...), test_image: UploadFile = File(...)):
    image_size = 160
    margin = 16

    # Read image files asynchronously
    train_image_data = await train_image.read()
    test_image_data = await test_image.read()

    print("Calculating embeddings for training image...")
    train_embedding = calculate_embedding(train_image_data, image_size, margin)
    print("Calculating embeddings for testing image...")
    test_embedding = calculate_embedding(test_image_data, image_size, margin)

    if train_embedding is not None and test_embedding is not None:
        dist = np.sqrt(np.sum(np.square(np.subtract(test_embedding, train_embedding))))
        threshold = 0.95  # Set a threshold for recognition
        if dist < threshold:
            return JSONResponse(content={"status": "recognized"})
        else:
            return JSONResponse(content={"status": "not recognized"})
    else:
        return JSONResponse(content={"status": "error", "message": "Error in calculating embeddings"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
