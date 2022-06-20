import io
import os
import cv2
import json
import numpy as np
import psycopg2
import requests
import threading
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify, make_response
from insightface.app import FaceAnalysis
from PIL import Image as im
from requests import RequestException, ConnectionError

from get_image import get_image as ins_get_image
from load_image import load_image


HOSTNAME = '0.0.0.0'
PORT = 6337
FEATURE_EXTRACTION_URL = 'http://127.0.0.1:5000/extract_feature'
DIR_PATH = "dataset/"
SAMPLE_FACE_VECTOR_DATABASE = []
FEATURE_EXTRACT_BATCH_SIZE = 10

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
EXTRACT_SAMPLE_VECTOR_OK = 200
EXTRACT_SAMPLE_VECTOR_ERR = 201
UPDATE_SAMPLE_FACES_OK = 202
UPDATE_SAMPLE_FACES_ERR = 203
FEATURE_EXTRACTION_SERVER_CONNECTION_ERR = 204
FEATURE_EXTRACTION_REQUEST_ERR = 205
FEATURE_EXTRACTION_SERVER_RESPONSE_OK = 206
FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR = 207
DB_CONNECTION_ERR = 208
FACE_DETECTION_OK = 210
FACE_DETECTION_ERR = 211
NO_FACE_DETECTED_ERR = 212
CALC_DISTANCE_OK = 220
CALC_DISTANCE_ERR = 221
NO_SAMPLE_VECTOR_ERR = 222
GET_SAMPLE_VECTOR_ERR = 223
NO_SUCH_FILE_ERR = 230
INVALID_REQUEST_ERR = 231
INVALID_IMAGE_ERR = 232
UNKNOWN_ERR = 500

ERR_MESSAGES = {
    IMAGE_PROCESS_OK: 'The image is processed successfully.',
    IMAGE_PROCESS_ERR: 'The image process has been failed.',
    UPDATE_SAMPLE_FACES_OK: 'Sample vector database has been updated successfully.',
    UPDATE_SAMPLE_FACES_ERR: 'Failed to update the sample vector database.',
    FEATURE_EXTRACTION_SERVER_CONNECTION_ERR: 'Feature extraction node is not running.',
    FEATURE_EXTRACTION_REQUEST_ERR: 'Bad request to feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_OK: 'Successfully received a response from feature extraction node.',
    FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR: 'Failed to parse a response from feature extraction node.',
    DB_CONNECTION_ERR: 'Database Connection Error',
    FACE_DETECTION_OK: 'Faces are successfully detected from the input image.',
    FACE_DETECTION_ERR: 'Failed to detect face from the input image',
    NO_FACE_DETECTED_ERR: 'No face detected from the input image.',
    CALC_DISTANCE_OK: 'Calculation of vector distance has been suceeded.',
    CALC_DISTANCE_ERR: 'Failed to calculate the vector distance.',
    NO_SAMPLE_VECTOR_ERR: 'There is no sample face data.',
    GET_SAMPLE_VECTOR_ERR: 'Couldn\'t get sample data, it seems like there is no database connection.',
    NO_SUCH_FILE_ERR: 'No such file.',
    INVALID_REQUEST_ERR: 'Invalid request.',
    INVALID_IMAGE_ERR: 'Invalid image has input. Could not read the image data.',
    UNKNOWN_ERR: 'Unknown error has occurred.'
}

app = Flask(__name__)
model = None
db_connection = None


def load_model():
    """
    Load Model
    """
    global model

    # use FaceAnalysis
    model = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
    model.prepare(ctx_id=0, det_size=(640, 640))


def get_env(key):
    try:
        #global env_path
        #load_dotenv(dotenv_path=env_path,override=True)
        load_dotenv(override=True)
        val = os.getenv(key)
        return val

    except :
        return None


def set_env(key, value):
    global env_path
    if key :
        if not value:
            value = '\'\''
        cmd = f'dotenv set -- {key} {value}'  # set env variable
        os.system(cmd)


def set_db_connection():
    """
    Set Database Connection
    """
    global db_connection

    load_dotenv(override=True)
    host = get_env('db_host')
    port = get_env('db_port')
    db_name = get_env('db_name')
    db_username = get_env('db_username')
    db_password = get_env('db_password')

    if host is None or db_name is None or db_username is None or db_password is None:
        return False

    if port is None:
        port = 5432     # 5432 as default port of postgres

    try:
        db_connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=db_username,
            password=db_password
        )
    except Exception as e:
        print(e)
        return False

    return True

def detect_faces(img):
    """
    Detect the faces from the base image
    """
    global model

    if model is None:
        load_model()

    if isinstance(img, str):
        if len(img) == 0:
            return None
        elif len(img) > 11 and img[0:11] == "data:image/":
            img = load_image(img)
        else:
            img = load_image(DIR_PATH + img)

    elif isinstance(img, bytes):
        img = np.array(im.open(io.BytesIO(img)))

    else:
        return None

    if not isinstance(img, np.ndarray):
        return None

    # faces stores list of bbox and kps
    faces = model.get(img)  # [{'bbox': [], 'kps': []}, {'bbox': [], 'kps': []}, ...]

    # no face is detected
    if len(faces) == 0:
        return None

    return img, faces


def call_feature_extractor(face_list):
    """
    Send request to feature extraction node. Request will contain list of face ids and detected face image
    Returns error code, and result string
    """
    success_feature_vectors = []
    failure_feature_vectors = []

    try:
        start_time = datetime.now()
        face_data = []
        image_files = []
        for f in face_list:
            face_data.append({
                'id': f['id']
            })

            # Convert the numpy array to bytes
            face_pil_img = im.fromarray(f['img'])
            byte_io = io.BytesIO()
            face_pil_img.save(byte_io, 'png')
            byte_io.seek(0)

            image_files.append((
                'image', byte_io
            ))

        # Send request to feature extraction node
        response = requests.post(FEATURE_EXTRACTION_URL, data={'face_data': json.dumps(face_data)}, files=image_files)

        # Parse the response and get the feature vectors
        try:
            feature_list = json.loads(response.text)

            # Determine which one is success, which one is failure
            for fe in feature_list:
                if len(fe['vector']) == 0:  # If feature extraction is failed
                    failure_feature_vectors.append({
                        'id': fe['id']
                    })
                else:   # If feature extraction is suceed
                    success_feature_vectors.append({
                        'id': fe['id'],
                        'vector': np.array(fe['vector'])
                    })
            
            print('log 1.1 ----------------- ', datetime.now() - start_time)

        except:
            return FEATURE_EXTRACTION_SERVER_RESPONSE_PARSE_ERR, None, None

    except ConnectionError:
        return FEATURE_EXTRACTION_SERVER_CONNECTION_ERR, None, None

    except RequestException:
        return FEATURE_EXTRACTION_REQUEST_ERR, None, None

    return FEATURE_EXTRACTION_SERVER_RESPONSE_OK, success_feature_vectors, failure_feature_vectors


def feature_extraction_thread(face_list, extract_success_list, extract_failure_list):
    """
    Call feature extraction module. this function will be run in multi-threads
    """
    # Prepare the MetaData's Map, this will be useful to determine which faces are success to extract features, and failure
    metadata_map = {}
    for f in face_list:
        metadata_map[f['id']] = f

    # Call API of feature extraction server
    res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)
            
    if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
        # Add all faces to failed list
        for f in face_list:
            extract_failure_list.append({
                'id': f['id']
            })
        return

    # Treat the success faces, add meta data
    for face in success_face_features:
        # If could not find meta data of this face, move it to failed list
        if face['id'] not in metadata_map:
            failure_face_features.append(face)
            continue

        # Add meta data
        meta_data = metadata_map[face['id']]
        face['name'] = meta_data['name']
        face['metadata'] = meta_data['metadata']
        face['action'] = meta_data['action']

    # Append to result arrays
    extract_success_list += success_face_features
    extract_failure_list += failure_face_features


def extract_sample_feature_vector(data_list):
    """
    Extract the feature vector from the sample images
    Return code, extract_success_list, extract_failure_list
    """
    face_list = []
    extract_success_list = []
    extract_failure_list = []
    thread_pool = []

    # Main loop, each element will contain one image and its metadata
    for data in data_list:
        try:
            sample_id = data['id']
            name = data['name']
            img = data['image']
            metadata = data['metadata']
            action = data['action']
        except:
            return INVALID_REQUEST_ERR, None, None

        # Detect face from sample image
        base_img, detected_faces = detect_faces(img)

        # No face detected
        if detected_faces is None:
            continue

        # Get the first face from the detected faces list. Suppose that the sample image has only 1 face
        face = detected_faces[0]    # {'bbox': [x1, y1, x2, y2], 'kps': []}

        # # Get face region from the base image(profile image)
        bbox = face['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        face_img = base_img[y1:y2,x1:x2]

        face_list.append({
            'id': sample_id,
            'img': face_img,
            'name': name,
            'metadata': metadata,
            'action': action
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
            th.start()
            thread_pool.append(th)

            face_list = []

    if len(face_list) > 0:
        th = threading.Thread(target=feature_extraction_thread, args=(face_list, extract_success_list, extract_failure_list))
        th.start()
        thread_pool.append(th)

    # Wait until all threads are finished
    for th in thread_pool:
        th.join()

    return EXTRACT_SAMPLE_VECTOR_OK, extract_success_list, extract_failure_list


def save_sample_database(sample_vectors):
    """
    Save the sample face feature vector into the database
    """
    global db_connection

    if db_connection is None:
        res = set_db_connection()
        if not res:
            return DB_CONNECTION_ERR

    cur = db_connection.cursor()

    # Run query
    for vector_data in sample_vectors:
        sample_id = vector_data['id']
        name = vector_data['name']
        metadata = vector_data['metadata']
        action = vector_data['action']
        vector = vector_data['vector']

        # Delete original vector
        sql_query = f'DELETE FROM sample_face_vectors WHERE sample_id = \'{sample_id}\';'
        cur.execute(sql_query)

        # Save new vector
        sql_query = f'INSERT INTO sample_face_vectors (sample_id, name, metadata, action, vector) ' \
            f'VALUES (\'{sample_id}\', \'{name}\', \'{metadata}\', \'{action}\', \'{json.dumps(vector.tolist())}\');'
        cur.execute(sql_query)

    db_connection.commit()

    cur.close()

    return UPDATE_SAMPLE_FACES_OK


def get_sample_database():
    """
    Read sample feature vector from database
    """
    global db_connection
    sample_vectors = []
    
    if db_connection is None:
        res = set_db_connection()
        if not res:
            return None

    cur = db_connection.cursor()
    cur.execute('SELECT * FROM sample_face_vectors')
    sample_vector_list = cur.fetchall()


    for vector_data in sample_vector_list:
        vector = np.array(json.loads(vector_data[7]))
        sample_vectors.append({
            'id': vector_data[3],
            'name': vector_data[4],
            'metadata': vector_data[5],
            'action': vector_data[6],
            'vector': vector
        })

    cur.close()
        
    return sample_vectors


def calculate_simulation(feat1, feat2):
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def find_face(face_feature_vectors, min_simulation):
    """
    Find the closest sample by comparing the feature vectors
    """
    # Read sample database
    sample_vectors = get_sample_database()

    if sample_vectors is None:
        return GET_SAMPLE_VECTOR_ERR, None

    if len(sample_vectors) == 0:
        return NO_SAMPLE_VECTOR_ERR, None

    candidates = []
    for vector_data in face_feature_vectors:
        face_feature_vector = vector_data['vector']

        # Initialize variables
        closest_id = ''
        closest_name = ''
        closest_metadata = ''
        closest_simulation = -1
        
        # Compare with sample vectors
        for i in range(len(sample_vectors)):
            sample = sample_vectors[i]
            sample_vector = sample['vector']

            try:
                # Calculate the distance between sample and the detected face.
                simulation = calculate_simulation(face_feature_vector, sample_vector)
            
                if (closest_id == '' or simulation > closest_simulation) and simulation > min_simulation:
                    closest_simulation = simulation
                    closest_id = sample['id']
                    closest_name = sample['name']
                    closest_metadata = sample['metadata']

            except Exception as e:
                print(e)
                pass
        
        # If not find fit sample, skip
        if closest_id == '':
            continue

        # Add candidate for this face
        candidates.append({
            'id': closest_id,
            'name': closest_name,
            'metadata': closest_metadata,
            'bbox': vector_data['bbox']
        })
        
    return CALC_DISTANCE_OK, candidates


def process_image(img, min_distance):
    """
    Face recognition
    """
    # Detect the faces from the image that is dedicated in the path or bytes
    start_time = datetime.now()
    try:
        base_img, faces = detect_faces(img)
        print('log 1 ------ ', datetime.now() - start_time)
    except:
        return FACE_DETECTION_ERR, None

    if len(faces) == 0:
        return NO_FACE_DETECTED_ERR, None

    bound_box_map = {}
    face_list = []
    face_feature_vector_list = []

    # Send request to feature_extraction module
    for i in range(len(faces)):
        face = faces[i]     # [{'bbox': [], 'kps': []}, {'bbox': [], 'kps': []}, ...]

        bbox = face['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        w = x2 - x1
        h = y2 - y1
        face_img = base_img[y1:y2,x1:x2]

        # Prepare bound box map
        bound_box_map[i] = f'{x1}, {y1}, {w}, {h}'

        # Make the face list, I will send bunch of faces to Feature Extraction Server at once
        face_list.append({
            'id': i,
            'img': face_img
        })

        if len(face_list) == FEATURE_EXTRACT_BATCH_SIZE:
            # Call the api to extract the feature from the detected faces
            res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

            if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
                return res_code, None

            face_feature_vector_list += success_face_features

            face_list = []

    print('log 2 ------ ', datetime.now() - start_time)

    if len(face_list) > 0:
        # Call the api to extract the feature from the detected faces
        res_code, success_face_features, failure_face_features = call_feature_extractor(face_list)

        if res_code != FEATURE_EXTRACTION_SERVER_RESPONSE_OK:
            return res_code, None

        face_feature_vector_list += success_face_features

    print('log 3 ------ ', datetime.now() - start_time)
    
    # Add bound box for each face feature vector
    vector_list = []
    for f in face_feature_vector_list:
        if int(f['id']) not in bound_box_map:
            continue
        
        f['bbox'] = bound_box_map[int(f['id'])]
        vector_list.append(f)

    # Find candidates by comparing feature vectors between detected face and samples
    status, candidates = find_face(vector_list, min_distance)

    print('log 4 ------ ', datetime.now() - start_time)

    if status != CALC_DISTANCE_OK:
        return status, None

    return IMAGE_PROCESS_OK, candidates


def update_sample_database(data_list):
    """
    Update the database that contains sample face vectors
    """
    # Extract the feature vector
    res, success_sample_vectors, failure_sample_vectors = extract_sample_feature_vector(data_list)

    if res != EXTRACT_SAMPLE_VECTOR_OK:
        return res, success_sample_vectors, failure_sample_vectors

    # Save the sample feature vector into database
    res = save_sample_database(success_sample_vectors)

    return res, success_sample_vectors, failure_sample_vectors


@app.route('/', methods=['GET'])
def welcome():
    """
    Welcome page
    """
    return Response("<h1 style='color:red'>Face detection server is running!</h1>", status=200)


@app.route('/config_database', methods=['GET'])
def config_database():
    """
    Set the database connection
    """
    data = request.get_json()

    if 'host' not in data:
        return Response('"host" is missing', status=400)

    port = 5432
    if 'port' not in data:
        port = data['port']
    
    if 'db_name' not in data:
        return Response('"db_name" is missing', status=400)

    if 'username' not in data:
        return Response('"username" is missing', status=400)
    
    if 'password' not in data:
        return Response('"password" is missing', status=400)

    host = data['host']
    db_name = data['db_name']
    username = data['username']
    password = data['password']

    set_env('db_host', host)
    set_env('db_port', port)
    set_env('db_name', db_name)
    set_env('db_username', username)
    set_env('db_password', password)

    set_db_connection()

    return Response('Database configuration is done', status=200)


@app.route('/init_database', methods=['GET'])
def init_database():
    if db_connection is None:
        return Response('Database is not configured yet', status=500)

    cur = db_connection.cursor()
    query = f"SELECT 1 FROM information_schema.tables WHERE table_name = 'sample_face_vectors';"
    cur.execute(query)
    tables = cur.fetchall()

    if tables is None:
        query = "DROP TABLE IF EXISTS sample_face_vectors; \
                CREATE TABLE sample_face_vectors ( \
                    id SERIAL PRIMARY KEY, \
                    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
                    modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
                    sample_id TEXT NOT NULL, \
                    name TEXT NOT NULL, \
                    metadata TEXT, \
                    action TEXT, \
                    vector TEXT NOT NULL \
                );"
        cur.execute(query)
        db_connection.commit()

    cur.close()

    return Response('Database has been initialized successfully.', status=200)


@app.route('/update-samples', methods=['GET', 'POST'])
def update_samples():
    # GET request
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST request
    data_list = request.json

    ## Try to extract features from samples, and update database
    res_code, success_list, failure_list  = update_sample_database(data_list)
    if res_code != UPDATE_SAMPLE_FACES_OK:
        response = {
            'error': ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 400)

    ## Make response
    response = {
        'success': [f['id'] for f in success_list],
        'fail': [f['id'] for f in failure_list]
    }
    return make_response(jsonify(response), 200)


@app.route('/clear-samples', methods=['GET', 'POST'])
def clear_samples():
    try:
        if db_connection is None:
            return Response('', status=200)

        cur = db_connection.cursor()
        sql_query = 'DELETE FROM sample_face_vectors;'
        cur.execute(sql_query)

        db_connection.commit()
        
        cur.close()

        response = {
            'success': 'Samples have been removed successfully'
        }
        
        return make_response(jsonify(response), 200)
    
    except Exception as e:
        response = {
            'fail': str(e)
        }
        return make_response(jsonify(response), 500)


@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'GET':
        return Response('Face detection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': ERR_MESSAGES[INVALID_REQUEST_ERR]
        }
        return make_response(jsonify(response), 400)

    # min_distance is optional parameter in request
    min_distance = 0.3  # default threshold for facenet, between 0 and 1
    if 'min_distance' in img_data:
        min_distance = float(img_data['min_distance'])

    # Process image
    res_code, candidates = process_image(img_data['image'], min_distance)

    if res_code != IMAGE_PROCESS_OK:
        response = {
            'error': ERR_MESSAGES[res_code]
        }
        return make_response(jsonify(response), 500)

    # Return candidates
    return make_response(jsonify(candidates), 200)


# if __name__ == '__main__':
#     # Run app in debug mode on port 6337
#     app.run(debug=True, host='0.0.0.0', port=6337, threaded=True)