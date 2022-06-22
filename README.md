# face_recognition_v2
## Prerequisite
1. Install Nvidia drivers on ubuntu-18.04 machine.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.
4. Install PostgreSQL
   - After install PostgreSQL, please set password for default user 'postgresql'. Otherwise, you can create a new user with password.
   - Create database 'FaceRecognition'. You can use the name what you want.  
   This user credential and database name will be used later.
## Installation
1. ```conda create -n face-recognition-v2 python=3.8```
2. ```conda activate face-recognition-v2```
3. ```pip install --upgrade pip```
4. ```sudo-apt install libpq-dev```
5. ```pip install -r requirements.txt```
6. ```pip install -r requirements_dev.txt```  
  Please pay attention to the version of onnxruntime-gpu. Please install suitable version of onnxruntime according to the versions of cuda and cudnn,you can find the table by the link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html. If you go through this guide, you can ignore this attention.
7. ```pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```
8. ```git clone https://github.com/JamesWanglf/face_recognition_v2.git```
9. For face detection node, please download the model.
   ```
   mkdir ~/.insightface
   mkdir ~/.insightface/models
   cd ~/.insightface/models
   ```  
   Download the model file from [here](https://drive.google.com/file/d/1_RbGpfrPbgDT8MiY0FTMkP8bor33OGmq/view?usp=sharing), and unzip it.
   If you can find *.onnx files in ~/.insightface/models/buffalo_s/, it's okay.
   
10. For feature extraction node, please download the model.
    ```
    cd ./face_recognition_v2/feature-extraction
    mkdir models
    ```  
    Download the model file from [here](https://drive.google.com/file/d/1py6MWvxugYBK-4YDNNdby955nZf-hjdN/view?usp=sharing), and place it under ./feature-extraction/models/ directory.  

## Run
### Initialize the database
This will create a table named by 'sample_face_vectors' in PostgreSQL database. The database should be created in PostgreSQL first.
```
cd ./face_detection
python init_db.py -H <db address> -P <db port> -d <db name> -u <username> -p <password>
```  
E.g.  
```
python init_db.py -H 127.0.0.1 -P 5432 -d FaceRecognition -u postgres -p postgres
```
*Attention: While the face detction server is running on any nodes, you can not initialize the database again.*
### Run Feature Detection Server
This app is responsible for the feature extraction from the input image data.  
```cd ./feature_extraction```  
```gunicorn -w <number of processes> -b 0.0.0.0:5000 wsgi:app```
### Run Face Detection Server
This app is responsible for configuration of database, the face detection, save sample face features to database and comparation between the face feature with sample data.  
```cd ./face_detection```   
```gunicorn -w <number of processes> -b 0.0.0.0:6337 wsgi:app```
#### endpoints
* http://0.0.0.0:6337/config-database  
  This endpoint will set the configuration to access to the remote database.  
  - request
    ```
    curl --location --request GET 'http://0.0.0.0:6337/config-database' 
    --header 'Content-Type: application/json' 
    --data-raw '{
        "host": <host domain/ip>,
        "port": "5432",
        "db_name": "FaceRecognition",
        "username": "postgres",
        "password": "postgres"
    }'
    ```
    "port" is optional. If it's not set, 5432 will be used as default.  
  - response
    - status_code: 200
      ```
      Database configuration is done.
      ```
    - status_code: 400
      ```
      "host" is missing.
      "db_name" is missing.
      "username" is missing.
      "password" is missing.
      ```
* http://0.0.0.0:6337/clear-samples
  This endpoint will remove all sample face vectors that are saved in database.
  - request
    ```
    curl --location --request GET 'http://0.0.0.0:6337/clear-samples'
    ```
  - response
    - status_code: 200
    ```
    {
        "success": "Samples have been removed successfully"
    }
    ```
    - status_code: 500
    ```
    {
        "fail": "error_message"
    }
    ```
* http://0.0.0.0:6337/update-samples
  This endpoint will process the sample faces, including face detection and feature extraction, and save them to database.
  - request
    You can send base64-encoded image to this endpoint.
    ```
    curl --location --request POST 'http://0.0.0.0:6337/update-samples'
    --header 'Content-Type: application/json'
    --data-raw '[
    {
        "id": "person1",
        "name: "person1",
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
        "metadata": "mydomain.com/myobject1",
        "action": "embedlink"
    },
    {
        "id": "person2",
        "name: "person2",
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
        "metadata": "mydomain.com/myobject2",
        "action": "embedlink"
    }
    ]'
    ```
  - response
    - status_code: 200
    ```
    {
        "success": [all success ids],
        "fail": [list of failed ids]
    }
    ```
    - status_code: 500
    ```
    {
        "error": "Invalid request."
    }
    ```
* http://0.0.0.0:6337/face-recognition
  This endpoint will return the list of verified face information inside the posted image, including id, name, metadata and bounding box of the closest sample face for each detected faces.
  - request
    ```
    curl --location --request POST 'http://0.0.0.0:6337/face-recognition'
    --header 'Content-Type: application/json'
    --data-raw '{
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD....",
        "min_distance": 0.35
     }'
     ```
     "min_distance" is optional field. The value of this field is related with the feature extraction model.
For example, it will be in [0, 1].
Since we are using "facenet" model as default, the ideal threshold is 0.35.
  - response
    - status_code: 200
      ```
      [
          {
              "bbox": "x, y, w, h", 
              "id": "", 
              "metadata": "", 
              "name": ""
           },
           {
              "bbox": "x, y, w, h", 
              "id": "", 
              "metadata": "", 
              "name": ""
           },
       ]
       ```
    - status_code: 400
      ```
      {
          "error": "Invalid request."
      }
      ```
    - status_code: 500
      ```
      {
          "error": "error message"
      }
      ```
