# face_recognition_v2
1. ```conda create -n face-recognition-v2 python=3.8```
2. ```conda activate face-recognition-v2```
3. ```pip install --upgrade pip```
4. ```sudo-apt install libpq-dev```
5. ```pip install -r requirements.txt```
6. ```pip install -r requirements_dev.txt```  
  Please pay attention to the version of onnxruntime-gpu. Please install suitable version of onnxruntime according to the versions of cuda and cudnn,you can find the table by the link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
7. ```pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```
