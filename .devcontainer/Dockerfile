from tensorflow/tensorflow:2.5.0-gpu

RUN apt update && apt install git vim python3.8 -y
RUN pip install datasets \
    git+https://github.com/huggingface/transformers \
    torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html