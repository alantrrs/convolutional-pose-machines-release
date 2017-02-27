FROM alantrrs/cuda-opencv-caffe:nvidia

# Tools
RUN apt-get install -y vim

# Install Jupyter
RUN pip install --upgrade pip
RUN pip install jupyter

# Set workdir
WORKDIR /conv-pose
