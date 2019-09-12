#Whole Installation Can Take around 30 minutes need >4 G disk space
FROM ubuntu:latest
ARG arg
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN DEBIAN_FRONTEND=noninteractive apt install -y jupyter-notebook jupyter-core python-ipykernel
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN apt-get install -y wget
RUN if [ "x$arg" = "xGPU" ] ; then \
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb && dpkg -i cuda-repo-ubuntu1604_10.1.105-1_amd64.deb; \
	wget -qO - http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add - ; \
	apt-get update ; \
	DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::="--force-overwrite" install -y cuda-9-0 ; \
	echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> /etc/bash.bashrc ; \
	echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /etc/bash.bashrc ; \
    fi
########Uncomment following lines when using GPU and after copying libcudnn in Human-Count/cuda directory##########
#COPY cuda/cudnn.h /usr/local/cuda/include/
#COPY cuda/libcudnn* /usr/local/cuda/lib64/
#RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
###################################################################################################################
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y python3-tk
RUN \
  wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
  echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list && \
  apt-get update && \
  apt-get install -y google-chrome-stable && \
rm -rf /var/lib/apt/lists/*
RUN echo jupyter nbextension enable --py --sys-prefix widgetsnbextension >> /etc/bash.bashrc
RUN echo jupyter notebook --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token=\'\' \& >> /etc/bash.bashrc
RUN echo google-chrome-stable "\"http://localhost:8888/notebooks/train.ipynb\"" --no-sandbox \& >> /etc/bash.bashrc
WORKDIR /usr/src/app
RUN if [ "x$arg" = "xGPU" ] ; then \
	pip3 install tensorflow-gpu==1.12.0; \
    else \
	pip3 install tensorflow==1.12.0; \
    fi
COPY . .  
RUN pip3 install -r requirements.txt
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
