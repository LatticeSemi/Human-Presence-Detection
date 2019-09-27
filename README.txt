1. Training Environment Setup Using Docker
------------------------------------------
Please refer "installdocker.txt" to install docker based on operating system.
Please refer "cuda/README.txt" to use GPU 
To build docker image: (This wil take around 30-40 minutes)
  For Linux:
  ----------
     - Open terminal
     - cd Human-Presence-Detetction
     - With GPU
       --------
	     - sudo docker build -f ./Dockerfile_GPU -t human_det .
     - Without GPU
       -----------
	     - sudo docker build -t human_det .

  For Windows:
  ------------
     - Open Docker Quickstart terminal
     - cd Human-Presence-Detetction
     - docker build -t human_det .


2. Dataset format
-----------------
The dataset should be in the following format:
  1. Training Images:
	-- <dataset_dir>/training/image_2/*.(jpg/png)
  2. Training Labels:
	-- <dataset_dir>/training/label_2/*.txt
  3. train.txt:
	-- <dataset_dir>/ImageSets/train.txt
  4. Test Images:
	-- <dataset_dir>/test/img/*.(jpg/png)
  5. Test Label:
	-- <dataset_dir>/test/labels/*.txt


3. Shared directory between docker container and host
-----------------------------------------------------
The shared folder should contain:
  1. dataset directory in above mentioned format
  2. train_logs directory to store training logs and final .pb file
  
4. Training
-----------
1. Run docker image and select the dataset path
Follow the main step 1 to build the docker image
	For Linux:
	----------
	  - $cd Human-Presence-Detetction
          - With GPU:
	    --------
		  - $sudo docker run --rm -it -p 6006:6006 -v <path to shared folder>:<path to shared folder> --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --net=host human_det
          - Without GPU:
	    -----------
		  - $sudo docker run --rm -it -p 6006:6006 -v <path to shared folder>:<path to shared folder> --net=host human_det
	For Windows:
	------------
           - Open the Docker Quickstart terminal.
           - $cd Human-Presence-Detetction
           - $docker run --rm -it -p 6006:6006 -v <path to shared folder>:<path to shared folder> --net=host human_det
	  
2. Trigger training:
    - Using Docker shell
      ------------------
	- $python src/train.py --dataset 'KITTI' --net 'squeezeDet' --data_path <dataset_dir> --train_dir <train_logs_dir> --image_set 'train' --summary_step 100 --max_steps 250000 --checkpoint_step 500
	- $tensorboard --logdir=<train_logs_dir> (Open link in host's browser Linux: http://localhost:6006 and Windows: http://192.168.99.100:6006)
	- To generate .pb file from latest checkpoint:
		- $python src/genpb.py --ckpt_dir <train_logs_dir> --freeze

4. Hint and troubleshotting:
----------------------------
 - If have access error during training, give permissions to the shared folder to resolve the issue.
