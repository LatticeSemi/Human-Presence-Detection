1. Training Environment Setup Using Docker
------------------------------------------
Please refer "installdocker.txt" to install docker based on operating system.
Please refer "cuda/README.txt" to use GPU 
To build docker image: (This wil take around 30-40 minutes)
  For Linux:
  ----------
     - Open terminal
     - cd iCE40_humandet_training
     - sudo docker build -t human_det .
  For Windows:
  ------------
     - Open Docker Quickstart terminal
     - cd iCE40_humandet_training
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
	  - $cd iCE40_humandet_training
	  - $python input.py
		arguments:
		   - shared folder(mentioned in above step)
		   - Docker image name(human_det) that has been built
	  - Docker shell will open from the script.
	For Windows:
	------------
	   - Open the Docker Quickstart terminal.
		- Make sure that Xlaunch to use display in docker. If it is not installed, please refer "installdocker.txt"
          - $cd iCE40_humandet_training
	  - $export DISPLAY=<Machine IP address>:0.0 #Same as mentioned in link: https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde 
          - $docker run --rm -it -p 8888:8888 -p 6006:6006 -v <path to shared folder>:<path to shared folder> -e DISPLAY=$DISPLAY --net=host human_det

2. There are two ways one can trigger training:
    1. Using Jupyter notebook GUI
	- To use GUI based training run command prompted in the docker shell to lauch jupyter notebook
        - For linux, jupyter notebook will run at "http://localhost:8888" and tensorboard will run at "http://localhost:6006" in the host browser.
        - For windows, jupyter notebook will run at "http://192.168.99.100:8888" and tensorboard will run at "http://192.168.99.100:6006" in the host browser. 192.168.99.100 is docker-machine default ip. Use command "docker-machine ip default" to get it.
	- Token id should be given which is available in the jupyter notebook link.
	- Open train.ipynb file
	- After notebook is opened, click "Kernel-> Restart & Run All" option from menubar  to load the GUI
	- The final .pb is copied to shared_folder/train_logs to access it from host machine. After that it can be used in SensAI tool.

    2. Using Docker shell
	- $python src/train.py --dataset 'KITTI' --net 'squeezeDet' --data_path <dataset_dir> --train_dir <train_logs_dir> --image_set 'train' --summary_step 100 --max_steps 250000 --checkpoint_step 500
        - $python src/genpb.py --ckpt_dir <train_logs_dir>
	- $tensorboard --logdir=<train_logs_dir>
	- To generate .pb file from latest checkpoint:
		- ckpt = tf.train.latest_checkpoint(<train_logs_dir>)
		- $rm -rf model; mkdir model
		- $cp $ckpt* model/
		- $cp <train_logs_dir>/graph.pbtxt model/model.pbtxt
		- $python trainckpt2inferencepb.py
		- $cp model/*.pb <train_log_dir>/


4. Hint and troubleshotting:
----------------------------
 - If have access error during training, give permissions to the shared folder to resolve the issue.
