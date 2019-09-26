import os

os.system("ls training/images/ | grep \".jpg\" | sed s\/.jpg\/\/ > ImageSets\/train.txt")
