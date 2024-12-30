cd ../src

workers=6
# python main.py -c 0 --task 0 --max_epochs 30 --basemodel yolov8n --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --max_det 100

python main.py -c 0 --task 0 --val_split 0.5 --max_epochs 5 --basemodel yolov8n --req_loss 0 -workers $workers
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 10 --reinforcement-loss True -workers $workers --no_augment
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 0 -workers $workers

# python main.py -c 0 --task 3 --max_epochs 20 --basemodel yolov8n --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 100

# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.1 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 100 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product

# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type nilpotent_minimum
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type weber
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1.0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -lrf 0.1
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type soft_godel
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 500 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect 128
# python main.py -c 0 --task 4 --max_epochs 100 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -lrf 0.005
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect 256
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.001 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.001 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
# python main.py -c 0 --task 2 --max_epochs 50 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road-r --dataset_path ../../road-dataset --max_det 300
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.001 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 0.01
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1.0 -workers $workers --optimizer Adam --lr 0.02 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 0
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.02 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 1
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.02 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 0.1
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer SGD --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 1 --remake
python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 -rs 1.05



# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 --reinforcement-loss -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
# python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 10 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8n --req_loss 10 -workers $workers --req-type frank --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++
# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++
# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 10 --reinforcement-loss True -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++

# python main.py -c 0 --task 3 --max_epochs 20 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.001
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 1 --reinforcement-loss True -workers $workers -lr 0.001

