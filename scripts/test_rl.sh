cd ../src

workers=6
# python main.py -c 0 --task 0 --max_epochs 30 --basemodel yolov8n --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --max_det 100

# python main.py -c 0 --task 0 --val_split 0.5 --max_epochs 5 --basemodel yolov8n --req_loss 0 -workers $workers
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 10 --reinforcement-loss True -workers $workers --no_augment
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 0 -workers $workers

# python main.py -c 0 --task 3 --max_epochs 20 --basemodel yolov8n --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 100

python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300
python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 10 --reinforcement-loss True -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8n --req_loss 10 -workers $workers --req-type frank --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300

# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++
# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 0.1 --reinforcement-loss True -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++
# python main.py -c 0 --task 4 --max_epochs 30 --basemodel yolov8l --req_loss 10 --reinforcement-loss True -workers $workers --optimizer SGD --lr 0.001 --dataset road++ --dataset_path ../../ROAD++

# python main.py -c 0 --task 3 --max_epochs 20 --basemodel yolov8l --req_loss 0 -workers $workers --optimizer SGD --lr 0.001
# python main.py -c 0 --task 3 --max_epochs 100 --basemodel yolov8l --req_loss 1 --reinforcement-loss True -workers $workers -lr 0.001

