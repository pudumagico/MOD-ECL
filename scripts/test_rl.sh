cd ../src

workers=6

python main.py -c 0 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rl