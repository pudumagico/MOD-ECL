cd ../src

workers=6

python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum
python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type yager
python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank
python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno_weber
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type dombi
python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina