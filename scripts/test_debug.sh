cd ../src

workers=6

# python main.py -c 0 --task 0 --max_epochs 50 --basemodel yolov8n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product -rl
python main.py -c 0 --task 0 --val_split 0.5 --max_epochs 5 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product -rl