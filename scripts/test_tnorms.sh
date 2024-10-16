cd ../src

workers=4
model=yolov8l
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 0 -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type godel -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 10 --basemodel yolov8l --req_loss 1 --req-type lukasiewicz
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type minimum -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type drastic -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type not_so_drastic -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type hamacher_product -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type weber -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type dubois_prade -workers $workers
# # python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 10 --basemodel yolov8l --req_loss 1 --req-type dombi
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type nilpotent_conorm -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type sugeno -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type yager -workers $workers
# python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 1 --req-type frank -workers $workers


python main.py -c 0 --task 2 --val_split 0.1 --max_epochs 20 --basemodel yolov8l --req_loss 0.1 --req-type nilpotent_minimum -workers $workers
