cd ../src

workers=30
rl=0.1

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum --req_num_detect -1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type yager --req_num_detect -1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank --req_num_detect -1
# # python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type dombi

# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_minimum_rsched0/train2 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_yager_rsched0/train2 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_frank_rsched0/train2 --conf 0.01 -pred -stats


# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno_weber --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_sugeno_weber_rsched0/train2 --conf 0.05 -pred -stats

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_aczel_alsina_rsched0/train --conf 0.05 -pred -stats



# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum --req_num_detect -1 -rlr 0.05
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_minimum_rlr0.05_rsched0/train --conf 0.05 -pred -stats


python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0 -rs 1.75
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss50.0_lukasiewicz_rsched1.75/train --conf 0.05 -pred -stats

python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0 -rs 2
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss50.0_lukasiewicz_rsched2/train --conf 0.05 -pred -stats

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0.1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.0_lukasiewicz_rlr0.1_rsched0/train --conf 0.05 -pred -stats
