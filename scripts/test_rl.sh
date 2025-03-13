cd ../src

workers=6
rl=0.1

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1
# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type drastic
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss $rl -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type nilpotent_minimum --req_num_detect -1


# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_product_rsched0/train2 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_lukasiewicz_rsched0/train2 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_nilpotent_minimum_rsched0/train2 --conf 0.01 -pred -stats



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1000 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 5000 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss5000.0_product_rsched0/train2 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss5000.0_product_rsched0/train2 --conf 0.05 -pred -stats


# python -m torch.distributed.run --nproc_per_node 2  main.py -c 0,1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1000 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# # python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 1 --req_num_detect -1 --req-type product
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1000.0_product_rsched0/train3 --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1000.0_product_rsched0/train3 --conf 0.05 -pred -stats

# python -m torch.distributed.run --nproc_per_node 2  main.py -c 0,1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 2500 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# # python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 1 --req_num_detect -1 --req-type product
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss2500.0_product_rsched0/train --conf 0.01 -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss2500.0_product_rsched0/train --conf 0.05 -pred -stats


# python -m torch.distributed.run --nproc_per_node 2  main.py -c 0,1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 250 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss250.0_product_rsched0/train2 --conf 0.05 -pred -stats

# # python -m torch.distributed.run --nproc_per_node 2  main.py -c 0,1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 750 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1
# # python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss750.0_product_rsched0/train --conf 0.05 -pred -stats


# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_lukasiewicz_rsched0/train --conf 0.05 -pred -stats

# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum --req_num_detect -1
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_minimum_rsched0/train --conf 0.05 -pred -stats


# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1 -rl
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_rl_rsched0/train --conf 0.05 -pred -stats



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss -1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1 -rlr 0.05
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss-1.0_product_rsched0/train2 --conf 0.05 -pred -stats


# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss -1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product --req_num_detect -1 -rlr 0.2
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss-1.0_product_rsched0/train3 --conf 0.05 -pred -stats

python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0 -rs 1.1
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss50.0_lukasiewicz_rsched1.1/train --conf 0.05 -pred -stats

python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0 -rs 1.25
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss50.0_lukasiewicz_rsched1.25/train --conf 0.05 -pred -stats

python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0 -rs 1.5
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss50.0_lukasiewicz_rsched1.5/train --conf 0.05 -pred -stats



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz --req_num_detect -1 -rlr 0.05
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.0_lukasiewicz_rlr0.05_rsched0/train --conf 0.05 -pred -stats
