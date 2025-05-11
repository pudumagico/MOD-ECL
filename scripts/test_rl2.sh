cd ../src

workers=6

# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.0_vanilla_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product

# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.1_product_rsched0/train --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss1.0_product_rsched0/train3 --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 1000 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss1000.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.0_vanilla_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type schweizer_sklar
# # python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_schweizer_sklar_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_frank_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type yager
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_yager_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno_weber
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_sugeno_weber_rsched0/train --threshold 0.05 --pred --stats -c 1

# # python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_aczel_alsina_rsched0/train --threshold 0.05 --pred --stats -c 1


# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_schweizer_sklar_rsched0/train2 --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type yager
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno_weber
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type dombi
# python main.py -c 1 --task 4 --max_epochs 50 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina



# python main.py -c 1 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 0.5 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss0.5_product_rsched0/train7 --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 5 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss5.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss50.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 500 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 5000 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss5000.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1

# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_product_rsched0/train --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss1.0_product_rsched0/train3 --threshold 0.05 --pred --stats -c 1
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.1_product_rsched0/train --threshold 0.05 --pred --stats -c 1



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.25 -rl --rl_mode "pglhpnmd_tnorm"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.1_delta0.25_modepglhpnmd_tnorm_rsched0/train3 --threshold 0.05 --pred --stats -c 0

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.5 -rl  --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.25_delta0.5_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.5 -delta_rl 0.5 -rl  --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.5_delta0.5_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 1



# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type scheizer_sklar
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_schweizer_sklar_rsched0/train --threshold 0.05 --pred --stats -c 1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type sugeno_weber
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_sugeno_weber_rsched0/train --threshold 0.05 --pred --stats -c 1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type frank
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_frank_rsched0/train --threshold 0.05 --pred --stats -c 1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type yager
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_yager_rsched0/train --threshold 0.05 --pred --stats -c 1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_aczel_alsina_rsched0/train2 --threshold 0.05 --pred --stats -c 1


# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.5 -rl  --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.25_delta0.5_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.5 -delta_rl 0.5 -rl  --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.5_delta0.5_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 1

# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.5 -rl --rl_mode "all_loss"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.1_delta0.5_modeall_loss_rsched0/train3 --threshold 0.05 --pred --stats -c 1
# python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.25 -rl --rl_mode "all_loss"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.1_delta0.25_modeall_loss_rsched0/train3 --threshold 0.05 --pred --stats -c 1



python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rs 1.1
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_product_rsched1.1/train --threshold 0.05 --pred --stats -c 1
python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rs 1.25
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_product_rsched1.25/train --threshold 0.05 --pred --stats -c 1
python main.py -c 1 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rs 1.5
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_product_rsched1.5/train --threshold 0.05 --pred --stats -c 1