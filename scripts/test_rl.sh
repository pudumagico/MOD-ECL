cd ../src

workers=6

# python main.py -c 0 --task 0 --max_epochs 50 --basemodel yolo11n --req_loss 0 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 0.5 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 5 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 50 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 500 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product

# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.5_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss5.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss50.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# t_norm_values = {
#     "product": 0.0,
#     "minimum": 0.0,
#     "hamacher_product": 0.0,
#     "schweizer_sklar": 0.0,
#     "lukasiewicz": 0.0,
#     "drastic": 0.0,
#     "nilpotent_minimum": 0.0,
#     "frank": 0.0,
#     "yager": 0.0,
#     "sugeno_weber": 0.0,
#     "aczel_alsina": 0.0,
#     "hamacher": 0.0,
# }
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 5000 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss5000.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_minimum_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_lukasiewicz_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type hamacher_product
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_hamacher_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type nilpotent_minimum
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_nilpotent_minimum_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type drastic
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_drastic_rsched0/train --threshold 0.05 --pred --stats -c 0

# # python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type hamacher
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss10.0_hamacher_rsched0/train --threshold 0.05 --pred --stats -c 0


# python main.py -c 0 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 0.1 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss0.1_product_rsched0/train7 --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss1.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 10 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss10.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss100.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 2 --max_epochs 10 --basemodel yolo11n --req_loss 1000 -workers $workers --optimizer Adam --lr 0.01 --max_det 300 --req-type product
# python tester.py --model nparam/task2_e10_yolo11n_reqloss1000.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type drastic
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 1 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type nilpotent_minimum

# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss50.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss5.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss0.5_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss5000.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0


# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam_ijcai/task4_e10_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam_ijcai/task4_e10_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam_ijcai/task4_e10_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam_ijcai/task4_e10_reqloss500.0_product_rsched0/train --threshold 0.05 --pred --stats -c 0


# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.25 -rl --rl_mode "pgl_tnorm"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.1_delta0.25_modepgl_tnorm_rsched0/train3 --threshold 0.05 --pred --stats -c 0



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.5 -rl --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.1_delta0.5_modec_violation_rsched0/train2 --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.25 -rl --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_rl_beta0.25_delta0.25_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 0


# β = 0.1, δ = 0.25
# β = 0.1, δ = 0.5 
# β = 0.25, δ = 0.25 
# β = 0.25, δ = 0.5 
# β = 0.5, δ = 0.5


# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type minimum
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_minimum_rsched0/train --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type lukasiewicz
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_lukasiewicz_rsched0/train --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type hamacher_product
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_hamacher_product_rsched0/train --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type nilpotent_minimum
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_nilpotent_minimum_rsched0/train --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type drastic
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_drastic_rsched0/train --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolo11n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type hamacher
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolo11n_reqloss100.0_hamacher_rsched0/train --threshold 0.05 --pred --stats -c 0




# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type aczel_alsina
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_aczel_alsina_rsched0/train --threshold 0.05 --pred --stats -c 0




# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.1 -delta_rl 0.5 -rl --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.1_delta0.5_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.25 -rl --rl_mode "c_violation"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.25_delta0.25_modec_violation_rsched0/train --threshold 0.05 --pred --stats -c 0



# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.5 -rl --rl_mode "all_loss"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.25_delta0.5_modeall_loss_rsched0/train3 --threshold 0.05 --pred --stats -c 0

# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.25 -delta_rl 0.25 -rl --rl_mode "all_loss"
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.25_delta0.25_modeall_loss_rsched0/train2 --threshold 0.05 --pred --stats -c 0
# python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -beta_rl 0.5 -delta_rl 0.5 -rl --rl_mode "all_loss"
# python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_rl_beta0.5_delta0.5_modeall_loss_rsched0/train2 --threshold 0.05 --pred --stats -c 0

python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rs 1.75
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_product_rsched1.75/train --threshold 0.05 --pred --stats -c 0
python main.py -c 0 --task 4 --max_epochs 10 --basemodel yolov8n --req_loss 100 -workers $workers --optimizer Adam --lr 0.01 --dataset road++ --dataset_path ../../ROAD++ --max_det 300 --req-type product -rs 2.0
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_yolov8n_reqloss100.0_product_rsched2.0/train --threshold 0.05 --pred --stats -c 0