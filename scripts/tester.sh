cd ../src

conf=0.1

# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.0_vanilla_rsched0/train --conf $conf -pred -stats
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss0.1_minimum_rsched0/train --conf $conf -pred -stats


# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_lukasiewicz_rsched0/train --conf $conf -pred -stats --maxsat -c 1
# #minimum
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_minimum_rsched0/train --conf $conf -pred -stats --maxsat -c 1
# #sugeno_weber
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_sugeno_weber_rsched0/train2 --conf $conf -pred -stats --maxsat -c 1
# #aczel_alsina
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_aczel_alsina_rsched0/train --conf $conf -pred -stats --maxsat -c 1

# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_rl_rsched0/train --conf 0.05 -pred -stats --maxsat
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss250.0_product_rsched0/train2 --conf 0.05 -pred -stats --maxsat
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1000.0_product_rsched0/train3 --conf 0.05 -pred -stats --maxsat
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss5000.0_product_rsched0/train2 --conf 0.05 -pred -stats --maxsat


# conf=0.05
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_lukasiewicz_rsched0/train --conf $conf -pred -stats --maxsat -c 1
# #minimum
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_minimum_rsched0/train --conf $conf -pred -stats --maxsat -c 1
# #sugeno_weber
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_sugeno_weber_rsched0/train2 --conf $conf -pred -stats --maxsat -c 1
# #aczel_alsina
# python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss100.0_aczel_alsina_rsched0/train --conf $conf -pred -stats --maxsat -c 1


python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_minimum_rlr0.1_rsched0/train --conf 0.05 -pred -stats
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_minimum_rlr0.2_rsched0/train --conf 0.05 -pred -stats
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_minimum_rlr0.3_rsched0/train --conf 0.05 -pred -stats

python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_product_rlr0.1_rsched0/train --conf 0.05 -pred -stats
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_product_rlr0.2_rsched0/train2 --conf 0.05 -pred -stats
python tester.py --dataset road++r --dataset_path ../../ROAD++ --model nparam/task4_e10_reqloss1.0_product_rlr0.3_rsched0/train --conf 0.05 -pred -stats