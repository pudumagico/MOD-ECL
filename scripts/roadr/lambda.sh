cd ../../src

model=yolov8n
epochs=10
workers=6

export OMP_NUM_THREADS=16

python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1,2,3 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 1 --req_num_detect -1 --req-type product &&
python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1,2,3 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 50 --req_num_detect -1 --req-type product &&
python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1,2,3 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 1000 --req_num_detect -1 --req-type product &&
python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1,2,3 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 5000 --req_num_detect -1 --req-type product &&
python -m torch.distributed.run --nproc_per_node 4  main.py -c 0,1,2,3 --task 2 --basemodel $model --max_epochs $epochs  --workers $workers  --req_loss 10000 --req_num_detect -1 --req-type product
