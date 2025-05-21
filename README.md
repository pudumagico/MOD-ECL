# T-norm Selection for Object Detection in Autonomous Driving with Logical Constraints
NeurIPS 2025 Paper ID: 23576

## Installation of Data

### Downloading ROAD++
Run the following commands in the root folder.
```
mkdir ../ROAD++
mkdir ../ROAD++/train
mkdir ../ROAD++/train/videos
```

Go to https://waymo.com/open/download/, connect your google account, then download the data from the google cloud console.
The videos should be downloaded into ../ROAD++/train/videos folder, and road_waymo_trainval_v1.0.json should be downloaded into the ROAD++ folder. No other files/folders should be required.
Once you have downloaded the dataset to correct positions, run the following commands.

```
cd scripts
bash road++_installation.sh
```

### Downloading ROAD
Run the following command in the root folder.
```
cd scripts
bash road_installation.sh
```



## Downloading experiment logs
Run the following command in the root folder.
```
cd scripts
bash dl_experiments.sh
```

## Running experiments

1. Create a conda environment:
   ```bash
   conda create --name modecl --file requirements.txt
   ```
2. Download the logs using the link provided in our repository.
3. Reproduce the main results with:
   ```bash
   python tester.py --model run_folder --stats --pred --task i
   ```
<!--
To produce constrained output results using MaxSAT:
```bash
python tester.py --model run_folder --stats --pred --task i --maxsat
``` 
-->

---

## Training Models from Scratch

You can produce baselines, use individual t-norms `T`, change the λ weight for the constrained loss, enable the λ scheduler, use the adaptive algorithm, or combine these.  

   Consider:
   - `e`: number of epochs  
   - `w`: number of workers  
   - `m`: a base model from the YOLOv8 family  
   - `--req_loss=l`: λ value for the constrained loss  

   Examples:
   - **Baseline**:
     ```bash
     python main.py --task i --basemodel m --max_epochs e --workers w --req_loss 0
     ```

   - **Individual t-norm**:
     ```bash
     python main.py --task i --basemodel m --max_epochs e --workers w --req_loss l --req-type T
     ```

   - **Adaptive algorithm** (uses all t-norms listed in the main paper):
     ```bash
     python main.py --task i --basemodel m --max_epochs e --workers w -rl
     ```

   - **λ scheduler** (with scheduler constant `s ≥ 0`):
     ```bash
     python main.py --task i --basemodel m --max_epochs e --workers w --req_loss l --req-type T --req_scheduler s
     ```

You may combine these arguments as needed.
