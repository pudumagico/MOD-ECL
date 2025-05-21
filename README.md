# T-norm Selection for Object Detection in Autonomous Driving with Logical Constraints
NeurIPS 2025 Paper ID: 23576

## Installation of Data



### Downloading ROAD
Run the following command in the root folder.
```bash
cd scripts
bash road_installation.sh
```


### Downloading ROAD-Waymo
Run the following commands in the root folder.
```bash
mkdir ../ROAD++
mkdir ../ROAD++/train
mkdir ../ROAD++/train/videos
```

Visit https://waymo.com/open/download/, sign in with your Google account, and download the data using the Google Cloud Console.
Place all video files into the ../ROAD++/train/videos directory, and place the file road_waymo_trainval_v1.0.json in the ../ROAD++ directory. No other files/folders should be required.
Once you have downloaded the dataset to the correct locations, run the following commands.

```bash
cd scripts
bash road++_installation.sh
```




# Environment Setup

You can run the experiments using either a Conda environment or Docker. Make sure to run the commands in the root folder.

### Conda (Recommended for local development)
To create and activate the Conda environment:
```bash
conda create --name modecl --file environment.yml
conda activate modecl
```

### Docker (Alternative for isolated environments)
If you prefer using Docker, make sure you have Docker and Docker Compose installed. Then, run the following command to start the environment:
```bash
cd docker
docker compose up --build
docker exec -it modecl /bin/bash
```

This will launch the container as defined in `docker-compose.yml`, which includes all required dependencies and runtime settings.

## Downloading Experimental Results
Run the following command in the root folder.
```bash
cd scripts
bash dl_experiments.sh
```


## Training Models

You can produce baselines, use individual t-norms `T`, change the λ weight for the constrained loss, enable the λ scheduler, use the adaptive algorithm, or combine these.  

   Consider:
   - `e`: number of epochs  
   - `w`: number of workers  
   - `m`: a base model from the YOLOv8 or YOLO11 family  
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

### Testing Models

Use the following commands to evaluate models on ROAD-R or ROAD-Waymo-R:
```bash
# For ROAD-R
python tester.py --model run_folder --stats --pred --task 2

# For ROAD-Waymo-R
python tester.py -dataset road++r --dataset_path ../../ROAD++ --model run_folder --stats --pred --task 4
```

