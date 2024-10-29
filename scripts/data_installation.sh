cd ../../
git clone https://github.com/gurkirt/road-dataset.git

cd ./road-dataset/road/

# wget "https://drive.usercontent.google.com/download?id=1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V&confirm=xxx" -O instance_counts.json 
wget "https://drive.usercontent.google.com/download?id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz&confirm=xxx" -O videos.zip
wget "https://drive.usercontent.google.com/download?id=1YIbfeU9kCw9qCDQyjsJxLZjGDcPYAf7N&confirm=xxx" -O road_trainval_v1.0.json 

unzip videos.zip
rm videos.zip

cd ../
python extract_videos2jpgs.py ./road/

rm -rf .git*
