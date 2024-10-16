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

mkdir road_test
cd road_test
mkdir videos
cd videos



wget "https://drive.usercontent.google.com/download?id=1dTdvipm3Y9xEISvlqkzWfQisUzMGvC-V&confirm=xxx" -O 2014-06-26-09-31-18_stereo_centre_02.mp4
wget "https://drive.usercontent.google.com/download?id=10eq0zDHInLCJS_sFfT2FApEeC86kEZ3K&confirm=xxx" -O 2014-12-10-18-10-50_stereo_centre_02.mp4
wget "https://drive.usercontent.google.com/download?id=1D7a_T0K5Xko-eZOVRJvIAxi2FpENz7_C&confirm=xxx" -O 2015-02-03-08-45-10_stereo_centre_04.mp4
wget "https://drive.usercontent.google.com/download?id=1fYiOdAND2xyML9fEgMTdWnO1PQf8a8GN&confirm=xxx" -O 2015-02-06-13-57-16_stereo_centre_01.mp4


cd ../../
python extract_videos2jpgs.py ./road_test/
