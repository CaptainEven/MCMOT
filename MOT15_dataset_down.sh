cd ..
wget https://motchallenge.net/data/2DMOT2015.zip
mkdir dataset
cd dataset
mkdir MOT15
cd MOT15
mkdir labels_with_ids
mkdir labels_with_ids/train
unzip ../../2DMOT2015.zip 
mv 2DMOT2015 images