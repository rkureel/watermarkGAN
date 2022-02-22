# Shell script to download COCO dataset - https://cocodataset.org/#home
# Refer to https://github.com/DAI-Lab/SteganoGAN/blob/master/research/data/download.sh

mkdir mscoco
cd mscoco

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mkdir train
mv train2017 train/_
rm train2017.zip

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mkdir val
mv test2017 val/_
rm test2017.zip

cd ..