# Shell script to download Div2k dataset - https://data.vision.ee.ethz.ch/cvl/DIV2K/
# Refer to https://github.com/DAI-Lab/SteganoGAN/blob/master/research/data/download.sh

mkdir div2k
cd div2k

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
mkdir train
unzip -j DIV2K_train_HR.zip -d train/_
rm DIV2K_train_HR.zip

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
mkdir val
unzip -j DIV2K_valid_HR.zip -d val/_
rm DIV2K_valid_HR.zip

cd ..