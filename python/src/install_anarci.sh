curl http://eddylab.org/software/hmmer/hmmer.tar.gz > hmmer.tar.gz
tar xf hmmer.tar.gz
cd hmmer-3.3.2
./configure
make
make install
cd ..

git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python3 -m pip install -r requirements.txt
sudo python3 setup.py install
cd ..
