conda create --name py35 python=3.5
conda install numpy scipy mkl-service libpython m2w64-toolchain
python -m pip install --upgrade pip
pip install theano
pip install lasagne
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
conda install pandas
conda install -c anaconda scikit-learn 
pip install pypiwin32