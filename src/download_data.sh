
wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
mkdir -p data/FDDB
tar xvzf originalPics.tar.gz -C data/FDDB
tar xvzf FDDB-folds.tgz -C data/FDDB
rm *gz
