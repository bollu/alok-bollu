# Grassmanian

#### Setting up armadillo

```
wget http://sourceforge.net/projects/arma/files/armadillo-9.900.3.tar.xz
tar -xf armadillo-9.900.3.tar.xz 
cd armadillo-9.900.3/
./configure --prefix=/home/bollu/.local -DCMAKE_INSTALL_PREFIX=/$HOME/.local
make install 
echo >> ~/.bashrc
echo '$LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
