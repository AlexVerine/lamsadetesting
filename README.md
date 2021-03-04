# lamsadetesting
Codes to test lamsade different servers



## Steps to do before launching the test:
* mkdir data
* virtualenv venv --python=python3.8
* source venv/bin/activate
* pip install -r requirements.txt





## To run the test:
> python main.py 

or 

> python main.py --n_memory 1000 --n_batchs 100 --n_test_gpu 20 --n_test_cpus 20

with:
*  --n_memory N     number of iterations for memory access
*  --n_batchs N     Number of batchs for the only epoch of testing for cpu/gpu
*  --n_test_gpus N  number of iterations of generating and training on one epoch on GPU
*  --n_test_cpus N  number of iterations of generating and training on one epoch on CPU
