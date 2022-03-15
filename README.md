DDH-LDL

Code for Deep Discrete Hashing for Label Distribution Learning
You can download dataset in https://pan.baidu.com/s/1Dw8jrxA5zSexpZ-3jkTgoA password：7fc6

Please run python main.py 

python main.py --epoch_code 20 --epoch_func 51 --dom_iter 1 --lr_code_net 0.0002 --lr_func_net 0.0002 --alpha 10 --beta 0.5 --dataset Movie.mat

python main.py --epoch_code 30 --epoch_func 26 --dom_iter 10 --lr_code_net 0.0002 --lr_func_net 0.0002 --alpha 10 --beta 0.9 --dataset RAF_ML.mat

python main.py --epoch_code 20 --epoch_func 71 --dom_iter 1 --lr_code_net 0.0002 --lr_func_net 0.0002 --alpha 0.1 --beta 0.5 --dataset fbp5500.mat 

python main.py --epoch_code 2 --epoch_func 36 --dom_iter 1 --lr_code_net 0.0002 --lr_func_net 0.0002 --alpha 0.1 --beta 0.9 --dataset Flickr_ldl.mat 

python main.py --epoch_code 5 --epoch_func 15 --dom_iter 1 --lr_code_net 0.0002 --lr_func_net 0.0002 --alpha 10 --beta 0.9 --dataset SCUT_FBP.mat



