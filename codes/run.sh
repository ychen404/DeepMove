rm -rf ../results/checkpoint/
python2.7 main_simple.py --pretrain=0 --model_mode=simple --data_name='foursquare_nyc_public' --accuracy_mode='top1'
