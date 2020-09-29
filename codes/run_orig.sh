rm -rf ../results/checkpoint/
# python3 main.py --pretrain=0 --model_mode=attn_local_long --L2=1e-6 --clip=2 --dropout=0.6 --hidden_size=300 \
      		# --learning_rate=0.0001 --loc_emb_size=300 --tim_emb_size=20 --data_name='tweets-cikm'

python3 main.py --pretrain=0 --model_mode=simple --L2=1e-6 --clip=5 --dropout=0.3 --hidden_size=500 \
     		--learning_rate=0.0001 --loc_emb_size=500 --tim_emb_size=10 --data_name='tweets-cikm-50'

#python3 main.py --pretrain=0 --model_mode=simple_long --L2=1e-5 --clip=5 --dropout=0.5 --hidden_size=200 \
#      		--learning_rate=0.0007 --loc_emb_size=500 --tim_emb_size=10 --data_name='tweets-cikm'

#python3 main.py --pretrain=0 --model_mode=attn_avg_long_user --L2=1e-5 --clip=5 --dropout=0.2 --hidden_size=300 \
#      		--learning_rate=0.0007 --loc_emb_size=100 --tim_emb_size=10 --data_name='tweets-cikm'
