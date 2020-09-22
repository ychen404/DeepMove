#!/bin/bash
# baseFolder="/a/b/c/x/y/"
# completeFilePath="a/b/c/x/y/z.txt"
# filename=`echo $completeFilePath | awk -v a=$baseFolder '{len=length(a)}{print substr($0,len)}'`
# echo $filename
# echo $filename | cut -d'.' -f1

baseFolder="/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/user_data//"
completeFilePath="/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/user_data/tweets-cikm-uid-*.txt"
i=0
for f in $completeFilePath
do
    echo "Processing $f"
    full_filename=`echo $f | awk -v a=$baseFolder '{len=length(a)}{print substr($0,len)}'`
    # i=$((i+1))
    # echo $i
    # echo $full_filename
    filename=`echo $full_filename | cut -d'.' -f1`
    # echo $filename
    # cat $f
    python3 parse_cikm.py --save_name=$filename --twitter_path=$f
done
# python3 parse_cikm.py --save_name='tweets-cikm-uid-2' \
#                     --twitter_path='/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/user_data/tweets-cikm-uid-2.txt' \



# python3 parse_cikm.py --save_name='tweets-cikm-uid-test' \
#                     --twitter_path='/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/user_data/tweets-cikm-uid-2-notworking-92983169.txt' \