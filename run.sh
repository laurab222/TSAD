#!/bin/bash
# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh


# echo "data set IEEECIS_new2.2"
# for i in {1..5}
# do
#     echo "rep $i"
#     # echo "data set $d"
#     python main.py --model TranAD --window_size 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name $i
#     # python main.py --model iTransformer --window_size 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
#     # python main.py --model iTransformer --window_size 100 --dataset IEEECIS_new2.2 --step_size 50 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
# done

# for d in SMAP_new MSL_new creditcard creditcard_normal # GECCO GECCO_normal # UCR SWaT_1D creditcard_normal SMAP_new creditcard
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name $i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 5 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 10 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#     done
# done

# for d in GECCO GECCO_normal
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name newloss3_$i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 5 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 10 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#     done
# done

# for d in SMD # SWaT UCR GECCO ATLAS_TS SMD MSL_new
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name $i --less
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#     done
# done

# # for d in  # SMD SMAP_new UCR ATLAS_TS SWaT
# do
#     echo "data set $d"
#     python main.py --model USAD --window_size 10 --dataset $d --step_size 1 --retrain --feats -1 
# done

# for d in SMD SMAP_new UCR ATLAS_TS
# do
#     echo "data set $d"
#     python main.py --model MERLIN --window_size 10 --dataset $d --step_size 1 --retrain --feats -1 
# done


# for d in lorenzetti_wa20_new # lorenzetti_wp40_wa5 lorenzetti_wa5  # lorenzetti_wp40_wa10 lorenzetti_wp40_wa20 lorenzetti_wa10 lorenzetti_wa20
# do
#     echo "data set $d"
#     for i in 1
#     do
#         python main.py --model iTransformer --dataset $d -w 2000 -s 1000 -e 10 -d 248 --retrain --feats -1 --k $i --name rep_$i #> iTransformer/iTransformer_$d/out_rep_$i.log 
#         python main.py --model iTransformer --dataset $d -w 200 -s 100 -e 10 -d 124 --retrain --feats -1 --k $i --name rep_$i #> iTransformer/iTransformer_$d/out_rep_$i.log 
#         python main.py --model iTransformer --dataset $d -w 20 -s 10 -e 10 -d 16 --retrain --feats -1 --k $i --name rep_$i #> iTransformer/iTransformer_$d/out_rep_$i.log 
#         python main.py --model iTransformer --dataset $d -w 10 -s 1 -e 10 -d 10 --retrain --feats -1 --k $i --name rep_$i #> iTransformer/iTransformer_$d/out_rep_$i.log     
#     done
# done

# for d in lorenzetti_wa20_new lorenzetti_wp40_wa5 lorenzetti_wa5 lorenzetti_wp40_wa10 lorenzetti_wp40_wa20 lorenzetti_wa10 lorenzetti_wa20
# do
#     echo "data set $d"
#     # Check if the folder exists
#     if [ ! -d "TranAD/TranAD_${d}" ]; then
#         echo "Creating folder TranAD/TranAD_${d}"
#         mkdir -p "TranAD/TranAD_${d}"
#     fi    
#     if [ ! -d "USAD/USAD_${d}" ]; then
#         echo "Creating folder USAD/USAD_${d}"
#         mkdir -p "USAD/USAD_${d}"
#     fi
#     for i in 1
#     do
#         python main.py --model TranAD --dataset $d -w 10 -s 1 -e 10 --retrain --feats -1 --k $i --name rep_$i > TranAD/TranAD_$d/out_rep_$i.log
#         python main.py --model USAD --dataset $d -w 10 -s 5 -e 10 --retrain --feats -1 --k $i --name rep_$i > USAD/USAD_$d/out_rep_$i.log
#     done
# done

# for d in lorenzetti_wa5_minmax lorenzetti_wp40_wa5_minmax lorenzetti_wa10_minmax lorenzetti_wp40_wa10_minmax lorenzetti_wa20_minmax lorenzetti_wp40_wa20_minmax 
# do 
#     for i in {1..5}
#     do
#         # python main.py --model iTransformer --dataset $d -w 2000 -s 1000 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         # python main.py --model iTransformer --dataset $d -w 200 -s 100 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         # python main.py --model iTransformer --dataset $d -w 20 -s 10 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         python main.py --model iTransformer --dataset $d -w 10 -s 1 -e 10 --test --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log    
#         python main.py --model TranAD --dataset $d -w 10 -s 1 -e 10 --test --feats -1 --k $i --name rep_$i > TranAD/TranAD_$d/out_minmax_rep_$i.log
#         python main.py --model USAD --dataset $d -w 10 -s 5 -e 10 --test --feats -1 --k $i --name rep_$i > USAD/USAD_$d/out_minmax_rep_$i.log
#     done
# done

# for d in lorenzetti_wa20_new_minmax lorenzetti_wp40_wa20_new_minmax lorenzetti_wa20_new2_minmax lorenzetti_wp40_wa20_new2_minmax lorenzetti_wa50_new2_minmax lorenzetti_wp40_wa50_new2_minmax  # lorenzetti_deadcells
# do
#     echo "data set $d"
#     # Check if the folder exists
#     # if [ ! -d "TranAD/TranAD_${d}" ]; then
#     #     echo "Creating folder TranAD/TranAD_${d}"
#     #     mkdir -p "TranAD/TranAD_${d}"
#     # fi    
#     # if [ ! -d "USAD/USAD_${d}" ]; then
#     #     echo "Creating folder USAD/USAD_${d}"
#     #     mkdir -p "USAD/USAD_${d}"
#     # fi
#     # if [ ! -d "iTransformer/iTransformer_${d}" ]; then
#     #     echo "Creating folder iTransformer/iTransformer_${d}"
#     #     mkdir -p "iTransformer/iTransformer_${d}"
#     # fi

#     for i in {1..5}
#     do
#         # python main.py --model iTransformer --dataset $d -w 2000 -s 1000 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         # python main.py --model iTransformer --dataset $d -w 200 -s 100 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         # python main.py --model iTransformer --dataset $d -w 20 -s 10 -e 10 --retrain --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log 
#         python main.py --model iTransformer --dataset $d -w 10 -s 1 -e 10 --test --feats -1 --k $i --name rep_$i > iTransformer/iTransformer_$d/out_minmax_rep_$i.log    
#         python main.py --model TranAD --dataset $d -w 10 -s 1 -e 10 --test --feats -1 --k $i --name rep_$i > TranAD/TranAD_$d/out_minmax_rep_$i.log
#         python main.py --model USAD --dataset $d -w 10 -s 5 -e 10 --test --feats -1 --k $i --name rep_$i > USAD/USAD_$d/out_minmax_rep_$i.log
#     done
# done

for d in SMAP_new MSL_new creditcard creditcard_normal GECCO GECCO_normal UCR SWaT_1D SMD
do 
    for i in {1..5}
    do 
        python main.py --model AnomalyTransformer --dataset $d -w 10 -s 1 -d 8 -e 10 --feats -1 --k $i --retrain

    done
done