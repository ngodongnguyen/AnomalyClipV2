
device=0


LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/AnomalyClipV2/data/CVC/CVC-ClinicDB --save_path ./results/${base_dir}/zero_shot/ClinicDB \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level
    wait
    done
done

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/AnomalyClipV2/data/CVC/CVC-ColonDB --save_path ./results/${base_dir}/zero_shot/ColonDB \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level
    wait
    done
done

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset colon \
        --data_path /home/ai3/NguyenND/AnomalyClipV2/data/CVC/Kvasir --save_path ./results/${base_dir}/zero_shot/Kvasir \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} \
        --metrics pixel-level
    wait
    done
done
