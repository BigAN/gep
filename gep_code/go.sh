#!/bin/bash

usage() { echo "Usage: $0 [-s <45|90>] [-p <string>]" 1>&2; exit 1; }
feas=""
paras="num_boost_round,2000==objective,regression_l2==metric,None=boosting_type,gbdt==learning_rate,0.15==num_leaves,100==max_depth,15==feature_fraction,0.7==min_child_samples,100==nthread,12==filter_thred,50000"
label_tran="tran_method,expk==value,1.0"
expk=""
rm="none"
while getopts ":m:t:k:i:v:e:f:p:l:r:n:" opt;  do
    case "${opt}" in
        m)
            model=${OPTARG}
            ;;
        t)
            test_mode=${OPTARG}
            ;;
        k)
            key=${OPTARG}
            ;;
        i)
            intro=${OPTARG}
            ;;
        v)
            valid=${OPTARG}
            ;;
        e)
            expk=${OPTARG}
            ;;
        f)
            feas=${OPTARG}
            ;;
        l)
            label_tran=${OPTARG}
            ;;
        p)
            paras=${OPTARG}
            ;;
        r)
            rm=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

shift $((OPTIND-1))

#echo $feas
new=`echo $feas | sed 's/,/_/g'|cut -c -220`
echo ".....",${new}
time=$(date "+%m%d_%H%M")
mess=log_${model}_${key}_${intro}_${new}_${time}

md5=`echo -n $mess | md5sum | awk '{print substr($1,0,12)}'`
#echo ${a:1:6}
#md5=${a:0:10}
mess2=${mess}_${md5}
echo "????",$md5
echo "????",$mess2
echo "????",$mess




echo $mess > ${mess2}_rs

if [ "$feas" = "" ]
then
    addfeas="deal,pv,dish2,pv_cross,barea_pv_cross,intersect_top50" # 2000 94.48

#    addfeas="deal,pv,dish3,pv_cross,barea_pv_cross,intersect_top50" # 2000 94.20

    echo "use script feas"
else
    addfeas=${feas}
    echo "use shell feas"
fi




#addfeas="deal,pv,dish2,pv_cross,barea_pv_cross,intersect,poi_lda" # 2000 94.48
#addfeas="deal,pv,dish2,pv_cross,barea_pv_cross,intersect" # 2000 94.48
#addfeas="deal,pv,dish2,pv_cross,barea_pv_cross,intersect,deal_lda" # 2000 94.48

#addfeas="deal,pv,dish2_new,pv_cross2,barea_pv_cross_new,intersect,pv3" # 2000 0.2
#addfeas="deal,pv,dish2_new,pv_cross,barea_pv_cross_new,intersect,pv3"
#addfeas="deal,pv,dish2,pv_cross,barea_pv_cross,intersect,pv_decay" # 2000 94.48


echo "paras: ${paras}"

echo "addfeas: ${addfeas}"

echo "label_tran: ${label_tran}"

echo $addfeas >> ${mess2}_rs
echo "python -u ${model}_${key}.py \
--labeltran $label_tran \
--paras $paras \
--intro $intro \
--addfeas $addfeas \
--md5 $md5 \
>> ${mess2}_rs"
#if [ ]
nohup python -u ${model}_${key}.py \
--rm $rm \
--labeltran $label_tran \
--paras $paras \
--intro $intro \
--addfeas $addfeas \
--md5 $md5 \
>> ${mess2}_rs 2>&1 &
echo "finish"
#--expk $expk \

#--rm feature_1,feature_2,feature_3 \
#--rm histtrans2_purchase_data_diff_card_id_diff_mean,hist_installments_var,hist_purchase_amount_max,outliers,hist_purchase_amount_var,new_hist_installments_var \

#--fea market_price,price,deal_max_num,deal_min_num,deal_avg_num,poi_count,beg_weekday,day,month,day2,days_to_side,open_hours,is_mid,is_night,is_midnight,av_5,av_6,av_7,av_days,mt_poi_cate2_name,price_person,has_parking_area,barea_id,has_booth,is_dining,mt_score,dp_score,dp_evn_score,dp_taste_score,dp_service_score,dp_star,poi_zlf,most_rec_cnt_price,dishes_price_bin,most_rec_dish_tag,rec_cnt,dishes_price,menu_name,poi_rank,rec_by_user,price_person \

#bareapvcross_add_cate2
#12
#--rm price_bin_rec_by_user_0,price_bin_rec_by_user_1,price_bin_rec_by_user_2,price_bin_rec_by_user_3,price_bin_rec_by_user_4,price_bin_rec_cnt_0 \
#nohup python -u lgbm_${key}.py \
#--fea poi_id,poi_zlf,market_price,price_person,price,mt_score,deal_max_num,deal_avg_num,dp_evn_score,beg_weekday,day,month,day2,days_to_side,open_hours,is_mid,is_night,is_midnight,av_5,av_6,av_7,av_days \
#--cat beg_weekday \
#--rm beg_weekday,day,month,day2,days_to_side,open_hours,is_mid,is_night,is_midnight,av_5,av_6,av_7,av_days \
#--addfeas deal \
#--detailfeas discount#price_avg_person \
#>> lgbm_${key}_rs 2>&1 &

#poi_pv_sum_1#poi_pv_mean_1#poi_uv_sum_1#poi_uv_mean_1#poi_pv_sum_3#poi_pv_mean_3#poi_uv_sum_3#poi_uv_mean_3#poi_pv_sum_7#poi_pv_mean_7#poi_uv_sum_7#poi_uv_mean_7#poi_pv_sum_14#poi_pv_mean_14#poi_uv_sum_14#poi_uv_mean_14#poi_pv_sum_30#poi_pv_mean_30#poi_uv_sum_30#poi_uv_mean_30 \



#--rm discount#price_avg_person \
#--cat poi_rank,mt_poi_cate2_name,dp_star \

#poi_pv_min_14#poi_pv_std_30#poi_pv_mean14#poi_pv_std_14#poi_pv_mean3#poi_pv_min_7#poi_pv_max_14#poi_pv_min_1#poi_pv_min_3#poi_pv_max_7#poi_pv_std_7#poi_pv_mean7#poi_pv_mean30#poi_pv_max_3#poi_pv_std_3#poi_pv_max_1#poi_pv_std_1#poi_pv_mean1#poi_pv_min_30#poi_pv_max_30,\



#--fea market_price,mt_poi_cate2_name,price,deal_max_num,deal_min_num,deal_avg_num,poi_count,price_person,has_parking_area,has_booth,is_dining,barea_id,mt_score,dp_score,dp_evn_score,dp_taste_score,dp_service_score,poi_zlf \

