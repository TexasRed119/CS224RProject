python dpo_better.py --curr_type curriculum --lr 1e-7 --beta 0.1 --num_epochs 4 --static_curr > new_dpo_results/dpo_curriculum_epochs-4_static_lr-1e-7_beta-0.1.txt;
python dpo_better.py --curr_type curriculum --lr 5e-7 --beta 0.25 --num_epochs 4 --static_curr > new_dpo_results/dpo_curriculum_epochs-4_static_lr-5e-7_beta-0.25.txt;
