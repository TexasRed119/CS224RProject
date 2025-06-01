python sft.py --curr_type anti --num_epochs 6 --lr 1e-5 --scheduler > anti_1e-5_scheduler.txt
python sft.py --curr_type curriculum --num_epochs 6 --lr 1e-5 --scheduler > curriculum_1e-5_scheduler.txt
python sft.py --curr_type none --num_epochs 6 --lr 1e-6 --scheduler > none_1e-6_scheduler.txt