"""
SMARTRENT — สคริปต์สำหรับสั่งรันตรวจผลหรือเทรนโมเดลซ้ำจาก Terminal

วิธีใช้งาน (รันจากโฟลเดอร์หลักของโปรเจกต์):
  1) ตรวจว่าโมเดลที่บันทึกไว้ให้ผลตรงกับรายงาน:
     python python_code/run_verify_or_retrain.py --mode verify
  2) เทรนโมเดล Ridge Regression ซ้ำตั้งแต่ต้น:
     python python_code/run_verify_or_retrain.py --mode ridge
  3) รันเปรียบเทียบครบทั้ง 4 โมเดล (Median, Ridge, Random Forest, Gradient Boosting):
     python python_code/run_verify_or_retrain.py --mode all
"""

from pathlib import Path
import argparse
from model_pipeline_helpers import verify_saved, reproduce_ridge, reproduce_all


def main():
    parser = argparse.ArgumentParser(
        description='Verify or retrain the SMARTRENT Phase 2 baseline model'
    )
    parser.add_argument(
        '--mode',
        choices=['verify', 'ridge', 'all'],
        default='verify',
        help='verify = ตรวจโมเดลเดิม | ridge = เทรน Ridge ใหม่ | all = รันเทียบทุกโมเดล',
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.mode == 'verify':
        result = verify_saved(root)
    elif args.mode == 'ridge':
        result = reproduce_ridge(root, root / 'my_results' / 'ridge')
    else:
        result = reproduce_all(root, root / 'my_results' / 'all_models')

    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
