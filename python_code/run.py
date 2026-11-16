"""
SMARTRENT — สคริปต์สำหรับสั่งรันตรวจผลหรือเทรนโมเดลซ้ำจาก Terminal

วิธีใช้งาน (รันจากโฟลเดอร์หลักของโปรเจกต์):
  1) ตรวจว่าโมเดล Phase 2 ให้ผลตรงกับรายงาน:
     python python_code/run.py --mode verify
  2) ตรวจผลโมเดล Phase 3 เทียบกับ Phase 2 บนชุด Test 800 รายการ:
     python python_code/run.py --mode phase3
  3) เทรนโมเดล Ridge Regression (Phase 2) ซ้ำตั้งแต่ต้น:
     python python_code/run.py --mode ridge
  4) รันเปรียบเทียบ Phase 2 ครบทั้ง 4 โมเดล:
     python python_code/run.py --mode all
"""

from pathlib import Path
import argparse
from pipeline import (
    verify_saved,
    verify_phase3_saved,
    reproduce_ridge,
    reproduce_all,
)


def main():
    parser = argparse.ArgumentParser(
        description='Verify or retrain the SMARTRENT Phase 2 & Phase 3 models'
    )
    parser.add_argument(
        '--mode',
        choices=['verify', 'phase3', 'ridge', 'all'],
        default='phase3',
        help='verify = ตรวจ Phase 2 | phase3 = ตรวจ Phase 3 | ridge = เทรน Ridge ใหม่ | all = รันเทียบทุกโมเดล',
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.mode == 'verify':
        result = verify_saved(root)
    elif args.mode == 'phase3':
        result = verify_phase3_saved(root)
    elif args.mode == 'ridge':
        result = reproduce_ridge(root, root / 'my_results' / 'ridge')
    else:
        result = reproduce_all(root, root / 'my_results' / 'all_models')

    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
