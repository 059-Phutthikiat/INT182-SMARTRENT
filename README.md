# SMARTRENT — Bangkok Condominium Monthly Rent Prediction

> **INT182 Data Science and AI - SMARTRENT**  
> โครงงานทำนายค่าเช่าคอนโดมิเนียมที่ลงประกาศต่อเดือน (หน่วยบาท/เดือน) ในกรุงเทพมหานคร จากข้อมูลลักษณะห้องและสิ่งอำนวยความสะดวกภายในห้อง

---

## สมาชิกทีม (Group Member)

| รหัสนักศึกษา  | ชื่อ–นามสกุล         |
| :-----------: | :------------------- |
| `68130500012` | ชยุตม์ ยอดยิ่ง       |
| `68130500036` | นภสร เล็กวิกรม       |
| `68130500038` | นิธิศ ถังทอง         |
| `68130500056` | พีรพัฒน์ เปรมจิตต์   |
| `68130500059` | พุฒิเกียรติ แก้วกล้า |

---

## 1. ภาพรวมโปรเจกต์ (Project Overview)

- **ปัญหา (Pain Point):**
  - **ฝั่งผู้เช่า (First Jobber & Gen Z):** ต้องการหาคอนโดในเมืองที่ตรงงบประมาณ แต่ข้อมูลราคาในตลาดกระจัดกระจาย เปรียบเทียบความคุ้มค่าได้ยาก
  - **ฝั่งเจ้าของห้อง / นักลงทุน:** ขาดราคาอ้างอิงกลาง ทำให้ตั้งราคาปล่อยเช่าไม่สอดคล้องกับตลาดหรือเสียโอกาสปล่อยเช่า
- **สิ่งที่โมเดลทำ (Problem Definition):**
  - งาน **Regression** ทำนาย **ค่าเช่าประกาศต่อเดือน (`rental`, บาท/เดือน)** จากข้อมูลสเปคห้องและสิ่งอำนวยความสะดวกในห้อง
  - **ขอบเขตข้อมูล:** อ้างอิงประกาศเดือนมิถุนายน 2023 จากชุดข้อมูลสาธารณะบน Kaggle: [Condominium Data in Bangkok](https://www.kaggle.com/datasets/krityodp/condominium-data-in-bangkok/data)

---

## 2. โครงสร้างไฟล์ใน Repository และหน้าที่ของแต่ละไฟล์

```text
smartrent-condo-rent-prediction/
├── README.md                              # เอกสารแนะนำโปรเจกต์หน้าแรก
├── requirements.txt                       # รายชื่อไลบรารี Python และเวอร์ชันที่ใช้รัน
├── data/                                  # ข้อมูลต้นฉบับและชุดข้อมูล Train / Val / Test
│   ├── 202306.xlsx                        # ข้อมูลดิบจาก Kaggle (10,312 แถว)
│   ├── train.csv                          # ชุดฝึกโมเดล (4,217 แถว / 138 โครงการ)
│   ├── validation.csv                     # ชุดเลือกโมเดล (1,408 แถว / 46 โครงการ)
│   └── test.csv                           # ชุดทดสอบวัดผลจริง (800 แถว / 46 โครงการ)
├── python_code/                           # โค้ด Python สำหรับเตรียมข้อมูล เทรน และตรวจโมเดล
│   ├── model_pipeline_helpers.py          # สูตรเตรียมข้อมูล สร้างโมเดล และคำนวณคะแนน
│   └── run_verify_or_retrain.py           # สคริปต์สั่งรันตรวจผลหรือเทรนโมเดลใหม่
├── notebooks/                             # ไฟล์ Jupyter Notebook (เปิดดูบน GitHub / Colab ได้)
│   ├── Phase2_Full_Data_Prep_and_Training.ipynb
│   └── Phase2_Verify_and_Retrain.ipynb
├── models/                                # ไฟล์โมเดลที่เทรนเสร็จแล้ว
│   └── phase2_ridge_rent_model.joblib     # โมเดล Ridge Regression (Phase 2)
├── results/                               # ตารางคะแนนและรูปกราฟผลลัพธ์
│   ├── validation_results.csv             # ผลเทียบโมเดลบน Validation set
│   ├── test_results.csv                   # ผลคะแนนบน Test set (800 รายการ)
│   ├── test_predictions.csv               # ค่าเช่าจริง vs ค่าทำนายครบทั้ง 800 รายการ
│   └── figures/
│       ├── correlation_train.png          # กราฟ Correlation Heatmap
│       └── actual_vs_predicted.png        # กราฟราคาจริงเทียบราคาทำนาย
└── slides/                                # สไลด์นำเสนอ (.pdf)
    ├── SMARTRENT_Phase1.pdf
    └── SMARTRENT_Phase2.pdf
```

### คำอธิบายไฟล์โค้ด Python ในโฟลเดอร์ `python_code/`

| ชื่อไฟล์                                                                         | หน้าที่การทำงาน                                                                                                                                                                                                                                                                |
| :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`python_code/model_pipeline_helpers.py`](python_code/model_pipeline_helpers.py) | **ไฟล์รวมสูตรและขั้นตอนการทำงานของโมเดล:** กำหนด 18 Features, ขั้นตอน Data Preprocessing (เติมค่าว่างด้วย Median, One-Hot Encoding, StandardScaler, แปลงค่าเช่าเป็น Log), โหลดไฟล์ `train / validation / test`, สูตรคำนวณคะแนน (`MAE`, `RMSE`, `R²`), และฟังก์ชันฝึก/ตรวจโมเดล |
| [`python_code/run_verify_or_retrain.py`](python_code/run_verify_or_retrain.py)   | **ไฟล์หลักสำหรับกดสั่งรันจากหน้าต่างคำสั่ง:** มี 3 โหมดให้เลือก คือ `--mode verify` (ตรวจโมเดลเดิม), `--mode ridge` (เทรน Ridge ใหม่), และ `--mode all` (รันเทียบทั้ง 4 โมเดล)                                                                                                 |

---

## 3. สรุปโมเดลและผลลัพธ์ปัจจุบัน (Phase 2 Baseline)

- **ตัวแปรที่ใช้ (18 Features ก่อน Encoding):** ขนาดห้องแปลง Log (`log_size`), ชั้น (`floor_no`), จำนวนห้องนอน (`bedrooms`), ประเภทห้อง (`unit_type`), ห้อง 2 ชั้น (`is_duplex`), จำนวนสิ่งอำนวยความสะดวกในห้อง (`n_reported_room_amenities`) และสิ่งอำนวยความสะดวกในห้อง 12 รายการ (`air_conditioner`, `digital_door_lock`, `furnished`, `hot_tub`, `in_room_wifi`, `kitchen_hood`, `kitchen_stove`, `phone`, `refrigerator`, `tv`, `washer`, `water_heater`)
- **การแบ่งข้อมูล:** แบ่งตามรหัสโครงการ (`project_id`) ไม่ให้โครงการเดียวกันซ้ำข้ามชุด เป็น `Train` 4,217 / `Validation` 1,408 / `Test` 800 รายการ (รวม 6,425 รายการ จาก 230 โครงการ)
- **โมเดลที่เลือกใช้:** **Ridge Regression (`alpha=10`)** ฝึกด้วยเป้าหมาย $\ln(1 + \text{rental})$ แล้วแปลงผลกลับเป็นบาท/เดือน หลังเลือกผู้ชนะจาก Validation แล้วจึงนำ `Train + Validation` (5,625 รายการ) มาเทรนซ้ำก่อนทดสอบกับ `Test` (800 รายการ)

### ผลประเมินบน Test Set (800 รายการ จาก 46 โครงการที่ไม่เคยเห็นตอนฝึก)

| โมเดล                             | MAE (บาท/เดือน) | RMSE (บาท/เดือน) |      $R^2$ | หมายเหตุ                                               |
| :-------------------------------- | --------------: | ---------------: | ---------: | :----------------------------------------------------- |
| **Ridge Regression (Phase 2)**    |   **13,866.36** |    **29,879.92** | **0.7089** | ทายคลาดเคลื่อนไม่เกิน ±20% จำนวน 251/800 ห้อง (31.38%) |
| Median Baseline (ทายค่ากลางเดียว) |       27,086.50 |        58,044.82 |    -0.0984 | ใช้เป็นเกณฑ์เปรียบเทียบพื้นฐาน                         |

---

## 4. วิธีรันโค้ดตรวจสอบหรือฝึกโมเดลซ้ำ

ติดตั้งไลบรารีตามไฟล์ `requirements.txt`:

```bash
pip install -r requirements.txt
```

รันคำสั่งจากโฟลเดอร์หลักของโปรเจกต์:

```bash
# 1) ตรวจว่าโมเดลที่บันทึกไว้ใน models/ ทำนาย 800 รายการตรงตาม results/test_predictions.csv
python python_code/run_verify_or_retrain.py --mode verify

# 2) เทรนโมเดล Ridge Regression ใหม่ (ผลใหม่จะถูกเก็บใน my_results/ridge/)
python python_code/run_verify_or_retrain.py --mode ridge

# 3) รันทดลองเปรียบเทียบทั้ง 4 โมเดล (ผลใหม่จะถูกเก็บใน my_results/all_models/)
python python_code/run_verify_or_retrain.py --mode all
```

---
