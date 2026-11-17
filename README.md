# SMARTRENT — Bangkok Condominium Monthly Rent Prediction

> **INT182 Data Science and AI**  
> โครงงานทำนายค่าเช่าคอนโดมิเนียมที่ลงประกาศต่อเดือน (หน่วยบาท/เดือน) ในกรุงเทพมหานคร พร้อม **เว็บสาธิต (Web Demo)** เปรียบเทียบผลทำนายระหว่างโมเดล Phase 2 และ Phase 3

---

## สมาชิกกลุ่ม (Group Member)

| รหัสนักศึกษา  | ชื่อ–นามสกุล         |
| :-----------: | :------------------- |
| `68130500012` | ชยุตม์ ยอดยิ่ง       |
| `68130500036` | นภสร เล็กวิกรม       |
| `68130500038` | นิธิศ ถังทอง         |
| `68130500056` | พีรพัฒน์ เปรมจิตต์   |
| `68130500059` | พุฒิเกียรติ แก้วกล้า |

---

## สรุปผลการพัฒนาจาก Phase 2 -> Phase 3

- **สิ่งที่ปรับปรุงแก้ไขใน Phase 3** เพิ่มข้อมูลระดับโครงการจากไฟล์ `202306.xlsx` (แท็บ `Project`) อีก **6 ตัวแปร** (`log_project_units`, `project_age_2023`, `project_buildings`, `project_max_floors`, `reported_common_amenities`, `project_district`) รวมกับข้อมูลห้องเดิม 18 ตัวแปร เป็น **24 Features (18 ห้อง + 6 โครงการ)** และใช้โมเดล **Ridge Regression**

| โมเดล                                        | จำนวนตัวแปร | MAE (บาท/เดือน) | RMSE (บาท/เดือน) |   $R^2$    | ทายใกล้เคียง (±20%) |
| :------------------------------------------- | :---------: | --------------: | ---------------: | :--------: | :-----------------: |
| **Phase 2 (18 ตัวแปร: เฉพาะข้อมูลห้อง)**     |     18      |       13,866.36 |        29,879.92 |   0.7089   |       31.38%        |
| **Phase 3 (24 ตัวแปร: 18 ห้อง + 6 โครงการ)** |   **24**    |   **10,013.88** |    **24,660.18** | **0.8017** |     **69.50%**      |

- **ผลลัพธ์ดีขึ้นชัดเจน:** `RMSE` ลดลงจาก `29,879.92` เหลือ **`24,660.18` บาท** (`-17.5%`), `MAE` ลดลงจาก `13,866.36` เหลือ **`10,013.88` บาท** (`-27.8%`), และห้องที่ทายคลาดเคลื่อนไม่เกิน 20% เพิ่มขึ้นจาก **31.38% → 69.50%**

---

## โครงสร้างไฟล์ใน Repository

```text
smartrent-condo-rent-prediction/
├── index.html                               # หน้าเว็บสาธิตการทำนายค่าเช่า Phase 2 เทียบ Phase 3
├── demo_data.js                             # พารามิเตอร์โมเดลและชุดข้อมูลทดสอบ 800 ห้องสำหรับหน้าเว็บ
├── README.md                                # เอกสารสรุปภาพรวมโปรเจกต์
├── requirements.txt                         # รายชื่อไลบรารี Python ที่ใช้ในโปรเจกต์
├── data/
│   ├── 202306.xlsx                          # ชุดข้อมูลดิบต้นฉบับจาก Kaggle (Room / Project)
│   ├── train.csv                            # ชุดข้อมูลฝึกสอน (4,217 แถว / 138 โครงการ)
│   ├── validation.csv                       # ชุดข้อมูลสำหรับเลือกโมเดล (1,408 แถว / 46 โครงการ)
│   ├── test.csv                             # ชุดข้อมูลทดสอบวัดผลจริง (800 แถว / 46 โครงการ)
│   └── project_summary.csv                  # ตารางสรุปข้อมูลระดับโครงการ 6 ตัวแปร (241 โครงการ)
├── python_code/
│   ├── pipeline.py                          # ชุดฟังก์ชันเตรียมข้อมูล สร้างโมเดล และวัดผล (Phase 2 & 3)
│   └── run.py                               # สคริปต์สำหรับตรวจสอบผลและฝึกโมเดลซ้ำ
├── notebooks/
│   ├── 1_Data_Prep.ipynb                    # ขั้นที่ 1: ทำความสะอาดข้อมูลจากไฟล์ดิบและเทียบ 4 โมเดล (Phase 2)
│   ├── 2_Phase2_Model.ipynb                 # ขั้นที่ 2: ตรวจสอบและฝึกโมเดลฐาน Phase 2 (18 ตัวแปรห้อง)
│   └── 3_Phase3_Model.ipynb                 # ขั้นที่ 3: อัปเกรดโมเดล Phase 3 (18 ห้อง + 6 โครงการ)
├── models/
│   ├── phase2_ridge_rent_model.joblib       # โมเดล Ridge Phase 2 ที่ฝึกเสร็จแล้ว (18 ตัวแปร)
│   └── phase3_ridge_rent_model.joblib       # โมเดล Ridge Phase 3 ที่ฝึกเสร็จแล้ว (24 ตัวแปร)
├── results/
│   ├── validation_results.csv               # ผลเปรียบเทียบ 4 โมเดลบนชุด Validation (Phase 2)
│   ├── test_results.csv                     # ผลคะแนน Phase 2 บนชุด Test 800 ห้อง
│   ├── test_predictions.csv                 # ค่าทำนายรายห้อง 800 แถวของ Phase 2
│   ├── phase3_test_comparison.csv           # ตารางเทียบผล Phase 2 กับ Phase 3 บน Test 800 ห้อง
│   ├── phase3_test_predictions.csv          # ค่าเช่าจริงเทียบค่าทำนาย Phase 2 และ Phase 3 ครบ 800 แถว
│   └── figures/
│       ├── correlation_train.png            # กราฟสหสัมพันธ์ของตัวแปรหลักในชุด Train
│       ├── actual_vs_predicted.png          # กราฟค่าเช่าจริงเทียบค่าทำนาย (Phase 2)
│       └── phase3_rmse_comparison.png       # กราฟแท่งเปรียบเทียบ RMSE (Phase 2 vs Phase 3)
│
│
└── slides/
    ├── SMARTRENT_Phase1.pdf                 # สไลด์นำเสนอ Phase 1
    ├── SMARTRENT_Phase2.pdf                 # สไลด์นำเสนอ Phase 2
    └── SMARTRENT_Phase3.pdf                 # สไลด์นำเสนอ Phase 3 (Final)
```

---

## แหล่งข้อมูล (Data Source)

- **Kaggle Dataset:** [Condominium Data in Bangkok (`krityodp`)](https://www.kaggle.com/datasets/krityodp/condominium-data-in-bangkok/data) — ข้อมูลประกาศให้เช่าคอนโดมิเนียมในกรุงเทพมหานคร เดือนมิถุนายน 2023
