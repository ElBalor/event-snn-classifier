# 📥 MVSEC DATA DOWNLOAD INSTRUCTIONS

## 🎯 WHAT TO DOWNLOAD

**Sequence:** Indoor Flying 1 (3.8 GB total)

| File | Size | Link |
|------|------|------|
| **Indoor Flying 1 Data** | 1.2 GB | [Google Drive](https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV) |
| **Indoor Flying 1 Ground Truth** | 2.6 GB | [Google Drive](https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV) |
| **Calibration** | ~50 MB | [Google Drive](https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV) |

**Google Drive Folder:** https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV

---

## 📁 WHERE TO PUT IT

```
experiments/02_event_snn_classifier/
└── data/
    └── mvsec/
        └── indoor_flying_1/
            ├── data.hdf5          (or similar name from Google Drive)
            └── ground_truth.hdf5  (or similar name from Google Drive)
```

---

## 🚀 AFTER DOWNLOAD

```powershell
cd experiments/02_event_snn_classifier
python train_drone.py --epochs 50 --batch_size 8
```

---

## 📊 EXPECTED RESULTS

| Metric | Target |
|--------|--------|
| **Training Time** | ~2-3 hours (50 epochs) |
| **Val Loss** | < 0.5 |
| **Depth MAE** | < 0.3 meters |

---

## 🔮 FUTURE UPGRADE

Once the pipeline is proven with Indoor Flying 1, upgrade to:
- **Outdoor Day 1** (19.2 GB) — Real-world drone obstacle detection

---

**Eric Yaka || The Digital Necromancer**
