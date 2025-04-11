# DMD³C: Distilling Monocular Foundation Model for Fine-grained Depth Completion  
**Official Code for the CVPR 2025 Paper**  
**"[CVPR 2025] Distilling Monocular Foundation Model for Fine-grained Depth Completion"**  

[📄 Paper on arXiv](https://arxiv.org/abs/2503.16970)

---

<div align="center">
  <img width="729" alt="DMD3C Results" src="https://github.com/user-attachments/assets/da4a34ea-0390-418c-8111-22b2096110eb" />
</div>

---

## 🔍 Overview

DMD³C introduces a novel framework for **fine-grained depth completion** by distilling knowledge from **monocular foundation models**. This approach significantly enhances depth estimation accuracy in sparse data, especially in regions without ground-truth supervision.

---

<div align="center">
  <img width="860" alt="DMD3C Results" src="https://github.com/user-attachments/assets/362ec3ec-3324-4d36-ab40-3c49d3623cc3" />
</div>

---

## 🆕 Update Log

- **[2025.04.11]** We have released the **inference code**! 🎉  
  Stay tuned for training code and more updates.

---

## 🚀 Getting Started (Inference Only)

### 1. Clone Base Repository

```bash
git clone https://github.com/kakaxi314/BP-Net.git
```

### 2. Copy This Repo into the BP-Net Directory

```bash
cp -r DMD3C/ BP-Net/
cd BP-Net/DMD3C/
```

### 3. Download Pretrained Checkpoints

- 📥 [Google Drive – Checkpoints](#) *(Link coming soon)*

### 4. Prepare KITTI Raw Data

Download any sequence from the **KITTI Raw dataset**, which includes:

- Camera intrinsics  
- Velodyne point cloud  
- Image sequences  

Make sure the structure follows the **standard KITTI format**.

### 5. Modify the Sequence in `demo.py`

Open `demo.py` and go to **line 338**, where you can modify the input sequence path according to your downloaded KITTI data.

```python
# demo.py (Line 338)
sequence = "/path/to/your/kitti/sequence"
```

### 6. Run Inference Demo

```bash
bash demo.sh
```

---

