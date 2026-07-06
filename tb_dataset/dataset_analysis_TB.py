"""
Ce module permet d'avoir un vu global sur le dataset. En sortie on aura deux fichiers 
 - Un pdf contenant toute les informations
  - Un fichier excel contenant l'image, la classe et le chemin de l'image
"""
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
from collections import defaultdict

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# =========================================================
# 🔥 METADATA LOADER
# =========================================================
def load_metadata_file(path):
    if path is None:
        return None

    if path.endswith(".csv"):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")

    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl")

    else:
        raise ValueError(f"Unsupported format: {path}")


def load_multiple_metadata(metadata_list):
    dfs = []

    if metadata_list is None:
        return None

    for path in metadata_list:
        try:
            df = load_metadata_file(path)
            df["__source__"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] {e}")

    if len(dfs) == 0:
        return None

    return pd.concat(dfs, ignore_index=True, sort=False)


# =========================================================
# 🧠 MAIN CLASS
# =========================================================
class KaggIeTBAnalyzer:

    def __init__(self, root_dir, metadata_files=None):
        self.root_dir = root_dir
        self.metadata = load_multiple_metadata(metadata_files)

        self.df = None
        self.corrupted = []
        self.duplicates_md5 = []
        self.size_stats = None

    # =====================================================
    # 1. LOAD DATASET
    # =====================================================
    def load_dataset(self):
        data = []

        for cls in os.listdir(self.root_dir):
            cls_path = os.path.join(self.root_dir, cls)

            if not os.path.isdir(cls_path):
                continue

            for img in os.listdir(cls_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    data.append({
                        "path": os.path.join(cls_path, img),
                        "label": cls,
                        "filename": img
                    })

        self.df = pd.DataFrame(data)
        print("[INFO] Images:", len(self.df))

    # =====================================================
    # 2. CORRUPTED IMAGES
    # =====================================================
    def check_corrupted(self):
        corrupted = []

        for p in tqdm(self.df["path"]):
            try:
                if cv2.imread(p) is None:
                    corrupted.append(p)
            except:
                corrupted.append(p)

        self.corrupted = corrupted

    # =====================================================
    # 3. IMAGE STATS
    # =====================================================
    def image_stats(self):
        w, h = [], []

        for p in tqdm(self.df["path"]):
            try:
                img = Image.open(p)
                ww, hh = img.size
                w.append(ww)
                h.append(hh)
            except:
                pass

        self.size_stats = {
            "mean_w": np.mean(w),
            "mean_h": np.mean(h),
            "std_w": np.std(w),
            "std_h": np.std(h)
        }

    # =====================================================
    # 4. DUPLICATES
    # =====================================================
    def md5(self, path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                h.update(b)
        return h.hexdigest()

    def detect_duplicates(self):
        hashes = defaultdict(list)

        for p in self.df["path"]:
            try:
                hashes[self.md5(p)].append(p)
            except:
                pass

        self.duplicates_md5 = [v for v in hashes.values() if len(v) > 1]

    # =====================================================
    # 5. SUMMARY
    # =====================================================
    def dataset_summary(self, patient_col="patient_id"):
        dist = self.df["label"].value_counts()

        summary = {
            "total_images": len(self.df),
            "num_classes": len(dist),
            "tb_ratio": dist.get("Tuberculosis", 0) / len(self.df),
            "imbalance_ratio": dist.max() / dist.min(),
            "is_imbalanced": dist.max() / dist.min() > 1.5,
            "patients": self.df[patient_col].nunique() if patient_col in self.df.columns else None,
            "distribution": dist
        }

        return summary

    # =====================================================
    # 6. CLASS PLOT
    # =====================================================
    def plot_distribution(self, path="class_dist.png"):
        dist = self.df["label"].value_counts()

        plt.figure()
        dist.plot(kind="bar")
        plt.title("Class Distribution")
        plt.ylabel("Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return path

    # =====================================================
    # 7. TEXT REPORT
    # =====================================================
    def imbalance_text(self, summary):
        if summary["is_imbalanced"]:
            return (
                 "🚨 WARNING: The dataset is imbalanced.\n"
                    "⚠️ Use class weighting, oversampling, or focal loss before training."
            )
        return "✔ Dataset is balanced."

    # =====================================================
    # 8. PDF REPORT (FULL)
    # =====================================================
    def generate_pdf_report(self, out_pdf="tb_report.pdf"):

        doc = SimpleDocTemplate(out_pdf)
        styles = getSampleStyleSheet()
        elements = []

        # ================= TITLE =================
        elements.append(Paragraph("TB Chest X-ray Dataset Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        # ================= SUMMARY =================
        summary = self.dataset_summary()
        plot_path = self.plot_distribution()

        elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
        elements.append(Paragraph(f"Total images: {summary['total_images']}", styles["Normal"]))
        elements.append(Paragraph(f"Number of classes: {summary['num_classes']}", styles["Normal"]))
        elements.append(Paragraph(f"TB ratio: {summary['tb_ratio']:.4f}", styles["Normal"]))
        #elements.append(Paragraph(f"Patients: {summary['patients']}", styles["Normal"]))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(self.imbalance_text(summary), styles["Normal"]))
        elements.append(Spacer(1, 12))

        # ================= CLASS TABLE =================
        table_data = [["Class", "Count"]]
        for k, v in summary["distribution"].items():
            table_data.append([k, str(v)])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

        # ================= IMAGE PLOT =================
        elements.append(Paragraph("Class Distribution", styles["Heading2"]))
        elements.append(RLImage(plot_path, width=400, height=250))
        elements.append(Spacer(1, 12))

        # ================= IMAGE STATS =================
        if self.size_stats:
            s = self.size_stats
            elements.append(Paragraph("Image Statistics", styles["Heading2"]))
            elements.append(Paragraph(f"Mean width: {s['mean_w']:.2f}", styles["Normal"]))
            elements.append(Paragraph(f"Mean height: {s['mean_h']:.2f}", styles["Normal"]))
            elements.append(Spacer(1, 12))

      
        # ================= BUILD =================
        doc.build(elements)
        print(f"[INFO] PDF saved: {out_pdf}")

    # =====================================================
    # FULL PIPELINE
    # =====================================================
    def run(self):
        self.load_dataset()
        self.check_corrupted()
        self.image_stats()
        self.detect_duplicates()
        self.generate_pdf_report()