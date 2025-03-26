import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Pastikan ultralytics sudah terinstal

# Pastikan file model tersedia di lokasi yang benar
MODEL_PATH = "best.pt"

# Memuat model dengan YOLO jika berasal dari Ultralytics
try:
    model = YOLO(MODEL_PATH)  # Gunakan YOLO jika model berasal dari Ultralytics
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Fungsi utama
def detect_canny_edges_with_model(low_threshold=50, high_threshold=150):
    cap = cv2.VideoCapture(0)  # Gunakan kamera default
    
    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame dari kamera!")
            break
        
        # Konversi ke grayscale untuk Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Konversi frame ke RGB untuk YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prediksi menggunakan model YOLO
        results = model(frame_rgb)  # Lakukan prediksi dengan model
        predictions = results[0]  # Ambil hasil prediksi pertama

        # Buat mask kosong untuk menandai area objek model
        mask_model = np.zeros_like(edges)

        # Buat overlay transparan untuk segmentasi warna
        overlay = frame.copy()

        # Cek apakah model menggunakan segmentasi atau bounding box
        if predictions.masks is not None and predictions.masks.xy:  # Model segmentasi
            for mask in predictions.masks.xy:
                mask = np.array(mask, np.int32)  # Konversi ke integer
                
                # Isi area objek dengan warna biru transparan
                cv2.fillPoly(overlay, [mask], (255, 0, 0))  # Biru
                
                # Garis hijau di tepi objek
                cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Tambahkan ke mask untuk mengubah tepi Canny dalam area ini
                cv2.fillPoly(mask_model, [mask], 255)
        
        elif predictions.boxes is not None and len(predictions.boxes.xyxy) > 0:  # Model deteksi objek
            for i, box in enumerate(predictions.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Isi area bounding box dengan warna biru transparan
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), thickness=cv2.FILLED)

                # Tambahkan garis hijau pada bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

                # Tandai area objek pada mask
                cv2.rectangle(mask_model, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

                # Tambahkan label jika tersedia
                if hasattr(predictions.boxes, "cls"):  
                    label = f"Objek {int(predictions.boxes.cls[i])}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Gabungkan overlay dengan transparansi 50%
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Buat hasil Canny dalam format BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Ubah warna tepi Canny hanya pada area objek menjadi merah
        edges_bgr[np.where((mask_model == 255) & (edges == 255))] = [0, 0, 255]  # Tepi model merah
        edges_bgr[np.where((mask_model == 0) & (edges == 255))] = [255, 255, 255]  # Tepi lainnya tetap putih

        # Gabungkan hasil
        combined = np.hstack((frame, edges_bgr))

        # Menampilkan hasil kombinasi
        cv2.imshow('Original & Canny Edge Detection + Model Prediction', combined)
        
        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Jalankan program
detect_canny_edges_with_model()
