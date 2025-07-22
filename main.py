import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

class MotionDetectionCounter:
    def __init__(self, counting_line_y=250, min_area=1000, max_area=50000, 
                 distance_threshold=50, aspect_ratio_range=(0.3, 3.0)):
        """
        Gerçek zamanlı hareket tespit ve sayım sistemi
        
        Args:
            counting_line_y: Sayım çizgisinin Y koordinatı
            min_area: Minimum nesne alanı (küçük gürültüleri filtreler)
            max_area: Maksimum nesne alanı (çok büyük alanları filtreler)
            distance_threshold: Aynı nesne kabul etmek için mesafe eşiği
            aspect_ratio_range: Kutu benzeri nesneler için en-boy oranı aralığı
        """
        self.counting_line_y = counting_line_y
        self.min_area = min_area
        self.max_area = max_area
        self.distance_threshold = distance_threshold
        self.aspect_ratio_range = aspect_ratio_range
        
        # Arka plan çıkarıcısı - optimize edilmiş parametreler
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        # Morfolojik işlemler için kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Sayım ve takip değişkenleri
        self.object_count = 0
        self.tracked_objects = deque(maxlen=100)  # Son 100 nesneyi takip et
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def calculate_fps(self):
        """FPS hesaplama"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Her 30 karede bir güncelle
            elapsed_time = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed_time
            self.fps_start_time = time.time()
    
    def is_box_like_shape(self, contour):
        """Konturu kutu benzeri şekil olup olmadığını kontrol et"""
        # Bounding rectangle al
        x, y, w, h = cv2.boundingRect(contour)
        
        # Alan kontrolü
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False, None, None
        
        # En-boy oranı kontrolü (kutu benzeri şekiller için)
        aspect_ratio = w / h if h > 0 else 0
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False, None, None
        
        # Dikdörtgen alanına oranla kontur alanı (kutu benzeri şekiller daha yoğun)
        rect_area = w * h
        if rect_area > 0:
            solidity = area / rect_area
            if solidity < 0.3:  # Çok parçalı şekilleri filtrele
                return False, None, None
        
        # Merkez koordinatları
        center_x = x + w // 2
        center_y = y + h // 2
        
        return True, (center_x, center_y), (x, y, w, h)
    
    def is_duplicate_object(self, center):
        """Aynı nesnenin tekrar sayılıp sayılmadığını kontrol et"""
        cx, cy = center
        for prev_center in self.tracked_objects:
            prev_cx, prev_cy = prev_center
            distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if distance < self.distance_threshold:
                return True
        return False
    
    def has_crossed_line(self, center):
        """Nesnenin sayım çizgisini geçip geçmediğini kontrol et"""
        _, cy = center
        # Sayım çizgisinin yakınında mı? (±10 piksel tolerans)
        return abs(cy - self.counting_line_y) <= 10
    
    def process_frame(self, frame):
        """Tek bir kareyi işle"""
        # Arka plan çıkarma
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Gürültü azaltma
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Gölgeleri kaldır
        fg_mask[fg_mask == 127] = 0
        
        # Konturları bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            is_box_like, center, bbox = self.is_box_like_shape(contour)
            
            if is_box_like and center:
                # Sayım çizgisini geçen ve daha önce sayılmamış nesneleri say
                if (self.has_crossed_line(center) and 
                    not self.is_duplicate_object(center)):
                    
                    self.object_count += 1
                    self.tracked_objects.append(center)
                    print(f"Yeni nesne tespit edildi! Toplam sayı: {self.object_count}")
                
                detected_objects.append((center, bbox))
        
        return detected_objects, fg_mask
    
    def draw_visualization(self, frame, detected_objects):
        """Profesyonel görselleştirme elemanlarını çiz"""
        frame_height, frame_width = frame.shape[:2]
        
        # Sayım çizgisini çiz (açık turuncu - dikkat çekici)
        cv2.line(frame, (0, self.counting_line_y), (frame_width, self.counting_line_y), 
                (0, 165, 255), 4)  # BGR: açık turuncu
        
        # Tespit edilen nesneleri çiz
        for center, bbox in detected_objects:
            cx, cy = center
            x, y, w, h = bbox
            
            # Açık mavi dikdörtgen kutu (soft ve sade)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 50), 2)  # BGR: açık mavi
            
            # Pastel pembe merkez noktası (gözü yormayan)
            cv2.circle(frame, (cx, cy), 6, (180, 120, 255), -1)  # BGR: pastel pembe
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)   # Beyaz kenar
            
            # Koordinat bilgisi için gölge arka planı
            coord_text = f'({cx},{cy})'
            text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (cx + 8, cy - 25), (cx + text_size[0] + 16, cy - 5), 
                         (30, 30, 30), -1)  # Koyu gri arka plan
            cv2.putText(frame, coord_text, (cx + 12, cy - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ana bilgi paneli için koyu gri arka plan kutusu
        current_time = datetime.now().strftime("%H:%M:%S")
        panel_width = 420
        panel_height = 120
        
        # Yumuşak gölge efekti
        cv2.rectangle(frame, (8, 8), (panel_width + 15, panel_height + 15), (0, 0, 0), -1)
        # Ana panel
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (30, 30, 30), -1)  # Koyu gri
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (80, 80, 80), 2)   # Kenar çizgisi
        
        # Başlık çizgisi (dekoratif)
        cv2.line(frame, (15, 35), (panel_width - 15, 35), (0, 165, 255), 2)
        
        # Sayaç bilgisi (büyük ve belirgin - yeşil)
        cv2.putText(frame, f'TOPLAM NESNE: {self.object_count}', (20, 55),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 255, 100), 2)  # Açık yeşil
        
        # Zaman bilgisi (beyaz)
        cv2.putText(frame, f'ZAMAN: {current_time}', (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # FPS bilgisi (açık mavi)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (250, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 180, 50), 2)  # Açık mavi
        
        # Sistem durumu göstergesi (küçük ikon)
        cv2.circle(frame, (panel_width - 30, 25), 8, (100, 255, 100), -1)  # Yeşil durum ışığı
        cv2.circle(frame, (panel_width - 30, 25), 8, (255, 255, 255), 1)   # Beyaz kenar
        cv2.putText(frame, 'AKTIF', (panel_width - 70, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Sayım çizgisi bilgisi (modern stil)
        line_info_text = f'SAYIM CIZGISI: Y={self.counting_line_y}'
        line_text_size = cv2.getTextSize(line_info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        line_bg_x = frame_width - line_text_size[0] - 20
        line_bg_y = self.counting_line_y - 35
        
        # Sayım çizgisi bilgisi için arka plan
        cv2.rectangle(frame, (line_bg_x - 10, line_bg_y), 
                     (frame_width - 10, self.counting_line_y - 10), (30, 30, 30), -1)
        cv2.rectangle(frame, (line_bg_x - 10, line_bg_y), 
                     (frame_width - 10, self.counting_line_y - 10), (0, 165, 255), 2)
        
        cv2.putText(frame, line_info_text, (line_bg_x, self.counting_line_y - 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sayım çizgisi kenarlarında ok işaretleri
        arrow_size = 15
        # Sol ok
        cv2.arrowedLine(frame, (arrow_size, self.counting_line_y), 
                       (0, self.counting_line_y), (0, 165, 255), 3, tipLength=0.3)
        # Sağ ok
        cv2.arrowedLine(frame, (frame_width - arrow_size, self.counting_line_y), 
                       (frame_width, self.counting_line_y), (0, 165, 255), 3, tipLength=0.3)
        
        # Sağ alt köşede şık branding
        brand_text = "BurakAI Systems"
        brand_text_size = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        brand_x = frame_width - brand_text_size[0] - 15
        brand_y = frame_height - 15
        
        # Branding için hafif arka plan
        cv2.rectangle(frame, (brand_x - 8, brand_y - 20), 
                     (frame_width - 5, frame_height - 5), (0, 0, 0), -1)  # Gölge
        cv2.rectangle(frame, (brand_x - 5, brand_y - 18), 
                     (frame_width - 8, frame_height - 8), (50, 50, 50), -1)  # Ana arka plan
        
        cv2.putText(frame, brand_text, (brand_x, brand_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)  # Gri tonda
        
        # Küçük dekoratif çizgi
        cv2.line(frame, (brand_x, brand_y - 2), (frame_width - 8, brand_y - 2), 
                (0, 165, 255), 1)
        
        return frame
    
    def run(self):
        """Ana sistem döngüsü"""
        # Kamerayi başlat
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Hata: Kamera açılamadı!")
            return
        
        # Kamera optimizasyonu
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Sistem başlatıldı!")
        print("- Yeşil kutular: Tespit edilen nesneler")
        print("- Kırmızı noktalar: Nesne merkezleri") 
        print("- Mavi çizgi: Sayım çizgisi")
        print("- 'q' tuşu ile çıkış")
        print(f"- Sayım çizgisi Y koordinatı: {self.counting_line_y}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Kare okunamadı!")
                    break
                
                # FPS hesaplama
                self.calculate_fps()
                
                # Frame işleme
                detected_objects, fg_mask = self.process_frame(frame)
                
                # Görselleştirme
                frame = self.draw_visualization(frame, detected_objects)
                
                # Pencereleri göster
                cv2.imshow('Hareket Tespit ve Sayim Sistemi', frame)
                cv2.imshow('Arka Plan Maske', fg_mask)
                
                # Çıkış kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # 'r' ile sayacı sıfırla
                    self.object_count = 0
                    self.tracked_objects.clear()
                    print("Sayaç sıfırlandı!")
                
        except KeyboardInterrupt:
            print("Sistem kullanıcı tarafından durduruldu.")
        
        finally:
            # Kaynakları temizle
            print(f"\nSon durum:")
            print(f"Toplam tespit edilen nesne: {self.object_count}")
            print("Sistem kapatılıyor...")
            
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Ana fonksiyon"""
    # Sistem parametrelerini ayarla
    detector = MotionDetectionCounter(
        counting_line_y=250,          # Sayım çizgisi Y koordinatı
        min_area=1000,                # Minimum nesne alanı
        max_area=50000,               # Maksimum nesne alanı  
        distance_threshold=50,        # Aynı nesne mesafe eşiği
        aspect_ratio_range=(0.3, 3.0) # Kutu benzeri şekiller için en-boy oranı
    )
    
    # Sistemi başlat
    detector.run()

if __name__ == "__main__":
    main()