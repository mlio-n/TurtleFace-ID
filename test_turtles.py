import sys
from pathlib import Path

# Proje dizinini yola ekleyelim
sys.path.insert(0, str(Path(__file__).parent))

from turtlefaceid.agents.identification_agent import IdentificationAgent
from turtlefaceid.utils.image_utils import ImageUtils

def run_tests():
    agent = IdentificationAgent(use_demo_db=True, detection_threshold=0.3, match_threshold=0.4)
    
    test_images = [
        r"C:\Users\msari\Downloads\turtle1.jpg",
        r"C:\Users\msari\Downloads\turtle2.jpeg",
        r"C:\Users\msari\Downloads\turtle3.jpeg"
    ]
    
    for img_path in test_images:
        path = Path(img_path)
        print(f"\n{'='*50}")
        print(f"Test Edilen Fotoğraf: {path.name}")
        print(f"{'='*50}")
        
        if not path.exists():
            print(f"HATA: Dosya bulunamadı - {path}")
            continue
            
        result = agent.identify_from_path(path)
        
        if not result.success:
            print(f"Pipeline Başarısız: {result.error_message}")
            continue
            
        print(f"1. Yüz Tespiti: Başarılı (Güven: {result.detection_result.confidence:.2%})")
        print(f"   - Kırpılan Alan BBox: {result.detection_result.bounding_box}")
        
        if result.scute_map.success:
            print(f"2. Pul Çıkarımı: Başarılı ({result.scute_map.scute_count} pul tespit edildi)")
        else:
            print(f"2. Pul Çıkarımı: Başarısız ({result.scute_map.error_message})")
            
        mr = result.match_result
        if mr.matched:
            print(f"3. Kimlik Eşleştirme: EŞLEŞME BULUNDU!")
            print(f"   - Kaplumbağa: {mr.top_match.name} ({mr.top_match.turtle_id})")
            print(f"   - Benzerlik Skoru: {mr.similarity_pct}")
            print(f"   - Güven Seviyesi: {mr.confidence_level.value}")
        else:
            print(f"3. Kimlik Eşleştirme: BİLİNMEYEN BİREY")
            print(f"   - En Yüksek Benzerlik: {mr.similarity_pct}")
            print(f"   - Muhtemelen veritabanında olmayan yeni bir kaplumbağa.")
            
        print(f"Toplam Süre: {result.total_time_ms:.0f} ms")

if __name__ == "__main__":
    run_tests()
