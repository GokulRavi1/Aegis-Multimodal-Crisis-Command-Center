"""
Hybrid Geolocation Utilities for Aegis.
Provides multi-tier location detection: EXIF → OCR → Visual AI
"""
from PIL import Image, ExifTags
from geopy.geocoders import Nominatim
import os

class HybridGeolocator:
    """Multi-tier geolocation resolver."""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="aegis_crisis_center", timeout=10)
    
    def get_exif_location(self, image_path: str) -> dict | None:
        """
        Extract GPS coordinates from image EXIF metadata.
        Returns: {"lat": float, "lon": float, "name": str} or None
        """
        try:
            image = Image.open(image_path)
            exif = image._getexif()
            if not exif:
                return None
            
            # Find GPSInfo tag
            gps_info = None
            for key, val in exif.items():
                if ExifTags.TAGS.get(key) == 'GPSInfo':
                    gps_info = val
                    break
            
            if not gps_info:
                return None

            # Helper to convert DMS (Degrees, Minutes, Seconds) to Decimal
            def dms_to_decimal(dms, ref):
                degrees = float(dms[0])
                minutes = float(dms[1])
                seconds = float(dms[2])
                decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                if ref in ['S', 'W']:
                    decimal = -decimal
                return decimal

            # GPS tags: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon
            lat = dms_to_decimal(gps_info[2], gps_info[1])
            lon = dms_to_decimal(gps_info[4], gps_info[3])
            
            # Reverse geocode to get address
            try:
                location = self.geolocator.reverse(f"{lat}, {lon}", language='en')
                address = location.address if location else f"GPS ({lat:.4f}, {lon:.4f})"
            except:
                address = f"GPS ({lat:.4f}, {lon:.4f})"
            
            return {"lat": lat, "lon": lon, "name": address, "source": "EXIF"}
            
        except Exception as e:
            # Silently return None - EXIF not available
            return None
    
    def reverse_geocode(self, lat: float, lon: float) -> str:
        """Convert coordinates to address."""
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", language='en')
            return location.address if location else f"({lat:.4f}, {lon:.4f})"
        except:
            return f"({lat:.4f}, {lon:.4f})"


# Singleton
_geolocator = None
def get_hybrid_geolocator():
    global _geolocator
    if _geolocator is None:
        _geolocator = HybridGeolocator()
    return _geolocator
