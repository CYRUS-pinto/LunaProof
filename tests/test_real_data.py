"""
Tests for Real PDS Lunar Data Loader & Fetcher
"""
import pytest
from lunaproof.real_data import load_real_lunar_pds_image, RealPDSDataFetcher


def test_load_real_lunar_pds_image():
    img = load_real_lunar_pds_image(crop_size=(256, 256))
    assert img.shape == (256, 256)
    assert img.dtype == "uint8"
    assert img.std() > 5.0  # Has real surface texture


def test_real_pds_data_fetcher():
    fetcher = RealPDSDataFetcher()
    img_a, img_b, meta = fetcher.fetch_real_pair()
    assert img_a.shape == (512, 512)
    assert img_b.shape == (512, 512)
    assert isinstance(meta["is_real_pds_data"], bool)
