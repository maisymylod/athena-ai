"""Tests for the Flask API endpoint."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from server.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Create a JPEG image in memory."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestIndexRoute:
    def test_get_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Athena" in resp.data


class TestDetectEndpoint:
    def test_no_file(self, client):
        resp = client.post("/api/detect")
        assert resp.status_code == 400
        assert b"No image file" in resp.data

    def test_invalid_extension(self, client):
        data = {"image": (io.BytesIO(b"not an image"), "test.txt")}
        resp = client.post("/api/detect", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert b"Unsupported file type" in resp.data

    @patch("server.app._get_inference")
    def test_no_model_returns_503(self, mock_get, client, sample_image_bytes):
        mock_get.return_value = None
        data = {"image": (sample_image_bytes, "test.jpg")}
        resp = client.post("/api/detect", data=data, content_type="multipart/form-data")
        assert resp.status_code == 503
        assert b"No trained model" in resp.data

    @patch("server.app._get_inference")
    def test_successful_detection(self, mock_get, client, sample_image_bytes):
        mock_inference = MagicMock()
        mock_inference.predict.return_value = (
            True,
            0.95,
            ["ML model confidence: 95%", "Very high synthetic probability"],
        )
        mock_get.return_value = mock_inference

        data = {"image": (sample_image_bytes, "test.jpg")}
        resp = client.post("/api/detect", data=data, content_type="multipart/form-data")

        assert resp.status_code == 200
        json_data = resp.get_json()
        assert json_data["is_synthetic"] is True
        assert json_data["confidence"] == 0.95
        assert json_data["label"] == "AI-Generated"
        assert len(json_data["indicators"]) == 2

    @patch("server.app._get_inference")
    def test_real_image_detection(self, mock_get, client, sample_image_bytes):
        mock_inference = MagicMock()
        mock_inference.predict.return_value = (
            False,
            0.15,
            ["ML model confidence (real): 85%"],
        )
        mock_get.return_value = mock_inference

        data = {"image": (sample_image_bytes, "photo.png")}
        resp = client.post("/api/detect", data=data, content_type="multipart/form-data")

        assert resp.status_code == 200
        json_data = resp.get_json()
        assert json_data["is_synthetic"] is False
        assert json_data["label"] == "Real"


class TestCORS:
    def test_cors_headers(self, client):
        resp = client.get("/")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
