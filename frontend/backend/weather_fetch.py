from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any

import requests

logger = logging.getLogger("weather")


class WeatherAPIError(Exception):
    """Raised when the weather provider returns an invalid response."""


@dataclass
class WeatherSnapshot:
    city: str
    country: str
    temperature: float
    humidity: float
    condition: str
    rainfall: float
    aqi: dict[str, Any]
    hourly: list[dict[str, Any]]
    daily: list[dict[str, Any]]


class OpenWeatherClient:
    """Fetches live weather data from OpenWeatherMap endpoints."""

    CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

    def __init__(self, api_key: str, timeout_seconds: int = 15) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        logger.info("Weather client initialized")

    def fetch_live_weather(self, lat: float, lon: float) -> WeatherSnapshot:
        if not self.api_key:
            raise WeatherAPIError("OPENWEATHER_API_KEY is not configured.")

        logger.info("Fetching current weather for lat=%s lon=%s", lat, lon)
        current_data = self._get_json(self.CURRENT_URL, lat=lat, lon=lon)

        if "main" not in current_data or "weather" not in current_data:
            raise WeatherAPIError("Unexpected response received from weather API.")

        rainfall = self._extract_rainfall_mm(current_data)
        city = current_data.get("name", "Unknown")
        country = current_data.get("sys", {}).get("country", "IN")
        hourly, daily = self._fetch_forecast(lat=lat, lon=lon)
        aqi = self._fetch_air_quality(lat=lat, lon=lon)

        logger.info("Weather fetch complete for city=%s rainfall=%s aqi=%s", city, rainfall, aqi["value"])

        return WeatherSnapshot(
            city=city,
            country=country,
            temperature=float(current_data["main"].get("temp", 0.0)),
            humidity=float(current_data["main"].get("humidity", 0.0)),
            condition=current_data["weather"][0].get("main", "Unknown"),
            rainfall=rainfall,
            aqi=aqi,
            hourly=hourly,
            daily=daily,
        )

    def _fetch_forecast(self, lat: float, lon: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            forecast_data = self._get_json(self.FORECAST_URL, lat=lat, lon=lon)
        except WeatherAPIError as exc:
            logger.warning("Forecast fetch skipped: %s", exc)
            return [], []

        items = forecast_data.get("list", [])
        if not items:
            return [], []

        hourly = []
        for item in items[:8]:
            rain_mm = self._extract_forecast_rainfall_mm(item)
            hourly.append(
                {
                    "time": item.get("dt_txt", "").split(" ")[-1][:5],
                    "temp": float(item.get("main", {}).get("temp", 0.0)),
                    "rain": rain_mm,
                }
            )

        daily_minmax: dict[str, dict[str, float]] = defaultdict(dict)
        for item in items:
            dt_text = item.get("dt_txt", "")
            if not dt_text:
                continue
            date_key = dt_text.split(" ")[0]
            temp_min = float(item.get("main", {}).get("temp_min", 0.0))
            temp_max = float(item.get("main", {}).get("temp_max", 0.0))
            rain_mm = self._extract_forecast_rainfall_mm(item)

            record = daily_minmax[date_key]
            record["date"] = date_key
            record["min"] = temp_min if "min" not in record else min(record["min"], temp_min)
            record["max"] = temp_max if "max" not in record else max(record["max"], temp_max)
            record["rain"] = record.get("rain", 0.0) + rain_mm

        daily = []
        for _, payload in sorted(daily_minmax.items())[:7]:
            daily.append(payload)

        return hourly, daily

    def _fetch_air_quality(self, lat: float, lon: float) -> dict[str, Any]:
        try:
            pollution_data = self._get_json(self.AIR_POLLUTION_URL, lat=lat, lon=lon)
        except WeatherAPIError as exc:
            logger.warning("AQI fetch skipped: %s", exc)
            return {"index": 0, "value": 0, "label": "Unavailable"}

        items = pollution_data.get("list", [])
        if not items:
            return {"index": 0, "value": 0, "label": "Unavailable"}

        aqi_index = int(items[0].get("main", {}).get("aqi", 0))
        aqi_value_map = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
        aqi_label_map = {
            1: "Good",
            2: "Moderate",
            3: "Poor",
            4: "Unhealthy",
            5: "Very Poor",
        }
        return {
            "index": aqi_index,
            "value": aqi_value_map.get(aqi_index, 0),
            "label": aqi_label_map.get(aqi_index, "Unavailable"),
        }

    def _get_json(self, url: str, lat: float, lon: float) -> dict[str, Any]:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = self._session.get(url, params=params, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise WeatherAPIError(f"Unable to reach weather API: {exc}") from exc

        if response.status_code == 401:
            raise WeatherAPIError("OpenWeather API authentication failed (401). Check OPENWEATHER_API_KEY.")
        if response.status_code >= 400:
            raise WeatherAPIError(f"Weather API request failed with status {response.status_code}.")

        try:
            return response.json()
        except ValueError as exc:
            raise WeatherAPIError("Weather API returned invalid JSON.") from exc

    @staticmethod
    def _extract_rainfall_mm(current_data: dict[str, Any]) -> float:
        rain_data = current_data.get("rain", {})

        if "1h" in rain_data:
            return float(rain_data["1h"])
        if "3h" in rain_data:
            return float(rain_data["3h"]) / 3.0

        return 0.0

    @staticmethod
    def _extract_forecast_rainfall_mm(item: dict[str, Any]) -> float:
        rain_data = item.get("rain", {})
        if "3h" in rain_data:
            return float(rain_data["3h"])
        if "1h" in rain_data:
            return float(rain_data["1h"])
        return 0.0
