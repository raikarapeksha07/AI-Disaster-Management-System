from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from statistics import pstdev
from typing import Any

import joblib
import pandas as pd


class PredictionError(Exception):
    """Raised when the model cannot produce a flood prediction."""


class FloodModelRegistry:
    """State-aware model registry to support multi-state expansion."""

    def __init__(self, model_paths: dict[str, Path], default_state: str) -> None:
        self._model_paths = {state.lower(): Path(path) for state, path in model_paths.items()}
        self._default_state = default_state.lower()
        self._loaded_models: dict[str, Any] = {}

    def resolve_state_key(self, state: str) -> str:
        state_key = state.lower() if state else self._default_state
        if state_key not in self._model_paths:
            state_key = self._default_state
        return state_key

    def get_model_path(self, state: str) -> Path:
        return self._model_paths[self.resolve_state_key(state)]

    def get_model(self, state: str) -> Any:
        state_key = self.resolve_state_key(state)
        if state_key in self._loaded_models:
            return self._loaded_models[state_key]

        model_path = self._model_paths[state_key]
        if not model_path.exists():
            raise PredictionError(f"Model file not found for state '{state_key}': {model_path}")

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise PredictionError(f"Unable to load model for state '{state_key}': {exc}") from exc

        self._loaded_models[state_key] = model
        return model

    def get_model_info(self, state: str) -> dict[str, Any]:
        state_key = self.resolve_state_key(state)
        model = self.get_model(state_key)
        feature_names = FloodPredictor.resolve_model_features(model)
        return {
            "state": state_key,
            "path": str(self.get_model_path(state_key)),
            "model_type": type(model).__name__,
            "feature_names": feature_names,
            "n_features": int(getattr(model, "n_features_in_", len(feature_names))),
            "supports_predict_proba": hasattr(model, "predict_proba"),
        }

    def set_model_path(self, state: str, model_path: Path) -> None:
        state_key = self.resolve_state_key(state)
        self._model_paths[state_key] = Path(model_path)
        self._loaded_models.pop(state_key, None)


class RainfallFeatureStore:
    """Maintains recent rainfall history per location to build rolling features."""

    def __init__(self, max_history: int = 60) -> None:
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_history))

    def build_features(
        self,
        location_key: str,
        rainfall: float,
        temperature: float,
        humidity: float,
        state: str,
    ) -> dict[str, float]:
        history = list(self._history[location_key])
        combined = history + [float(rainfall)]

        features = {
            "PRECTOTCORR": float(rainfall),
            "rain_lag1": self._lag(history, 1, rainfall),
            "rain_lag2": self._lag(history, 2, rainfall),
            "rain_lag3": self._lag(history, 3, rainfall),
            "rain_3day_sum": self._window_sum(combined, 3),
            "rain_5day_sum": self._window_sum(combined, 5),
            "rain_7day_sum": self._window_sum(combined, 7),
            "rain_10day_sum": self._window_sum(combined, 10),
            "rain_15day_sum": self._window_sum(combined, 15),
            "rain_30day_sum": self._window_sum(combined, 30),
            "rain_ma7": self._window_mean(combined, 7),
            "rain_ma30": self._window_mean(combined, 30),
            "rain_diff": float(rainfall) - self._lag(history, 1, rainfall),
            "rain_std7": self._window_std(combined, 7),
            "rain_std30": self._window_std(combined, 30),
            "T2M": float(temperature),
            "RH2M": float(humidity),
            "STATE_CODE": float(self._state_code(state)),
        }

        self._history[location_key].append(float(rainfall))
        return features

    @staticmethod
    def _lag(history: list[float], steps: int, fallback: float) -> float:
        if len(history) >= steps:
            return float(history[-steps])
        return float(fallback)

    @staticmethod
    def _window_sum(series: list[float], window: int) -> float:
        values = series[-window:] if series else [0.0]
        return float(sum(values))

    @staticmethod
    def _window_mean(series: list[float], window: int) -> float:
        values = series[-window:] if series else [0.0]
        return float(sum(values) / len(values))

    @staticmethod
    def _window_std(series: list[float], window: int) -> float:
        values = series[-window:] if series else [0.0]
        if len(values) == 1:
            return 0.0
        return float(pstdev(values))

    @staticmethod
    def _state_code(state: str) -> int:
        normalized = (state or "").strip().lower()
        state_codes = {
            "andhra pradesh": 1,
            "arunachal pradesh": 2,
            "assam": 3,
            "bihar": 4,
            "chhattisgarh": 5,
            "goa": 6,
            "gujarat": 7,
            "haryana": 8,
            "himachal pradesh": 9,
            "jharkhand": 10,
            "karnataka": 11,
            "kerala": 12,
            "madhya pradesh": 13,
            "maharashtra": 14,
            "manipur": 15,
            "meghalaya": 16,
            "mizoram": 17,
            "nagaland": 18,
            "odisha": 19,
            "punjab": 20,
            "rajasthan": 21,
            "sikkim": 22,
            "tamil nadu": 23,
            "telangana": 24,
            "tripura": 25,
            "uttar pradesh": 26,
            "uttarakhand": 27,
            "west bengal": 28,
        }
        return state_codes.get(normalized, 0)


class FloodPredictor:
    """Builds flood features and returns probability + categorical risk."""

    DEFAULT_FEATURES = ["PRECTOTCORR", "T2M", "RH2M"]

    def __init__(self, model_registry: FloodModelRegistry, feature_store: RainfallFeatureStore) -> None:
        self._model_registry = model_registry
        self._feature_store = feature_store

    def predict(
        self,
        state: str,
        location_key: str,
        rainfall: float,
        temperature: float,
        humidity: float,
    ) -> dict[str, Any]:
        model = self._model_registry.get_model(state=state)
        model_features = self.resolve_model_features(model)

        features = self._feature_store.build_features(
            location_key=location_key,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity,
            state=state,
        )

        try:
            filtered_features = {name: features[name] for name in model_features}
        except KeyError as exc:
            raise PredictionError(f"Missing feature required by model: {exc.args[0]}") from exc

        frame = pd.DataFrame([filtered_features], columns=model_features)
        self._validate_frame(frame)

        print("Features sent to model:", frame.columns.tolist())
        print("Feature values:", frame.to_dict(orient="records")[0])

        flood_probability = self._predict_probability(model, frame)

        return {
            "flood_probability": flood_probability,
            "flood_prediction": self._probability_to_risk(flood_probability),
            "features": filtered_features,
            "model_features": model_features,
        }

    @staticmethod
    def _predict_probability(model: Any, frame: pd.DataFrame) -> float:
        try:
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(frame)[0][1])
            elif hasattr(model, "predict"):
                prediction = float(model.predict(frame)[0])
                probability = prediction
            else:
                raise PredictionError("Loaded model supports neither predict_proba nor predict.")
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc

        if not 0.0 <= probability <= 1.0:
            raise PredictionError(f"Model returned invalid flood probability: {probability}")
        return probability

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if frame.empty:
            raise PredictionError("Prediction frame is empty.")
        if frame.isnull().any().any():
            raise PredictionError("Prediction frame contains null values.")

    @staticmethod
    def _probability_to_risk(probability: float) -> str:
        if probability > 0.7:
            return "HIGH"
        if probability >= 0.4:
            return "MEDIUM"
        return "LOW"

    @classmethod
    def resolve_model_features(cls, model: Any) -> list[str]:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return [str(name) for name in feature_names]

        n_features = int(getattr(model, "n_features_in_", len(cls.DEFAULT_FEATURES)))
        if n_features == len(cls.DEFAULT_FEATURES):
            return cls.DEFAULT_FEATURES[:]
        raise PredictionError(
            "Model does not expose feature_names_in_ and its feature count cannot be mapped safely."
        )
