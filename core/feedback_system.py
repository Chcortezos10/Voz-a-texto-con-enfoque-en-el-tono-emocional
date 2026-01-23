import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class FeedbackCollector:
    def __init__(self, storage_dir: str = "./feedback_data"):
        self.storage_dir = storage_dir
        self.predictions_file = os.path.join(storage_dir, "predictions.json")
        self.validated_file = os.path.join(storage_dir, "validated.json")
        os.makedirs(storage_dir, exist_ok=True)
        self._predictions = self._load_json(self.predictions_file)
        self._validated = self._load_json(self.validated_file)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_json(self, path: str, data: Dict[str, Any]):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")

    def save_prediction(
        self,
        audio_path: str,
        predicted_label: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        prediction_id = str(uuid.uuid4())
        self._predictions[prediction_id] = {
            "audio_path": audio_path,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "validated": False
        }
        self._save_json(self.predictions_file, self._predictions)
        return prediction_id

    def validate_prediction(
        self,
        prediction_id: str,
        correct_label: str,
        user_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        if prediction_id not in self._predictions:
            raise ValueError(f"Prediction {prediction_id} not found")

        pred = self._predictions[prediction_id]
        pred["validated"] = True
        pred["correct_label"] = correct_label
        pred["user_comment"] = user_comment
        pred["validation_timestamp"] = datetime.now().isoformat()

        self._validated[prediction_id] = pred
        self._save_json(self.validated_file, self._validated)
        self._save_json(self.predictions_file, self._predictions)

        return pred

    def get_validated_count(self) -> int:
        return len(self._validated)

    def get_pending_count(self) -> int:
        return sum(1 for p in self._predictions.values() if not p.get("validated", False))

    def get_training_data(self) -> List[Dict[str, Any]]:
        return list(self._validated.values())

    def export_for_training(self, output_path: str) -> str:
        data = self.get_training_data()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
