import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_state(decision: bool, confidence: float, ta: str = "пенсионеры"):
    return {
        "audiences": [ta],
        "counts": [5],
        "question": "возьмут кредит?",
        "result": {
            "results": [
                {
                    "profile": {"target_audience_name": ta},
                    "survey_responses": [{
                        "full_state": {
                            "final_decision": {
                                "decision": decision,
                                "confidence": confidence,
                                "reasoning": "потому что так"
                            }
                        }
                    }]
                }
            ]
        }
    }
