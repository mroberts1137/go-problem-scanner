import json
import numpy as np


def save_to_json(data, filename):
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    serializable_data = serialize(data)

    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
