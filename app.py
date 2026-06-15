from flask import Flask, request, jsonify
from scipy import stats
import numpy as np

app = Flask(__name__)


def _parse_groups(data):
    groups = []
    for g in data.get("groups", []):
        arr = np.array(g, dtype=float)
        if arr.ndim != 1:
            raise ValueError("每个组必须是一维数组")
        if arr.size < 2:
            raise ValueError("每个组至少需要 2 个观测值")
        groups.append(arr)
    if len(groups) < 2:
        raise ValueError("至少需要 2 个组")
    return groups


def _levene(groups, center):
    center_map = {
        "median": "median",
        "mean": "mean",
        "trimmed": "trimmed",
    }
    c = center_map.get(center, "median")
    stat, p = stats.levene(*groups, center=c)
    return stat, p


def _bartlett(groups):
    stat, p = stats.bartlett(*groups)
    return stat, p


@app.route("/test/levene", methods=["POST"])
def levene_test():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        center = data.get("center", "median")
        stat, p = _levene(groups, center)
        return jsonify({
            "test": "levene",
            "center": center,
            "statistic": float(stat),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test/bartlett", methods=["POST"])
def bartlett_test():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        stat, p = _bartlett(groups)
        return jsonify({
            "test": "bartlett",
            "statistic": float(stat),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test/both", methods=["POST"])
def both_tests():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        center = data.get("center", "median")
        l_stat, l_p = _levene(groups, center)
        b_stat, b_p = _bartlett(groups)
        return jsonify({
            "levene": {
                "test": "levene",
                "center": center,
                "statistic": float(l_stat),
                "p_value": float(l_p),
                "significant_005": bool(l_p < 0.05),
            },
            "bartlett": {
                "test": "bartlett",
                "statistic": float(b_stat),
                "p_value": float(b_p),
                "significant_005": bool(b_p < 0.05),
            },
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
