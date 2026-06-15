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


def _check_normality(groups, alpha=0.05):
    details = []
    non_normal_indices = []
    for i, g in enumerate(groups):
        if g.size >= 3:
            stat, p_val = stats.shapiro(g)
            passed = bool(p_val >= alpha)
            details.append({
                "group_index": i,
                "size": int(g.size),
                "shapiro_statistic": float(stat),
                "shapiro_p_value": float(p_val),
                "normal_at_alpha": passed,
            })
            if not passed:
                non_normal_indices.append(i)
        else:
            details.append({
                "group_index": i,
                "size": int(g.size),
                "shapiro_statistic": None,
                "shapiro_p_value": None,
                "normal_at_alpha": False,
                "note": "样本量 < 3，无法进行 Shapiro-Wilk 正态性检验",
            })
            non_normal_indices.append(i)
    normal = len(non_normal_indices) == 0
    return normal, non_normal_indices, details


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
        alpha = float(data.get("normality_alpha", 0.05))
        normal, non_normal, normality_details = _check_normality(groups, alpha)
        stat, p = _bartlett(groups)
        result = {
            "test": "bartlett",
            "statistic": float(stat),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
            "normality_assumed": normal,
            "normality_alpha": alpha,
            "normality_details": normality_details,
        }
        if not normal:
            result["warning"] = (
                "Bartlett 检验假设数据服从正态分布，但第 {} 组未通过 Shapiro-Wilk "
                "正态性检验 (α={})，结果可能不可靠，建议使用 Levene 检验"
                .format(", ".join(str(i) for i in non_normal), alpha)
            )
            result["recommendation"] = "levene"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test/both", methods=["POST"])
def both_tests():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        center = data.get("center", "median")
        alpha = float(data.get("normality_alpha", 0.05))
        normal, non_normal, normality_details = _check_normality(groups, alpha)
        l_stat, l_p = _levene(groups, center)
        b_stat, b_p = _bartlett(groups)
        bartlett_result = {
            "test": "bartlett",
            "statistic": float(b_stat),
            "p_value": float(b_p),
            "significant_005": bool(b_p < 0.05),
            "normality_assumed": normal,
        }
        if not normal:
            bartlett_result["warning"] = (
                "Bartlett 检验假设数据服从正态分布，但第 {} 组未通过 Shapiro-Wilk "
                "正态性检验 (α={})，结果可能不可靠，建议使用 Levene 检验"
                .format(", ".join(str(i) for i in non_normal), alpha)
            )
            bartlett_result["recommendation"] = "levene"
        levene_result = {
            "test": "levene",
            "center": center,
            "statistic": float(l_stat),
            "p_value": float(l_p),
            "significant_005": bool(l_p < 0.05),
        }
        if not normal:
            levene_result["note"] = "数据非正态，Levene 检验更稳健，推荐使用本结果"
            levene_result["recommended"] = True
        return jsonify({
            "levene": levene_result,
            "bartlett": bartlett_result,
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
            "normality_alpha": alpha,
            "normality_assumed": normal,
            "normality_details": normality_details,
            "recommended_test": "levene" if not normal else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
