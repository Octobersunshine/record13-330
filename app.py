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


def _interpret_result(test_name, p_value, alpha, group_count):
    significant = p_value < alpha
    if significant:
        conclusion = (
            "在 α={} 的显著性水平下，{}检验结果显著 (p={:.6g})，"
            "拒绝方差齐性假设，认为各组的方差不相等（方差不齐）"
            .format(alpha, test_name, p_value)
        )
        suggestions = []
        if group_count == 2:
            suggestions.append(
                "两组比较：建议使用 Welch t 检验代替 Student t 检验，"
                "Welch t 检验不要求方差齐性"
            )
        else:
            suggestions.append(
                "多组比较：建议使用 Welch ANOVA 代替传统 ANOVA，"
                "Welch ANOVA 不要求方差齐性"
            )
            suggestions.append(
                "事后多重比较：建议使用 Games-Howell 检验，"
                "该检验不要求方差齐性且适用于非等样本量"
            )
        suggestions.append("可尝试对数据做变换（如对数变换、平方根变换）以稳定方差")
        suggestions.append("考虑使用非参数方法（如 Kruskal-Wallis 检验）作为替代")
        return {
            "conclusion": conclusion,
            "variance_homogeneous": False,
            "significant": True,
            "suggestions": suggestions,
        }
    else:
        conclusion = (
            "在 α={} 的显著性水平下，{}检验结果不显著 (p={:.6g})，"
            "不能拒绝方差齐性假设，可认为各组方差相等（方差齐）"
            .format(alpha, test_name, p_value)
        )
        return {
            "conclusion": conclusion,
            "variance_homogeneous": True,
            "significant": False,
            "suggestions": [],
        }


@app.route("/test/levene", methods=["POST"])
def levene_test():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        center = data.get("center", "median")
        alpha = float(data.get("alpha", 0.05))
        stat, p = _levene(groups, center)
        interpretation = _interpret_result("Levene", p, alpha, len(groups))
        return jsonify({
            "test": "levene",
            "center": center,
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "significant": interpretation["significant"],
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
            "interpretation": interpretation,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test/bartlett", methods=["POST"])
def bartlett_test():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        alpha = float(data.get("alpha", 0.05))
        normality_alpha = float(data.get("normality_alpha", 0.05))
        normal, non_normal, normality_details = _check_normality(groups, normality_alpha)
        stat, p = _bartlett(groups)
        interpretation = _interpret_result("Bartlett", p, alpha, len(groups))
        if not normal:
            interpretation["warning"] = (
                "Bartlett 检验假设数据服从正态分布，但第 {} 组未通过 Shapiro-Wilk "
                "正态性检验 (α={})，结果可能不可靠，建议使用 Levene 检验"
                .format(", ".join(str(i) for i in non_normal), normality_alpha)
            )
            interpretation["suggestions"].insert(0, "数据非正态，Bartlett 检验结果可能不可靠，建议以 Levene 检验结果为准")
        result = {
            "test": "bartlett",
            "statistic": float(stat),
            "p_value": float(p),
            "alpha": alpha,
            "significant": interpretation["significant"],
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
            "normality_assumed": normal,
            "normality_alpha": normality_alpha,
            "normality_details": normality_details,
            "interpretation": interpretation,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test/both", methods=["POST"])
def both_tests():
    try:
        data = request.get_json(force=True)
        groups = _parse_groups(data)
        center = data.get("center", "median")
        alpha = float(data.get("alpha", 0.05))
        normality_alpha = float(data.get("normality_alpha", 0.05))
        normal, non_normal, normality_details = _check_normality(groups, normality_alpha)
        l_stat, l_p = _levene(groups, center)
        b_stat, b_p = _bartlett(groups)
        l_interp = _interpret_result("Levene", l_p, alpha, len(groups))
        b_interp = _interpret_result("Bartlett", b_p, alpha, len(groups))
        if not normal:
            b_interp["warning"] = (
                "Bartlett 检验假设数据服从正态分布，但第 {} 组未通过 Shapiro-Wilk "
                "正态性检验 (α={})，结果可能不可靠，建议以 Levene 检验结果为准"
                .format(", ".join(str(i) for i in non_normal), normality_alpha)
            )
            b_interp["suggestions"].insert(0, "数据非正态，Bartlett 检验结果可能不可靠，建议以 Levene 检验结果为准")
        levene_result = {
            "test": "levene",
            "center": center,
            "statistic": float(l_stat),
            "p_value": float(l_p),
            "significant": l_interp["significant"],
            "interpretation": l_interp,
        }
        bartlett_result = {
            "test": "bartlett",
            "statistic": float(b_stat),
            "p_value": float(b_p),
            "significant": b_interp["significant"],
            "normality_assumed": normal,
            "interpretation": b_interp,
        }
        recommended_test = None
        if not normal:
            recommended_test = "levene"
            levene_result["recommended"] = True
        return jsonify({
            "levene": levene_result,
            "bartlett": bartlett_result,
            "group_count": len(groups),
            "group_sizes": [int(g.size) for g in groups],
            "alpha": alpha,
            "normality_alpha": normality_alpha,
            "normality_assumed": normal,
            "normality_details": normality_details,
            "recommended_test": recommended_test,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
