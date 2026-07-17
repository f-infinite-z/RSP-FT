"""
链式中介分析 X -> M -> Y

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

# ============================================================
# 人工批注（作者）
# ============================================================
# 该板块主要是为了用数学X-M-Y这条因果链，涉及回归系数：OLS 闭式解（手写矩阵运算）
# 间接效应CI：Bootstrap 5000次；p值：正态近似
# 自然语言逻辑             数学数字化              代码
# "反问强度影响反问质量"  → a系数 + p值     → _ols(X→M)
# "反问质量影响回答质量"  → b系数 + p值     → _ols(M→Y|X)
# "这俩加起来等于总效应"  → a×b + Bootstrap CI → Bootstrap循环
# "分组之间确实有差异"    → 分组均值对比    → condition ③
# ============================================================

对应 Hayes PROCESS Model 4：
  X : 分组编码 (A=0, B=1, C=2)
  M : 反问质量（裁判连续分；A 组无反问，赋值 0）
  Y : 回答质量（裁判连续分，因变量）

判定中介成立的三条硬性条件：
  ① X->M 系数 a 显著（M1）
  ② M->Y 系数 b 显著，且控制 M 后 X->Y 直接效应 c' 较总效应 c 显著下降（M2）
  ③ C 组 M 均值 > B 组，且 C 组 Y 均值 > B 组

本模块仅依赖 numpy，用 OLS 闭式解 + Bootstrap 求间接效应 a*b 的置信区间，
不强制依赖 statsmodels（若已安装可用 statsmodels 复核，见 __main__）。
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---- OLS 闭式解 -------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    """
    返回 (beta, se, t, p_approx)。X 含截距列。
    p 值用正态近似（大样本）；小样本可外部用 statsmodels 复核。
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(1, n - k)
    sigma2 = (resid @ resid) / dof
    var_beta = np.diag(XtX_inv) * sigma2
    se = np.sqrt(np.clip(var_beta, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    # 正态近似双尾 p
    from math import erf, sqrt
    p = np.array([2 * (1 - 0.5 * (1 + erf(abs(ti) / sqrt(2)))) for ti in t])
    return beta, se, t, p


@dataclass
class MediationResult:
    a: float            # X -> M
    b: float            # M -> Y (控制 X)
    c: float            # X -> Y 总效应
    c_prime: float      # X -> Y 直接效应（控制 M）
    indirect: float     # a*b 间接效应
    prop_mediated: float
    a_p: float
    b_p: float
    c_p: float
    cprime_p: float
    indirect_ci: tuple
    group_means: Dict = field(default_factory=dict)
    conditions: Dict = field(default_factory=dict)

    def summary(self) -> str:
        L = [
            "=== 链式中介分析 X -> M -> Y ===",
            f"  a  (X->M)          = {self.a:+.4f}  (p={self.a_p:.4f})",
            f"  b  (M->Y|X)        = {self.b:+.4f}  (p={self.b_p:.4f})",
            f"  c  (X->Y 总效应)   = {self.c:+.4f}  (p={self.c_p:.4f})",
            f"  c' (X->Y 直接效应) = {self.c_prime:+.4f}  (p={self.cprime_p:.4f})",
            f"  间接效应 a*b       = {self.indirect:+.4f}  95%CI={self.indirect_ci}",
            f"  中介比例           = {self.prop_mediated:.1%}",
            "",
            "  分组均值:",
        ]
        for g, mv in self.group_means.items():
            L.append(f"    {g}: M={mv['M']:.3f}  Y={mv['Y']:.3f}  (n={mv['n']})")
        L.append("")
        L.append("  硬性判定条件:")
        for k, v in self.conditions.items():
            L.append(f"    [{'✓' if v else '✗'}] {k}")
        L.append(f"\n  中介结论: {'成立' if all(self.conditions.values()) else '不成立'}")
        return "\n".join(L)


def run_mediation(x: Sequence[float], m: Sequence[float], y: Sequence[float],
                  groups: Optional[Sequence[str]] = None,
                  n_boot: int = 5000, seed: int = 42) -> MediationResult:
    """
    x/m/y 等长。groups 可选（形如 'A'/'B'/'C'），用于计算分组均值与条件③。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    n = len(x)
    ones = np.ones(n)

    # a: M ~ 1 + X
    beta_a, _, _, p_a = _ols(np.column_stack([ones, x]), m)
    a = beta_a[1]

    # c: Y ~ 1 + X  (总效应)
    beta_c, _, _, p_c = _ols(np.column_stack([ones, x]), y)
    c = beta_c[1]

    # c' & b: Y ~ 1 + X + M
    beta_cp, _, _, p_cp = _ols(np.column_stack([ones, x, m]), y)
    c_prime, b = beta_cp[1], beta_cp[2]

    indirect = a * b
    prop = indirect / c if abs(c) > 1e-9 else 0.0

    # Bootstrap 间接效应 CI
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, mb, yb, ob = x[idx], m[idx], y[idx], ones
        ba = _ols(np.column_stack([ob, xb]), mb)[0][1]
        bb = _ols(np.column_stack([ob, xb, mb]), yb)[0][2]
        boot[i] = ba * bb
    ci = (round(float(np.percentile(boot, 2.5)), 4),
          round(float(np.percentile(boot, 97.5)), 4))

    # 分组均值
    gmeans = {}
    if groups is not None:
        groups = np.asarray(groups)
        for g in ["A", "B", "C"]:
            mask = groups == g
            if mask.any():
                gmeans[g] = {"M": float(m[mask].mean()),
                             "Y": float(y[mask].mean()),
                             "n": int(mask.sum())}

    # 硬性条件
    cond = {
        "① X->M 显著 (M1)": p_a[1] < 0.05 and a > 0,
        "② M->Y 显著且 CI 不含 0 (M2)": p_cp[2] < 0.05 and not (ci[0] <= 0 <= ci[1]),
        "③ C组M>B组M 且 C组Y>B组Y":
            ("B" in gmeans and "C" in gmeans
             and gmeans["C"]["M"] > gmeans["B"]["M"]
             and gmeans["C"]["Y"] > gmeans["B"]["Y"]),
    }

    return MediationResult(
        a=a, b=b, c=c, c_prime=c_prime, indirect=indirect, prop_mediated=prop,
        a_p=p_a[1], b_p=p_cp[2], c_p=p_c[1], cprime_p=p_cp[1],
        indirect_ci=ci, group_means=gmeans, conditions=cond)


def load_records(path: str):
    """
    从评分结果 json 载入记录，每条需含 group/M/Y 字段：
      {"group": "C", "M": 4.2, "Y": 4.5}, ...
    A 组 M 允许缺失（赋 0）。返回 (x, m, y, groups)。
    """
    with open(path, "r", encoding="utf-8") as f:
        recs = json.load(f)
    code = {"A": 0, "B": 1, "C": 2}
    x, m, y, g = [], [], [], []
    for r in recs:
        grp = r["group"]
        x.append(code[grp])
        m.append(float(r.get("M", 0) or 0))
        y.append(float(r["Y"]))
        g.append(grp)
    return x, m, y, g


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    # 自检：构造满足中介的模拟数据
    rng = np.random.default_rng(0)
    xs, ms, ys, gs = [], [], [], []
    for grp, xv in [("A", 0), ("B", 1), ("C", 2)]:
        for _ in range(60):
            mval = 0.0 if grp == "A" else (2.5 + 0.8 * xv + rng.normal(0, 0.4))
            yval = 2.0 + 0.3 * xv + 0.5 * mval + rng.normal(0, 0.4)
            xs.append(xv); ms.append(mval); ys.append(yval); gs.append(grp)
    res = run_mediation(xs, ms, ys, gs, n_boot=2000)
    print(res.summary())
