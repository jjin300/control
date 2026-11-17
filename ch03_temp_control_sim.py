"""
온오프 / P / PID 온도 제어 비교 시뮬레이션
공정 모델 (FOPDT): 300·dT/dt = -(T - 25) + 16·u(t - 30)
  - T: 온도 [°C], u: 히터 출력 [0~100 %]
  - 시정수 300 s, 정상상태 이득 16 °C/%, 무효시간 30 s, 주위온도 25 °C
오일러법, dt = 1 s, 총 5400 s, 설정값 1000 °C
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ---------- 한글 폰트 ----------
for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK TC"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        rcParams["font.family"] = f
        break
rcParams["axes.unicode_minus"] = False

# ---------- 공정·시뮬레이션 파라미터 ----------
TAU, K, T_AMB, DEAD = 300.0, 16.0, 25.0, 30   # 시정수, 이득, 주위온도, 무효시간[s]
DT, T_END, SP = 1.0, 5400, 1000.0
N = int(T_END / DT)
DELAY = int(DEAD / DT)                          # 지연 샘플 수 = 30


def clamp(u):
    return min(max(u, 0.0), 100.0)


def simulate(controller):
    """controller(k, T_now) -> u[k]. 반환: t, T, u 배열"""
    t = np.arange(N + 1) * DT
    T = np.zeros(N + 1)
    u = np.zeros(N + 1)
    T[0] = T_AMB
    for k in range(N):
        u[k] = controller(k, T[k])
        u_delayed = u[k - DELAY] if k >= DELAY else 0.0   # u(t-30), t<30 이전은 0
        dTdt = (-(T[k] - T_AMB) + K * u_delayed) / TAU
        T[k + 1] = T[k] + DT * dTdt                      # 오일러법
    u[N] = u[N - 1]
    return t, T, u


# ---------- ① 온오프 (히스테리시스 ±5 °C) ----------
def make_onoff(low=995.0, high=1005.0):
    state = {"u": 100.0}            # 시작은 저온이므로 ON
    def ctrl(k, T):
        if T < low:
            state["u"] = 100.0
        elif T > high:
            state["u"] = 0.0
        # 995~1005 사이: 직전 상태 유지
        return state["u"]
    return ctrl


# ---------- ② P 제어 ----------
def make_p(Kc=0.3):
    def ctrl(k, T):
        return clamp(Kc * (SP - T))
    return ctrl


# ---------- ③ PID (출력 포화 시 적분 누적 중지) ----------
def make_pid(Kc=0.3, tauI=200.0, tauD=25.0):
    st = {"I": 0.0, "e_prev": None}
    def ctrl(k, T):
        e = SP - T
        de = 0.0 if st["e_prev"] is None else (e - st["e_prev"]) / DT
        st["e_prev"] = e
        I_new = st["I"] + e * DT                              # 적분 후보값
        u_raw = Kc * (e + I_new / tauI + tauD * de)
        if 0.0 <= u_raw <= 100.0:                             # 포화가 아닐 때만 적분 누적
            st["I"] = I_new
        else:                                                 # 포화 → 적분 동결
            u_raw = Kc * (e + st["I"] / tauI + tauD * de)
        return clamp(u_raw)
    return ctrl


t, T_on, u_on = simulate(make_onoff())
_, T_p, u_p = simulate(make_p())
_, T_pid, u_pid = simulate(make_pid())
t_min = t / 60.0

# ---------- 그래프 ----------
fig, axes = plt.subplots(3, 1, figsize=(11, 13))

# 1) 온오프 온도 전체 곡선
ax = axes[0]
ax.plot(t_min, T_on, color="tab:red", lw=1.2, label="온오프 온도")
ax.axhline(SP, color="k", ls="--", lw=1, label="설정값 1000 °C")
ax.axhspan(995, 1005, color="gray", alpha=0.2, label="히스테리시스 995~1005 °C")
ax.set(title="① 온오프 제어 – 온도 전체 곡선 (0~90분)",
       xlabel="시간 [분]", ylabel="온도 [°C]", xlim=(0, 90))
ax.grid(alpha=0.3); ax.legend(loc="lower right")

# 2) 40~70분 확대: 온도 + 히터 출력(계단)
ax = axes[1]
m = (t_min >= 40) & (t_min <= 70)
ax.plot(t_min[m], T_on[m], color="tab:red", lw=1.5, label="온도")
ax.axhline(SP, color="k", ls="--", lw=1)
ax.axhline(995, color="gray", ls=":", lw=1)
ax.axhline(1005, color="gray", ls=":", lw=1)
ax.set(title="② 온오프 제어 – 40~70분 확대 (온도 & 히터 출력)",
       xlabel="시간 [분]", ylabel="온도 [°C]", xlim=(40, 70))
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.step(t_min[m], u_on[m], where="post", color="tab:blue", lw=1.2, label="히터 출력 u")
ax2.set_ylabel("히터 출력 [%]", color="tab:blue")
ax2.set_ylim(-5, 160)
ax2.tick_params(axis="y", colors="tab:blue")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")

# 3) 세 제어기 온도 비교
ax = axes[2]
ax.plot(t_min, T_on, color="tab:red", lw=1.0, label="온오프")
ax.plot(t_min, T_p, color="tab:green", lw=1.8, label="P (Kc=0.3)")
ax.plot(t_min, T_pid, color="tab:blue", lw=1.8, label="PID (Kc=0.3, τI=200 s, τD=25 s)")
ax.axhline(SP, color="k", ls="--", lw=1, label="설정값")
ax.set(title="③ 세 제어기 온도 비교", xlabel="시간 [분]", ylabel="온도 [°C]",
       xlim=(0, 90), ylim=(750, 1100))
ax.grid(alpha=0.3); ax.legend(loc="lower right", ncol=2, framealpha=0.95)

fig.tight_layout()
fig.savefig("temp_control_comparison.png", dpi=150)

# ---------- 성능 지표 ----------
last30 = t >= T_END - 1800
pp = T_on[last30].max() - T_on[last30].min()
offset_p = SP - T_p[-1]
offset_p_theory = SP - (T_AMB + K * 0.3 * SP) / (1 + K * 0.3)
err_pid = SP - T_pid[-1]

print("=" * 52)
print(f"온오프  진동폭(마지막 30분)   : {T_on[last30].min():.2f} ~ {T_on[last30].max():.2f} °C")
print(f"                              → 피크-피크 {pp:.2f} °C (±{pp/2:.2f} °C)")
print(f"P 제어  잔류편차 (t=5400 s)   : {offset_p:.2f} °C  (최종 온도 {T_p[-1]:.2f} °C)")
print(f"        이론 정상상태 편차     : {offset_p_theory:.2f} °C")
print(f"PID     최종 오차 (t=5400 s)  : {err_pid:.4f} °C  (최종 온도 {T_pid[-1]:.4f} °C)")
print("=" * 52)
