import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate
import sympy

config = {"font.family": "serif",
          "font.size": 12,
          "mathtext.fontset": "stix",
          "font.serif": ["SimSun"]}
rcParams.update(config)

# 参数解释
m_v = 1000e-3  # 车轴质量[kg]
M_v = 2000e-3  # 车头部质量[kg]
L1_v = 150e-3  # 车头部高度[m]
L2_v = 350e-3  # 车轴部高度[m]
I_v = M_v * pow((L1_v / 2 + L2_v), 2) + m_v * pow(L2_v, 2) / 12  # 车身转动惯量[kg*m^2]
L_v = L2_v / 2 + ((L1_v + L2_v) * M_v / (2 * (m_v + M_v)))  # 重心位置[m]
mu_v = 0.012  # 摩擦系数[1]
tau_v = 0  # 驱动力[N]
g_v = 9.8  # 重力加速度[m/s**2]

# 数值解存储列表
lit_theta = []
lit_x = []

tt = np.linspace(0, 2.5, 10000)


def function(inv):
    # 因变量设置
    t, g, m, M, I, L, mu, tau = sympy.symbols("t, g, m, M, I, L, mu, tau")

    # 自变量设置
    theta, x = sympy.symbols("theta, x", cls=sympy.Function)

    # 微分方程
    ode1 = sympy.Eq((I + m * L ** 2) * theta(t).diff(t, t) - m * g * L * theta(t) - m * L * x(t).diff(t, t), 0)
    # print(ode1)

    ode2 = sympy.Eq((M + m) * x(t).diff(t, t) + mu * x(t).diff(t) - m * L * theta(t).diff(t, t) - tau, 0)
    # print(ode2)

    # 微分方程降阶
    y1, y2, y3, y4 = sympy.symbols("y1, y2, y3, y4", cls=sympy.Function)

    varchange = {theta(t).diff(t, t): y2(t).diff(t),
                 theta(t): y1(t),
                 x(t).diff(t, t): y4(t).diff(t),
                 x(t): y3(t)}

    ode1_vc = ode1.subs(varchange)
    ode2_vc = ode2.subs(varchange)
    ode3 = y1(t).diff(t) - y2(t)
    ode4 = y3(t).diff(t) - y4(t)

    y = sympy.Matrix([y1(t), y2(t), y3(t), y4(t)])

    vcsol = sympy.solve((ode1_vc, ode2_vc, ode3, ode4), y.diff(t), dict=True)

    f = y.diff(t).subs(vcsol[0])

    # 已知参数初始化
    params = {m: m_v,
              M: M_v,
              I: I_v,
              L: L_v,
              mu: mu_v,
              tau: tau_v,
              g: g_v}

    _f_np = sympy.lambdify((t, y), f.subs(params), "numpy")
    f_np = lambda _t, _y, *args: _f_np(_t, _y)

    jac = sympy.Matrix([[fj.diff(yi) for yi in y] for fj in f])

    _jac_np = sympy.lambdify((t, y), jac.subs(params), "numpy")
    jac_np = lambda _t, _y, *args: _jac_np(_t, _y)

    y0 = [inv, 0, 0, 0]

    r = integrate.ode(f_np, jac_np).set_initial_value(y0, tt[0])
    dt = tt[1] - tt[0]
    yy = np.zeros((len(tt), len(y0)))

    idx = 0
    while r.successful() and r.t < tt[-1]:
        yy[idx, :] = r.y
        r.integrate(r.t + dt)
        idx += 1

    lit_theta.append(yy[:, 0])
    lit_x.append(yy[:, 2])

    return 0


def draw():
    lit_label = [0, 2.5, 5, 7.5, 10]
    lit_color = ["red", "orange", "yellow", "green", "blue"]

    plt.figure(figsize=(8, 8 / 0.618))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for i in range(5):
        ax1.plot(tt, lit_theta[i], lit_color[i], label="$theta$ = {}$°$".format(lit_label[i]))
    ax1.set_ylabel("车身偏移角度 $theta(rad)$")
    ax1.grid()
    ax1.legend()

    for j in range(5):
        ax2.plot(tt, lit_x[j], lit_color[j], label="$theta$ = {}$°$".format(lit_label[j]))
    ax2.set_xlabel("时间($s$)")
    ax2.set_ylabel("小车位移 $x(m)$")
    ax2.grid()
    ax2.legend()

    plt.savefig("result.jpg")
    plt.show()

    return 0


def save():
    file_1 = open("theta.txt", 'w', encoding="utf-8")
    file_2 = open("x.txt", 'w', encoding="utf-8")

    for i in range(len(lit_theta)):
        file_1.write(str(lit_theta[i]))
        file_1.write('\n')
    file_1.close()

    for j in range(len(lit_x)):
        file_2.write(str(lit_x[j]))
        file_2.write('\n')
    file_2.close()

    return 0


def main():
    lit_inv = [0, 2.5 * 3.14 / 180, 5 * 3.14 / 180, 7.5 * 3.14 / 180, 10 * 3.14 / 180]
    for i in range(len(lit_inv)):
        function(lit_inv[i])
    draw()
    save()

    return 0


if __name__ == "__main__":
    main()
