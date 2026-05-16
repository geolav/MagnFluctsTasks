import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

# ==========================================
# Папка для сохранения графиков
# ==========================================
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(filename):
    """
    Сохранение графика в папку plots
    """
    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()


# ==========================================================
# ЗАДАЧА 1
# ДВИЖЕНИЕ ЭЛЕКТРОНА В МАГНЕТРОНЕ
# ==========================================================
def task1_magnetron():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 1. ДВИЖЕНИЕ ЭЛЕКТРОНА В МАГНЕТРОНЕ")
    print("=" * 60)

    # параметры магнетрона
    Ra = 0.045       # радиус анода (м)
    Rk = 0.005       # радиус катода (м)
    U = 150          # напряжение (В)
    n = 1000         # витков на метр

    # константы
    e = 1.6e-19
    m = 9.1e-31
    mu0 = 4 * np.pi * 1e-7

    # скорость электрона
    v = np.sqrt(2 * e * U / m)

    # критическое поле
    r_target = (Ra + Rk) / 2
    B = m * v / (e * r_target)

    # критический ток
    I = B / (mu0 * n)

    print("\nИсходные параметры:")
    print(f"Радиус анода: {Ra} м")
    print(f"Радиус катода: {Rk} м")
    print(f"Напряжение: {U} В")

    print("\nРезультаты расчета:")
    print(f"Скорость электрона: {v:.3e} м/с")
    print(f"Магнитная индукция: {B:.3e} Тл")
    print(f"Критический ток: {I:.3f} А")

    # система уравнений движения
    def system(t, state):
        x, y, vx, vy = state

        r = np.sqrt(x**2 + y**2)

        if r < Rk:
            r = Rk

        # радиальное электрическое поле
        E = U / (r * np.log(Ra / Rk))

        Ex = E * x / r
        Ey = E * y / r

        ax = -(e / m) * (Ex + vy * B)
        ay = -(e / m) * (Ey - vx * B)

        return [vx, vy, ax, ay]

    y0 = [Rk, 0, 0, v]

    sol = solve_ivp(
        system,
        [0, 2e-7],
        y0,
        max_step=1e-11
    )

    plt.figure(figsize=(6, 6))
    plt.plot(sol.y[0], sol.y[1], label="Траектория электрона")

    plt.gca().add_patch(
        plt.Circle((0, 0), Ra, fill=False, color='red', label='Анод')
    )

    plt.gca().add_patch(
        plt.Circle((0, 0), Rk, color='gray', label='Катод')
    )

    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.title("Траектория электрона в магнетроне")

    save_plot("task1_magnetron.png")


# ==========================================================
# ЗАДАЧА 2
# ПАРАМЕТРЫ КАТУШКИ
# ==========================================================
def task2_coil():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 2. РАСЧЕТ ПАРАМЕТРОВ КАТУШКИ")
    print("=" * 60)

    mu0 = 4 * np.pi * 1e-7

    # параметры катушки
    L_wire = 10      # длина провода (м)
    D = 0.05         # диаметр катушки (м)
    l = 0.1          # длина катушки (м)
    I = 1.0          # ток (А)

    # число витков
    N = int(L_wire / (np.pi * D))

    # площадь сечения
    A = np.pi * (D / 2)**2

    # индуктивность
    L_ind = mu0 * N**2 * A / l

    print("\nИсходные параметры:")
    print(f"Длина провода: {L_wire} м")
    print(f"Диаметр катушки: {D} м")
    print(f"Длина катушки: {l} м")
    print(f"Ток: {I} А")

    print("\nРезультаты:")
    print(f"Число витков: {N}")
    print(f"Площадь сечения: {A:.6e} м²")
    print(f"Индуктивность: {L_ind:.6e} Гн")

    # -----------------------------------------
    # График распределения поля вдоль оси
    # -----------------------------------------
    z = np.linspace(-0.2, 0.2, 500)

    Bz = (mu0 * N * I / (2 * l)) * (
        (z + l/2)/np.sqrt((z + l/2)**2 + (D/2)**2)
        -
        (z - l/2)/np.sqrt((z - l/2)**2 + (D/2)**2)
    )

    print(f"\nМаксимальное поле в центре катушки: {np.max(Bz):.6e} Тл")

    plt.figure(figsize=(8,5))
    plt.plot(z, Bz)
    plt.grid()

    plt.xlabel("z (м)")
    plt.ylabel("B(z) (Тл)")
    plt.title("Распределение магнитного поля вдоль оси катушки")

    save_plot("task2_coil.png")

# ==========================================================
# ЗАДАЧА 3
# ПОЛЕ СОЛЕНОИДА
# ==========================================================
def task3_solenoid():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 3. МАГНИТНОЕ ПОЛЕ СОЛЕНОИДА")
    print("=" * 60)

    mu0 = 4 * np.pi * 1e-7

    R = 0.05
    L = 0.4
    N = 200
    I = 1

    print("\nПараметры соленоида:")
    print(f"Радиус: {R} м")
    print(f"Длина: {L} м")
    print(f"Количество витков: {N}")
    print(f"Ток: {I} А")

    z_turns = np.linspace(-L/2, L/2, N)

    x = np.linspace(-0.15, 0.15, 60)
    z = np.linspace(-0.3, 0.3, 60)

    X, Z = np.meshgrid(x, z)

    Bx = np.zeros_like(X)
    Bz = np.zeros_like(X)

    print("\nВычисляется суммарное поле от всех витков...")

    for z0 in z_turns:
        rz = Z - z0
        r = np.sqrt(X**2 + rz**2 + R**2)

        Bz += mu0 * I * R**2 / (2 * r**3)
        Bx += mu0 * I * X * rz / (2 * r**5)

    plt.figure(figsize=(8, 6))
    plt.streamplot(
        X,
        Z,
        Bx,
        Bz,
        color=np.sqrt(Bx**2 + Bz**2)
    )

    plt.title("Силовые линии магнитного поля соленоида")
    plt.grid()

    save_plot("task3_solenoid.png")


# ==========================================================
# ЗАДАЧА 4
# СВЯЗАННЫЕ МАЯТНИКИ
# ==========================================================
def task4_pendulums():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 4. СВЯЗАННЫЕ МАЯТНИКИ")
    print("=" * 60)

    g = 9.81
    L = 1
    k = 2
    m = 0.5

    def equations(t, y):
        phi1, phi2, w1, w2 = y

        a1 = -(g / L) * phi1 + k * (phi2 - phi1) / m
        a2 = -(g / L) * phi2 + k * (phi1 - phi2) / m

        return [w1, w2, a1, a2]

    t = np.linspace(0, 40, 5000)

    sol = solve_ivp(
        equations,
        [0, 40],
        [0.3, -0.2, 0, 0],
        t_eval=t
    )

    phi = sol.y[0]

    dt = t[1] - t[0]

    window = np.hanning(len(phi))

    fft_vals = np.fft.rfft(phi * window)
    freqs = np.fft.rfftfreq(len(phi), dt)

    amplitude = np.abs(fft_vals)

    mask = freqs < 3

    peaks, _ = find_peaks(
        amplitude[mask],
        height=np.max(amplitude) * 0.1
    )

    found_freqs = freqs[mask][peaks]

    print("\nНормальные частоты системы:")
    for i, freq in enumerate(found_freqs):
        print(f"Частота {i+1}: {freq:.4f} Гц")

    plt.figure(figsize=(8, 5))
    plt.plot(freqs[mask], amplitude[mask])
    plt.scatter(
        freqs[mask][peaks],
        amplitude[mask][peaks],
        color='red'
    )

    plt.title("Спектр связанных маятников")
    plt.grid()

    save_plot("task4_pendulums.png")


# ==========================================================
# ЗАДАЧА 5
# ФУРЬЕ-ФИЛЬТРАЦИЯ СИГНАЛА
# ==========================================================
def task5_fourier_filter():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 5. ФИЛЬТРАЦИЯ СИГНАЛА")
    print("=" * 60)

    data = np.load("signal_1.npy")
    data = data.flatten()

    dt = 1e-4
    N = len(data)

    print(f"\nКоличество точек сигнала: {N}")
    print(f"Шаг дискретизации: {dt}")

    window = np.hanning(N)

    fft_vals = np.fft.rfft(data * window)
    freqs = np.fft.rfftfreq(N, dt)

    amplitude = np.abs(fft_vals)

    threshold = np.mean(amplitude) + 5*np.std(amplitude)

    peaks, props = find_peaks(
        amplitude,
        height=threshold,
        distance=20
    )

    top = np.argsort(props["peak_heights"])[-10:]
    peaks = peaks[top]

    print("\nНайдены значимые частоты:")
    for i, p in enumerate(peaks):
        print(f"Гармоника {i+1}: {freqs[p]:.2f} Гц")

    filtered_fft = np.zeros_like(fft_vals)

    bandwidth = 5

    for p in peaks:
        filtered_fft[p-bandwidth:p+bandwidth] = \
            fft_vals[p-bandwidth:p+bandwidth]

    filtered_signal = np.fft.irfft(filtered_fft, n=N)

    t = np.arange(N) * dt

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t[:5000], data[:5000])
    plt.title("Исходный сигнал")

    plt.subplot(3, 1, 2)
    plt.plot(freqs, amplitude)
    plt.scatter(freqs[peaks], amplitude[peaks], color='red')
    plt.xlim(0, 5000)
    plt.title("Спектр сигнала")

    plt.subplot(3, 1, 3)
    plt.plot(t[:5000], filtered_signal[:5000])
    plt.title("Отфильтрованный сигнал")

    plt.tight_layout()

    save_plot("task5_fourier.png")

    print("\nШум успешно удален.")
    print("Восстановленный сигнал построен.")


# ==========================================================
# ЗАПУСК ВСЕХ ЗАДАЧ
# ==========================================================
if __name__ == "__main__":
    task1_magnetron()
    task2_coil()
    task3_solenoid()
    task4_pendulums()
    task5_fourier_filter()

    print("\n" + "=" * 60)
    print("ВСЕ ЗАДАЧИ УСПЕШНО ВЫПОЛНЕНЫ")
    print("Графики сохранены в папке plots")
    print("=" * 60)