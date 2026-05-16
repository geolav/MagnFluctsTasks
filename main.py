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
    """Сохранение графика в папку plots"""
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
    Ra = 0.045   # радиус анода (м)
    Rk = 0.005   # радиус катода (м)
    U  = 150     # напряжение (В)
    n  = 1000    # витков на метр

    # константы
    e   = 1.6e-19
    m   = 9.1e-31
    mu0 = 4 * np.pi * 1e-7

    # скорость электрона
    v = np.sqrt(2 * e * U / m)

    # критическое поле
    r_target = (Ra + Rk) / 2
    B = m * v / (e * r_target)

    # критический ток
    I = B / (mu0 * n)

    print("\nИсходные параметры:")
    print(f"Радиус анода:  {Ra} м")
    print(f"Радиус катода: {Rk} м")
    print(f"Напряжение:    {U} В")
    print("\nРезультаты расчета:")
    print(f"Скорость электрона: {v:.3e} м/с")
    print(f"Магнитная индукция: {B:.3e} Тл")
    print(f"Критический ток:    {I:.3f} А")

    # система уравнений движения
    def system(t, state):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        if r < Rk:
            r = Rk

        # радиальное электрическое поле
        E  = U / (r * np.log(Ra / Rk))
        Ex = E * x / r
        Ey = E * y / r

        ax = -(e / m) * (Ex + vy * B)
        ay = -(e / m) * (Ey - vx * B)
        return [vx, vy, ax, ay]

    y0  = [Rk, 0, 0, v]
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
    L_wire = 10    # длина провода (м)
    D      = 0.05  # диаметр катушки (м)
    l      = 0.1   # длина катушки (м)
    I      = 1.0   # ток (А)

    # число витков
    N = int(L_wire / (np.pi * D))

    # площадь сечения
    A = np.pi * (D / 2)**2

    # индуктивность
    L_ind = mu0 * N**2 * A / l

    print("\nИсходные параметры:")
    print(f"Длина провода:    {L_wire} м")
    print(f"Диаметр катушки:  {D} м")
    print(f"Длина катушки:    {l} м")
    print(f"Ток:              {I} А")
    print("\nРезультаты:")
    print(f"Число витков:     {N}")
    print(f"Площадь сечения:  {A:.6e} м²")
    print(f"Индуктивность:    {L_ind:.6e} Гн")

    # -----------------------------------------
    # График распределения поля вдоль оси
    # -----------------------------------------
    z  = np.linspace(-0.2, 0.2, 500)
    Bz = (mu0 * N * I / (2 * l)) * (
        (z + l/2) / np.sqrt((z + l/2)**2 + (D/2)**2)
        -
        (z - l/2) / np.sqrt((z - l/2)**2 + (D/2)**2)
    )

    print(f"\nМаксимальное поле в центре катушки: {np.max(Bz):.6e} Тл")

    plt.figure(figsize=(8, 5))
    plt.plot(z, Bz)
    plt.grid()
    plt.xlabel("z (м)")
    plt.ylabel("B(z) (Тл)")
    plt.title("Распределение магнитного поля вдоль оси катушки")
    save_plot("task2_coil.png")


# ==========================================================
# ЗАДАЧА 3
# ПОЛЕ СОЛЕНОИДА  (ИСПРАВЛЕНО)
# ==========================================================

def task3_solenoid():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 3. МАГНИТНОЕ ПОЛЕ СОЛЕНОИДА")
    print("=" * 60)

    mu0 = 4 * np.pi * 1e-7
    R   = 0.05
    L   = 0.4
    N   = 200
    I   = 1

    print("\nПараметры соленоида:")
    print(f"Радиус:            {R} м")
    print(f"Длина:             {L} м")
    print(f"Количество витков: {N}")
    print(f"Ток:               {I} А")

    z_turns = np.linspace(-L/2, L/2, N)

    x = np.linspace(-0.15, 0.15, 60)
    z = np.linspace(-0.3,  0.3,  60)
    X, Z = np.meshgrid(x, z)

    Bx = np.zeros_like(X)
    Bz = np.zeros_like(X)

    print("\nВычисляется суммарное поле от всех витков...")

    # ИСПРАВЛЕНИЕ: правильные формулы для поля кругового витка
    # Bz = mu0*I*R^2 / (2*(R^2 + rz^2 + x^2)^(3/2))  — аппроксимация на оси
    # Bx — радиальная компонента через закон Био-Савара (дипольное приближение)
    for z0 in z_turns:
        rz    = Z - z0
        denom = (X**2 + rz**2 + R**2)

        # осевая компонента
        Bz += mu0 * I * R**2 / (2 * denom**1.5)

        # радиальная компонента (исправленный коэффициент)
        Bx += (3 * mu0 * I * R**2 / 4) * X * rz / denom**2.5

    plt.figure(figsize=(8, 6))
    plt.streamplot(
        X, Z, Bx, Bz,
        color=np.sqrt(Bx**2 + Bz**2),
        linewidth=1.2,
        cmap='plasma'
    )
    # рисуем границы соленоида
    plt.axvline( R, color='gray', linestyle='--', linewidth=1, label='Стенка соленоида')
    plt.axvline(-R, color='gray', linestyle='--', linewidth=1)
    plt.axhline( L/2,  color='gray', linestyle=':',  linewidth=1)
    plt.axhline(-L/2,  color='gray', linestyle=':',  linewidth=1)
    plt.legend(fontsize=8)
    plt.title("Силовые линии магнитного поля соленоида")
    plt.xlabel("x (м)")
    plt.ylabel("z (м)")
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

    t   = np.linspace(0, 40, 5000)
    sol = solve_ivp(
        equations,
        [0, 40],
        [0.3, -0.2, 0, 0],
        t_eval=t
    )

    phi    = sol.y[0]
    dt     = t[1] - t[0]
    window = np.hanning(len(phi))

    fft_vals  = np.fft.rfft(phi * window)
    freqs     = np.fft.rfftfreq(len(phi), dt)
    amplitude = np.abs(fft_vals)

    mask       = freqs < 3
    peaks, _   = find_peaks(
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
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.grid()
    save_plot("task4_pendulums.png")


# ==========================================================
# ЗАДАЧА 5
# ФУРЬЕ-ФИЛЬТРАЦИЯ СИГНАЛА  (ИСПРАВЛЕНО — метод порога PSD)
# ==========================================================

def task5_fourier_filter():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 5. ФИЛЬТРАЦИЯ СИГНАЛА")
    print("=" * 60)

    data = np.load("signal_1.npy")
    data = data.flatten()

    dt = 1e-4
    N  = len(data)

    print(f"\nКоличество точек сигнала: {N}")
    print(f"Шаг дискретизации: {dt}")

    t     = np.arange(N) * dt
    freqs = np.fft.rfftfreq(N, dt)

    # --- ШАГ 1: FFT и спектр мощности (PSD) ---
    fft_raw = np.fft.fft(data, N)
    psd     = np.abs(fft_raw)**2 / N          # спектр мощности
    freq_full = (1 / (dt * N)) * np.arange(N) # частоты для полного FFT

    # --- ШАГ 2: фильтрация по порогу мощности ---
    # Оставляем только топ-1% самых мощных частот — это и есть гармоники сигнала
    threshold    = np.percentile(psd, 99)
    mask         = psd > threshold             # True там где сигнал, False где шум
    fft_filtered = mask * fft_raw              # обнуляем шумовые частоты
    filtered_signal = np.real(np.fft.ifft(fft_filtered))  # обратное преобразование

    # --- Поиск частот пиков для вывода ---
    half = N // 2
    psd_half  = psd[:half]
    freq_half = freq_full[:half]
    peaks, props = find_peaks(psd_half, height=threshold, distance=20)
    top   = np.argsort(props["peak_heights"])[-10:]
    peaks = peaks[top]

    print("\nНайдены значимые частоты:")
    for i, p in enumerate(peaks):
        print(f"Гармоника {i+1}: {freq_half[p]:.2f} Гц")

    print(f"\nПорог PSD: {threshold:.2f}")
    print(f"Амплитуда исходного сигнала:    {np.std(data):.3f}")
    print(f"Амплитуда отфильтрованного:     {np.std(filtered_signal):.3f}")

    # --- Визуализация ---
    plt.figure(figsize=(10, 9))

    # График 1: исходный сигнал + отфильтрованный поверх (как на эталоне)
    plt.subplot(3, 1, 1)
    plt.plot(t[:5000], data[:5000],
             color='steelblue', lw=0.8, alpha=0.7, label='Исходный (зашумлённый)')
    plt.plot(t[:5000], filtered_signal[:5000],
             color='red', lw=1.5, label='Отфильтрованный')
    plt.title("Исходный сигнал и отфильтрованный")
    plt.xlabel("t (с)")
    plt.legend(fontsize=9)

    # График 2: спектр мощности PSD с порогом
    plt.subplot(3, 1, 2)
    plt.plot(freq_half, psd_half, color='steelblue', lw=0.8, label='PSD (зашумлённый)')
    plt.axhline(threshold, color='red', lw=1.2,
                linestyle='--', label=f'Порог = {threshold:.0f}')
    plt.scatter(freq_half[peaks], psd_half[peaks], color='red', zorder=5)
    plt.xlim(0, 5000)
    plt.yscale('log')
    plt.title("Спектр мощности (PSD) с порогом фильтрации")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Мощность")
    plt.legend(fontsize=9)

    # График 3: только отфильтрованный сигнал
    plt.subplot(3, 1, 3)
    plt.plot(t[:5000], filtered_signal[:5000], color='red', lw=1.5)
    plt.title("Отфильтрованный сигнал (чистые гармоники)")
    plt.xlabel("t (с)")

    plt.tight_layout()
    save_plot("task5_fourier.png")

    print("\nФильтрация завершена.")


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