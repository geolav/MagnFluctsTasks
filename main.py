import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os


# --------------------------------------------------------------
# ЗАДАЧА 1: МАГНЕТРОН
# --------------------------------------------------------------
def task1_magnetron():
    print("\n=== ЗАДАЧА 1: Магнетрон ===")
    # Параметры (можно менять)
    D = 0.1  # диаметр соленоида, м
    n = 1000  # число витков на метр
    Ra = 0.045  # радиус анода, м
    Rk = 0.005  # радиус катода, м
    U = 150.0  # напряжение, В

    mu0 = 4 * np.pi * 1e-7
    e = 1.6e-19
    m = 9.1e-31

    # Условие: диаметр окружности электрона = Ra - Rk
    radius_electron = (Ra - Rk) / 2.0
    v = np.sqrt(2 * e * U / m)
    B_needed = m * v / (e * radius_electron)
    Ic = B_needed / (mu0 * n)

    print(f"Необходимый ток соленоида Ic = {Ic:.3f} А")
    print(f"Индукция магнитного поля B = {B_needed:.6f} Тл")

    # Траектория электрона
    def lorentz(t, state, B):
        x, y, vx, vy = state
        ax = (e / m) * (vy * B)
        ay = -(e / m) * (vx * B)
        return [vx, vy, ax, ay]

    x0 = Rk
    y0 = 0.0
    v0 = np.sqrt(2 * e * U / m)
    vx0 = v0
    vy0 = 0.0

    sol = solve_ivp(lorentz, [0, 1e-6], [x0, y0, vx0, vy0],
                    args=(B_needed,), max_step=1e-10, method='RK45')

    # Построение траектории
    plt.figure(figsize=(6, 6))
    plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=1.5, label='Траектория электрона')
    anode_circle = plt.Circle((0, 0), Ra, color='red', fill=False, lw=2, label='Анод')
    cathode_circle = plt.Circle((0, 0), Rk, color='gray', fill=True, alpha=0.3, label='Катод')
    plt.gca().add_artist(anode_circle)
    plt.gca().add_artist(cathode_circle)
    plt.axis('equal')
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title(f'Траектория электрона, Ic = {Ic:.3f} А')
    plt.legend()
    plt.grid()
    plt.show()

    # Диаграмма Ic(U)
    U_range = np.linspace(50, 300, 50)
    Ic_range = []
    for U_val in U_range:
        v_val = np.sqrt(2 * e * U_val / m)
        B_val = m * v_val / (e * radius_electron)
        Ic_val = B_val / (mu0 * n)
        Ic_range.append(Ic_val)

    plt.figure(figsize=(8, 5))
    plt.plot(U_range, Ic_range, 'r-', linewidth=2)
    plt.fill_between(U_range, 0, Ic_range, alpha=0.3, label='Область окружности')
    plt.xlabel('Напряжение U, В')
    plt.ylabel('Ток соленоида Ic, А')
    plt.title('Диаграмма Ic(U) для окружности диаметром Ra-Rk')
    plt.grid()
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# ЗАДАЧА 2: ОПТИМИЗАЦИЯ КАТУШКИ
# --------------------------------------------------------------
def task2_coil():
    print("\n=== ЗАДАЧА 2: Оптимизация катушки ===")
    # Параметры
    L_wire = 10.0  # длина провода, м
    d_wire = 0.001  # диаметр провода, м
    D_coil = 0.05  # диаметр каркаса, м
    l_coil = 0.1  # длина каркаса, м
    mu0 = 4 * np.pi * 1e-7

    # Число витков
    N_possible = L_wire / (np.pi * D_coil)
    N = int(np.floor(N_possible))
    print(f"Максимальное число витков N = {N}")

    # Индукция в центре
    theta = np.arctan2(D_coil / 2, l_coil / 2)
    cos_theta = np.cos(theta)
    B_center_per_A = mu0 * N / (2 * l_coil) * (cos_theta + cos_theta)
    print(f"Индукция на 1 А тока в центре: {B_center_per_A:.6f} Тл/А")

    # Индуктивность катушки
    A = np.pi * (D_coil / 2) ** 2
    L_ind = mu0 * N ** 2 * A / l_coil
    print(f"Индуктивность катушки: {L_ind:.6f} Гн")

    # График B(l) вдоль оси
    z_axis = np.linspace(-l_coil, l_coil, 200)
    B_axis = []
    I_current = 1.0
    for z in z_axis:
        z1 = l_coil / 2 - z
        z2 = l_coil / 2 + z
        cos1 = z1 / np.sqrt(z1 ** 2 + (D_coil / 2) ** 2)
        cos2 = z2 / np.sqrt(z2 ** 2 + (D_coil / 2) ** 2)
        B = mu0 * N * I_current / (2 * l_coil) * (cos1 + cos2)
        B_axis.append(B)

    plt.figure(figsize=(8, 5))
    plt.plot(z_axis, B_axis, 'b-')
    plt.xlabel('Расстояние от центра, м')
    plt.ylabel('Магнитная индукция B, Тл')
    plt.title('Распределение B на оси катушки (I=1 А)')
    plt.grid()
    plt.show()


# --------------------------------------------------------------
# ЗАДАЧА 3: ВИЗУАЛИЗАЦИЯ ПОЛЯ СОЛЕНОИДА
# --------------------------------------------------------------
def task3_solenoid_field():
    print("\n=== ЗАДАЧА 3: Визуализация поля соленоида ===")
    # Параметры
    D_sol = 0.1  # диаметр соленоида, м
    L_sol = 0.4  # длина соленоида, м
    N_sol = 200  # число витков
    I_sol = 1.0  # ток, А
    mu0 = 4 * np.pi * 1e-7

    n_sol = N_sol / L_sol

    def B_z_on_axis(z):
        z1 = L_sol / 2 - z
        z2 = L_sol / 2 + z
        cos1 = z1 / np.sqrt(z1 ** 2 + (D_sol / 2) ** 2)
        cos2 = z2 / np.sqrt(z2 ** 2 + (D_sol / 2) ** 2)
        return 0.5 * mu0 * n_sol * I_sol * (cos1 + cos2)

    x = np.linspace(-D_sol, D_sol, 30)
    z = np.linspace(-L_sol, L_sol, 30)
    X, Z = np.meshgrid(x, z)
    Bx = np.zeros_like(X)
    Bz_2d = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(z)):
            r = abs(x[i])
            zz = z[j]
            Bz_center = B_z_on_axis(zz)
            factor = 1.0 - (r / (D_sol / 2)) ** 2
            if factor < 0:
                factor = 0
            Bz_2d[j, i] = Bz_center * factor
            Bx[j, i] = -0.5 * r * (Bz_center - B_z_on_axis(zz + 0.01)) / 0.01

    plt.figure(figsize=(8, 6))
    strm = plt.streamplot(X, Z, Bx, Bz_2d, color=np.sqrt(Bx ** 2 + Bz_2d ** 2),
                          cmap='viridis', linewidth=1, density=1.2)
    plt.colorbar(strm.lines, label='|B|, Тл')
    plt.xlabel('x (радиус), м')
    plt.ylabel('z (ось), м')
    plt.title(f'Магнитное поле соленоида, D={D_sol} м, L={L_sol} м, N={N_sol}')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlim(-D_sol, D_sol)
    plt.ylim(-L_sol, L_sol)
    plt.gca().set_aspect('equal')
    plt.show()


# --------------------------------------------------------------
# ЗАДАЧА 4: СВЯЗАННЫЕ МАЯТНИКИ С ЗАТУХАНИЕМ
# --------------------------------------------------------------
def task4_coupled_pendulums():
    print("\n=== ЗАДАЧА 4: Связанные маятники ===")
    # Параметры
    L = 1.0  # длина маятника, м
    m = 0.5  # масса, кг
    k = 2.0  # жёсткость пружины, Н/м
    L1 = 0.3  # расстояние от точки подвеса до пружины, м
    beta = 0.2  # коэффициент затухания, с^-1
    g = 9.81

    # Начальные условия
    phi1_0 = 0.3  # рад
    phi2_0 = -0.2  # рад
    dphi1_0 = 0.0
    dphi2_0 = 0.0

    def pend_system(t, y):
        phi1, phi2, omega1, omega2 = y
        torque_spring_on_1 = k * (phi2 - phi1) * L1 * L1
        torque_spring_on_2 = k * (phi1 - phi2) * L1 * L1
        I = m * L * L
        alpha1 = (-m * g * L * phi1 - beta * omega1 + torque_spring_on_1) / I
        alpha2 = (-m * g * L * phi2 - beta * omega2 + torque_spring_on_2) / I
        return [omega1, omega2, alpha1, alpha2]

    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)
    sol = solve_ivp(pend_system, t_span, [phi1_0, phi2_0, dphi1_0, dphi2_0],
                    t_eval=t_eval, method='RK45')

    phi1 = sol.y[0]
    phi2 = sol.y[1]
    omega1 = sol.y[2]
    omega2 = sol.y[3]

    # Поиск нормальных частот (Фурье)
    dt = t_eval[1] - t_eval[0]
    freqs = np.fft.rfftfreq(len(phi1), dt)
    F1 = np.abs(np.fft.rfft(phi1))
    peaks, _ = find_peaks(F1, height=0.01)
    normal_freqs = freqs[peaks]
    print(f"Нормальные частоты (по Фурье): {normal_freqs[:2]} Гц")

    # Графики
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t_eval, phi1, label='Маятник 1')
    axes[0, 0].plot(t_eval, phi2, label='Маятник 2')
    axes[0, 0].set_xlabel('Время, с')
    axes[0, 0].set_ylabel('Угол, рад')
    axes[0, 0].legend()
    axes[0, 0].grid()
    axes[0, 0].set_title('Углы отклонения')

    axes[0, 1].plot(t_eval, omega1, label='Скорость 1')
    axes[0, 1].plot(t_eval, omega2, label='Скорость 2')
    axes[0, 1].set_xlabel('Время, с')
    axes[0, 1].set_ylabel('Угл. скорость, рад/с')
    axes[0, 1].legend()
    axes[0, 1].grid()
    axes[0, 1].set_title('Угловые скорости')

    axes[1, 0].plot(phi1, phi2, 'b-', alpha=0.7)
    axes[1, 0].set_xlabel('φ1, рад')
    axes[1, 0].set_ylabel('φ2, рад')
    axes[1, 0].set_title('Фазовый портрет')
    axes[1, 0].grid()

    axes[1, 1].plot(freqs, F1)
    axes[1, 1].set_xlim(0, 2)
    axes[1, 1].set_xlabel('Частота, Гц')
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].set_title('Спектр сигнала φ1')
    axes[1, 1].grid()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# ЗАДАЧА 5: ФУРЬЕ-ФИЛЬТРАЦИЯ (из .npy файла)
# --------------------------------------------------------------
def task5_fourier_filter(filename="signal.npy"):
    print("\n=== ЗАДАЧА 5: Фурье-фильтрация ===")

    if not os.path.exists(filename):
        print(f"Файл {filename} не найден.")
        print("Скачайте файл с Google Drive и положите в папку со скриптом.")
        print("Файлы должны быть в формате .npy")
        return

    # Читаем .npy файл
    data = np.load(filename)

    # Преобразуем в одномерный массив если нужно
    if isinstance(data, np.ndarray):
        if data.ndim > 1:
            data = data.flatten()
    else:
        data = np.array(data)

    N = len(data)
    dt = 0.0001  # 0.1 мс = 0.0001 с
    time = np.arange(N) * dt

    print(f"Загружено {N} точек")
    print(f"Длительность сигнала: {time[-1]:.3f} с")

    # Фурье-преобразование
    fft_vals = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, dt)
    amplitude = np.abs(fft_vals)

    # Оценка шума по высоким частотам
    high_freq_start = min(1000, len(amplitude) // 2)
    mean_noise_amp = np.mean(amplitude[high_freq_start:high_freq_start + 500])

    if mean_noise_amp == 0 or np.isnan(mean_noise_amp):
        mean_noise_amp = np.mean(amplitude[len(amplitude) // 4:len(amplitude) // 2])

    threshold = 2.0 * mean_noise_amp
    peaks, _ = find_peaks(amplitude, height=threshold, distance=2)

    # Исключаем нулевую частоту
    peaks = [p for p in peaks if freqs[p] > 1.0]

    signal_freqs = freqs[peaks]
    print(f"Порог шума: {threshold:.3e}")
    print(f"Найденные частоты полезного сигнала (Гц): {signal_freqs}")

    # Фильтрация
    fft_filtered = np.zeros_like(fft_vals, dtype=complex)
    bandwidth = 5

    for p in peaks:
        idx_start = max(0, p - bandwidth)
        idx_end = min(len(fft_vals), p + bandwidth)
        fft_filtered[idx_start:idx_end] = fft_vals[idx_start:idx_end]

    filtered_signal = np.fft.irfft(fft_filtered, n=N)

    # Визуализация
    plt.figure(figsize=(14, 10))

    show_points = min(5000, N)

    plt.subplot(3, 1, 1)
    plt.plot(time[:show_points], data[:show_points], 'gray', alpha=0.7, linewidth=0.5)
    plt.title(f'Исходный зашумлённый сигнал (первые {show_points} точек)')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    max_freq = min(10000, freqs[-1])
    freq_mask = freqs <= max_freq
    plt.plot(freqs[freq_mask], amplitude[freq_mask], 'b-', linewidth=0.5)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Порог ({threshold:.3e})')
    plt.plot(signal_freqs, amplitude[peaks], 'ro', markersize=8, label='Полезные частоты')
    plt.xlabel('Частота, Гц')
    plt.ylabel('|FFT|')
    plt.title('Спектр сигнала')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)

    for freq, peak_amp in zip(signal_freqs, amplitude[peaks]):
        if freq < 5000:
            plt.annotate(f'{freq:.0f} Гц',
                         xy=(freq, peak_amp),
                         xytext=(freq + 100, peak_amp + peak_amp * 0.1),
                         fontsize=8, alpha=0.7)

    plt.subplot(3, 1, 3)
    plt.plot(time[:show_points], filtered_signal[:show_points], 'b-', linewidth=1)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.title('Отфильтрованный сигнал')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nЧастоты полезного сигнала: {', '.join([f'{f:.1f}' for f in signal_freqs])} Гц")
    print(f"Всего найдено {len(signal_freqs)} гармоник")

    return signal_freqs, filtered_signal, time


# --------------------------------------------------------------
# ЗАПУСК ВСЕХ ЗАДАЧ
# --------------------------------------------------------------
if __name__ == "__main__":
    task1_magnetron()
    task2_coil()
    task3_solenoid_field()
    task4_coupled_pendulums()
    task5_fourier_filter("signal_1.npy")