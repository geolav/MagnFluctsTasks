# MagnFluctsTasks

Моделирование задач по магнитным колебаниям и полям на Python.

---

## Задачи и графики

### Задача 1 — Движение электрона в магнетроне

Моделирование траектории электрона в скрещенных электрическом и магнитном полях цилиндрического магнетрона.

**Полезные источники:**
- [Feynman Lectures, глава 29](https://www.feynmanlectures.caltech.edu/II_29.html) — классическое объяснение циклоидального движения в скрещенных полях E и B, с иллюстрациями
- [PDF: симуляция траекторий заряженных частиц в полях](https://d-arora.github.io/Doing-Physics-With-Matlab/mpDocs/em_vBE.pdf) — много графиков траекторий, примеры на Matlab

---

### Задача 2 и 3 — Катушка и соленоид

Расчёт параметров катушки и визуализация силовых линий магнитного поля соленоида.

**Полезные источники:**
- [Medium: Visualizing Magnetic Field in Python](https://medium.com/@mathcube7/visualizing-the-magnetic-field-in-python-2d4fc4509a21) — про matplotlib и закон Био-Савара, с примерами кода и картинками силовых линий
- [Python Matplotlib Tips: streamplot для силовых линий](https://pythonmatplotlibtips.blogspot.com/2017/12/plot-continuous-magnetic-field-lines.html) — готовые примеры с кодом
- [Mayavi: magnetic field lines](https://docs.enthought.com/mayavi/mayavi/auto/example_magnetic_field_lines.html) — как должны выглядеть правильно замкнутые силовые линии витка и соленоида

---

### Задача 4 — Связанные маятники

Нахождение нормальных мод системы двух связанных маятников через FFT-анализ.

**Полезные источники:**
- [MIT OpenCourseWare, глава 3 Normal Modes](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/29afe7e96aadefce1bd80486771aeae9_MIT8_03SCF16_Text_Ch3.pdf) — подробный разбор с графиками FFT-спектра
- [Physics LibreTexts — связанные маятники](https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Graduate_Classical_Mechanics_(Fowler)/17:_Small_Oscillations/17.05:_Three_Coupled_Pendulums) — теория и примеры нормальных мод
- [PhET: симулятор Normal Modes](https://phet.colorado.edu/en/simulations/normal-modes) — интерактивно видно как выглядят спектры при разных начальных условиях

---

### Задача 5 — Фурье-фильтрация сигнала

Очистка зашумлённого сигнала методом порогового фильтра в частотной области (PSD threshold).

**Полезные источники:**
- [Earth Inversion: Signal denoising using FFT](https://earthinversion.com/techniques/signal-denoising-using-fast-fourier-transform/) — очень похожая задача, есть графики исходный / спектр / отфильтрованный
- [Towards Data Science: Clean Up Data Noise with Fourier Transform](https://towardsdatascience.com/clean-up-data-noise-with-fourier-transform-in-python-7480252fd9c9/) — простой и наглядный пример с кодом на Python
- [Real Python: Fourier Transforms with scipy.fft](https://realpython.com/python-scipy-fft/) — подробный туториал по всему процессу

---

## Запуск

```bash
python main.py
```

Графики сохраняются в папку `plots/`.

## Зависимости

```bash
pip install numpy matplotlib scipy
```