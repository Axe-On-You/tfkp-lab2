import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Пути для сохранения
img_dir = os.path.join(project_root, "output", "img")
gif_dir = os.path.join(project_root, "output", "gif")

# Создание папок для сохранения, если они не существуют
os.makedirs(img_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)


# =========================================================================
# 1. ГЕНЕРАЦИЯ СЕТКИ (для исходной области H: Im(z1) > 0)
# =========================================================================

def get_grid_points():
    """
    Создает сетку точек в Верхней полуплоскости H.
    Используем прямоугольную сетку.
    """

    # Вещественные линии (вертикальные)
    reals = np.linspace(-4, 4, 15)
    # Мнимые линии (горизонтальные)
    imags = np.linspace(0.1, 4, 10)

    lines = []

    # 1. Вертикальные линии (Re = const)
    for r in reals:
        im_line = np.linspace(0, 4, 100)
        lines.append(r + 1j * im_line)

    # 2. Горизонтальные линии (Im = const)
    for i in imags:
        re_line = np.linspace(-4, 4, 100)
        lines.append(re_line + 1j * i)

    # Преобразуем список массивов в один длинный массив для анимации
    # Добавляем NaN разделители, чтобы линии не соединялись
    Z_lines = []
    for line in lines:
        Z_lines.extend(line)
        Z_lines.append(np.nan + 1j * np.nan)

    return np.array(Z_lines)


# =========================================================================
# 2. ФУНКЦИЯ ОТОБРАЖЕНИЯ (Шаг 2: H -> K)
# =========================================================================

def mapping(z1):
    """
    Отображение Верхней полуплоскости H на Единичный круг K.
    z2 = (z1 - i) / (z1 + i)
    """
    return (z1 - 1j) / (z1 + 1j)


# Исходные и конечные точки
Z1 = get_grid_points()
Z2 = mapping(Z1)


# =========================================================================
# 3. СОХРАНЕНИЕ СТАТИЧЕСКОЙ КАРТИНКИ (для отчета)
# =========================================================================

def save_static_report_image():
    """
    Генерирует и сохраняет статическое изображение для отчета,
    сравнивающее исходную и отображенную области.
    """
    # Генерируем плотное облако точек
    num_pts = 10000
    Z1_real = np.random.uniform(-4, 4, num_pts)
    Z1_imag = np.random.uniform(0.1, 4, num_pts)
    Z1_cloud = Z1_real + 1j * Z1_imag
    Z2_cloud = mapping(Z1_cloud)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Раскраска по углу (чтобы отследить конформность)
    colors = np.angle(Z1_cloud)

    # Левая часть: Исходная область H
    ax[0].scatter(Z1_cloud.real, Z1_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[0].set_title("Исходная область $H$\n($\\text{Im } z_1 > 0$)")
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].axvline(0, color='k', lw=0.8)
    ax[0].set_xlim(-4.5, 4.5)
    ax[0].set_ylim(0, 4.5)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_aspect('equal')

    # Правая часть: Образ конформного отображения K
    ax[1].scatter(Z2_cloud.real, Z2_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[1].set_title("Результат отображения $K$\n($z_2 = (z_1 - i)/(z_1 + i)$, $|z_2| < 1$)")
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].axvline(0, color='k', lw=0.8)
    # Добавляем границу единичного круга
    ax[1].add_patch(plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--'))
    ax[1].set_xlim(-1.5, 1.5)
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "static_mapping2.png"), dpi=200)
    print("Картинка 'output/img/static_mapping2.png' сохранена.")
    plt.close()


save_static_report_image()

# =========================================================================
# 4. АНИМАЦИЯ
# =========================================================================

fig, ax = plt.subplots(figsize=(6, 6))
# Устанавливаем масштаб, чтобы вместить обе области (H и K)
# У Верхней полуплоскости Re от -4 до 4, Im от 0 до 4
# У Единичного круга Re от -1 до 1, Im от -1 до 1
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal')

line_plot, = ax.plot([], [], 'b-', lw=1, alpha=0.6)
title = ax.set_title("Конформное отображение: Шаг 2 (H $\\to$ K)")


def update(frame):
    t = frame  # t меняется от 0 (H) до 1 (K)

    # Линейная интерполяция между Z1 и Z2
    Z_curr = (1 - t) * Z1 + t * Z2

    line_plot.set_data(Z_curr.real, Z_curr.imag)

    # Обновление заголовка
    if t < 0.01:
        title.set_text("Начало: Верхняя полуплоскость $H$")
    elif t > 0.99:
        title.set_text("Конец: Единичный круг $K$")
    else:
        title.set_text(f"Преобразование Мёбиуса... t={t:.2f}")

    return line_plot, title


# Кадры: 10 пауз в начале, 80 кадров движения, 20 пауз в конце
frames = np.concatenate([np.zeros(10), np.linspace(0, 1, 80), np.ones(20)])

try:
    # Сохранение с высокой частотой кадров для плавности
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    ani.save(os.path.join(gif_dir, "conformal_animation2.gif"), writer=PillowWriter(fps=25))
    print("Анимация 'output/gif/conformal_animation2.gif' сохранена.")
except Exception as e:
    print(f"Не удалось сохранить GIF. Убедитесь, что установлены numpy, matplotlib, Pillow: {e}")