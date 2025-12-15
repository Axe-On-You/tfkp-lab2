import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Создание папок для сохранения
os.makedirs("output/img", exist_ok=True)
os.makedirs("output/gif", exist_ok=True)


# =========================================================================
# 1. ГЕНЕРАЦИЯ СЕТКИ (для исходной области D)
# =========================================================================

def get_grid_points():
    """
    Создает сетку точек в исходном секторе D: pi/4 < arg(z) < 3pi/4.
    """
    rs = np.linspace(0.1, 2.0, 15)  # Радиусы от 0.1 до 2.0
    thetas = np.linspace(np.pi / 4, 3 * np.pi / 4, 30)  # Углы от pi/4 до 3pi/4

    lines = []
    # 1. Радиальные линии
    for t in thetas:
        r_line = np.linspace(0, 2.0, 100)
        lines.append(r_line * np.exp(1j * t))
    # 2. Дуговые линии
    for r in rs:
        t_arc = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        lines.append(r * np.exp(1j * t_arc))

    Z_lines = []
    for line in lines:
        Z_lines.extend(line)
        Z_lines.append(np.nan + 1j * np.nan)
    return np.array(Z_lines)


# =========================================================================
# 2. ФУНКЦИИ ОТОБРАЖЕНИЯ
# =========================================================================

def f1(z):
    # D -> H: z1 = -i * z^2
    return -1j * (z ** 2)


def f2(z1):
    # H -> K: z2 = (z1 - i) / (z1 + i)
    return (z1 - 1j) / (z1 + 1j)


def f3(z2):
    # K -> G: w = pi * z2
    return np.pi * z2


# Исходные точки и все промежуточные/конечные результаты
Z = get_grid_points()
Z1 = f1(Z)  # H
Z2 = f2(Z1)  # K
W = f3(Z2)  # G


# =========================================================================
# 3. СОХРАНЕНИЕ СТАТИЧЕСКОЙ КАРТИНКИ (4 этапа в одной фигуре)
# =========================================================================

def get_cloud_points(num_pts=10000):
    """Генерирует облако точек в исходном секторе D."""
    r_rand = np.sqrt(np.random.uniform(0, 4, num_pts))
    t_rand = np.random.uniform(np.pi / 4, 3 * np.pi / 4, num_pts)
    return r_rand * np.exp(1j * t_rand)


def save_full_static_image():
    Z_cloud = get_cloud_points()
    Z1_cloud = f1(Z_cloud)
    Z2_cloud = f2(Z1_cloud)
    W_cloud = f3(Z2_cloud)

    clouds = [Z_cloud, Z1_cloud, Z2_cloud, W_cloud]
    titles = [
        "(a) Область $D$\n($z$-плоскость)",
        "(b) Область $H$\n($z_1 = -i z^2$)",
        "(c) Область $K$\n($z_2 = (z_1-i)/(z_1+i)$)",
        "(d) Область $G$\n($w = \\pi z_2$)"
    ]
    x_limits = [(-2.5, 2.5), (-4.5, 4.5), (-1.5, 1.5), (-4, 4)]
    y_limits = [(-0.5, 2.5), (-0.5, 4.5), (-1.5, 1.5), (-4, 4)]
    colors = np.angle(Z_cloud)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    for i in range(4):
        ax[i].scatter(clouds[i].real, clouds[i].imag, c=colors, cmap='hsv', s=1, alpha=0.5)
        ax[i].set_title(titles[i])
        ax[i].set_xlim(x_limits[i])
        ax[i].set_ylim(y_limits[i])
        ax[i].set_aspect('equal')
        ax[i].axhline(0, color='k', lw=0.8)
        ax[i].axvline(0, color='k', lw=0.8)

        # Добавляем границы для кругов
        if i == 2:  # K
            ax[i].add_patch(plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--'))
        if i == 3:  # G
            ax[i].add_patch(plt.Circle((0, 0), np.pi, color='red', fill=False, linestyle='--'))

    plt.tight_layout()
    plt.savefig(os.path.join("output/img", "full_mapping.png"), dpi=200)
    print("Статическое изображение 'output/img/full_mapping.png' сохранено.")
    plt.close()


save_full_static_image()

# =========================================================================
# 4. АНИМАЦИЯ D -> H -> K -> G
# =========================================================================

fig, ax = plt.subplots(figsize=(7, 7))
# Устанавливаем широкий масштаб, чтобы вместить все преобразования,
# включая большой круг G (радиус pi ~ 3.14)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal')

line_plot, = ax.plot([], [], 'b-', lw=1, alpha=0.6)
title = ax.set_title("Конформное отображение: $D \\to H \\to K \\to G$")

# Определяем "целевые" позиции для анимации
POSITIONS = [Z, Z1, Z2, W]
# Общее число сегментов для интерполяции
NUM_SEGMENTS = len(POSITIONS) - 1


def update(frame_index):
    # Общая шкала времени t от 0 до 100
    total_frames = 100
    t_global = frame_index / total_frames

    # Определяем, в каком сегменте мы находимся
    segment_length = total_frames / NUM_SEGMENTS  # 33.33 кадра на сегмент

    # Чтобы обеспечить плавный переход и остановки, используем явные интервалы
    # Сегмент 0: Z -> Z1 (0-25)
    # Сегмент 1: Z1 -> Z2 (35-60)
    # Сегмент 2: Z2 -> W (70-95)

    # 1. Начальная пауза в D
    if frame_index < 5:
        Z_curr = Z
        title.set_text("Стадия 1: Исходный сектор $D$")

    # 2. Переход D -> H
    elif 5 <= frame_index < 30:
        t_local = (frame_index - 5) / 25  # t_local от 0 до 1
        Z_curr = (1 - t_local) * Z + t_local * Z1
        title.set_text(f"Переход $D \\to H$: $z_1 = -i z^2$")

    # 3. Пауза в H
    elif 30 <= frame_index < 35:
        Z_curr = Z1
        title.set_text("Стадия 2: Верхняя полуплоскость $H$")

    # 4. Переход H -> K
    elif 35 <= frame_index < 60:
        t_local = (frame_index - 35) / 25
        Z_curr = (1 - t_local) * Z1 + t_local * Z2
        title.set_text(r"Переход $H \to K$: $z_2 = \frac{z_1 - i}{z_1 + i}$")

    # 5. Пауза в K
    elif 60 <= frame_index < 65:
        Z_curr = Z2
        title.set_text("Стадия 3: Единичный круг $K$")

    # 6. Переход K -> G
    elif 65 <= frame_index < 90:
        t_local = (frame_index - 65) / 25
        Z_curr = (1 - t_local) * Z2 + t_local * W
        title.set_text(f"Переход $K \\to G$: $w = \\pi z_2$")

    # 7. Финальная пауза в G
    else:  # frame_index >= 90
        Z_curr = W
        title.set_text("Конец: Целевой круг $G$ радиуса $\\pi$")

    line_plot.set_data(Z_curr.real, Z_curr.imag)
    return line_plot, title


# Запускаем анимацию на 100 кадров (с паузами и переходами)
frames = np.arange(100)

try:
    ani = FuncAnimation(fig, update, frames=frames, interval=60, blit=True)
    ani.save("output/gif/conformal_animation_full.gif", writer=PillowWriter(fps=15))
    print("Анимация 'output/gif/conformal_animation_full.gif' сохранена.")
except Exception as e:
    print(f"Не удалось сохранить GIF. Убедитесь, что установлены numpy, matplotlib, Pillow: {e}")