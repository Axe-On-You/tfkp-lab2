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

def get_grid_points():
    """
    Создает сетку точек в Единичном круге K.
    Используем полярную сетку для красивых линий сетки.
    """

    rs = np.linspace(0.1, 1.0, 10)  # Радиусы от 0.1 до 1.0
    thetas = np.linspace(0, 2 * np.pi, 30, endpoint=False)  # Углы от 0 до 2pi

    lines = []

    # 1. Радиальные линии (лучи)
    for t in thetas:
        r_line = np.linspace(0, 1.0, 100)
        lines.append(r_line * np.exp(1j * t))

    # 2. Дуговые линии (окружности)
    for r in rs:
        t_arc = np.linspace(0, 2 * np.pi, 100)
        lines.append(r * np.exp(1j * t_arc))

    # Преобразуем список массивов в один длинный массив для анимации
    # Добавляем NaN разделители, чтобы линии не соединялись
    Z_lines = []
    for line in lines:
        Z_lines.extend(line)
        Z_lines.append(np.nan + 1j * np.nan)

    return np.array(Z_lines)


# =========================================================================
# 2. ФУНКЦИЯ ОТОБРАЖЕНИЯ (Шаг 3: K -> G)
# =========================================================================

def mapping(z2):
    """
    Отображение Единичного круга K на Круг G радиуса pi.
    w = pi * z2 (Гомотетия)
    """
    return np.pi * z2


# Исходные и конечные точки
Z2 = get_grid_points()
W = mapping(Z2)


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
    # Используем sqrt для равномерного распределения точек внутри круга
    r_rand = np.sqrt(np.random.uniform(0, 1, num_pts))
    t_rand = np.random.uniform(0, 2 * np.pi, num_pts)
    Z2_cloud = r_rand * np.exp(1j * t_rand)
    W_cloud = mapping(Z2_cloud)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Раскраска по углу
    colors = np.angle(Z2_cloud)

    # Левая часть: Исходная область K
    ax[0].scatter(Z2_cloud.real, Z2_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[0].set_title("Исходная область $K$\n($|z_2| < 1$)")
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].axvline(0, color='k', lw=0.8)
    ax[0].add_patch(plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--'))
    ax[0].set_xlim(-1.5, 1.5)
    ax[0].set_ylim(-1.5, 1.5)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_aspect('equal')

    # Правая часть: Образ конформного отображения G
    ax[1].scatter(W_cloud.real, W_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[1].set_title("Результат отображения $G$\n($w = \\pi z_2$, $|w| < \\pi$)")
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].axvline(0, color='k', lw=0.8)
    # Добавляем границу целевого круга
    ax[1].add_patch(plt.Circle((0, 0), np.pi, color='red', fill=False, linestyle='--'))
    ax[1].set_xlim(-4, 4)
    ax[1].set_ylim(-4, 4)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "static_mapping3.png"), dpi=200)
    print("Картинка 'output/img/static_mapping3.png' сохранена.")
    plt.close()


save_static_report_image()

# =========================================================================
# 4. АНИМАЦИЯ
# =========================================================================

fig, ax = plt.subplots(figsize=(6, 6))
# Устанавливаем масштаб, чтобы вместить обе области (K и G)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal')

line_plot, = ax.plot([], [], 'b-', lw=1, alpha=0.6)
title = ax.set_title("Конформное отображение: Шаг 3 (K $\\to$ G)")


def update(frame):
    t = frame  # t меняется от 0 (K) до 1 (G)

    # Линейная интерполяция между Z2 и W
    Z_curr = (1 - t) * Z2 + t * W

    line_plot.set_data(Z_curr.real, Z_curr.imag)

    # Обновление заголовка
    if t < 0.01:
        title.set_text("Начало: Единичный круг $K$")
    elif t > 0.99:
        title.set_text("Конец: Целевой круг $G$ радиуса $\\pi$")
    else:
        title.set_text(f"Гомотетия... t={t:.2f}")

    return line_plot, title


# Кадры: 10 пауз в начале, 80 кадров движения, 20 пауз в конце
frames = np.concatenate([np.zeros(10), np.linspace(0, 1, 80), np.ones(20)])

try:
    # Сохранение с высокой частотой кадров для плавности
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    ani.save(os.path.join(gif_dir, "conformal_animation2.gif"), writer=PillowWriter(fps=25))
    print("Анимация 'output/gif/conformal_animation3.gif' сохранена.")
except Exception as e:
    print(f"Не удалось сохранить GIF. Убедитесь, что установлены numpy, matplotlib, Pillow: {e}")

# plt.show()