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


# === 1. ГЕНЕРАЦИЯ ТОЧЕК (Сектор) ===
def get_grid_points():
    """
    Создает сетку точек в секторе pi/4 < arg(z) < 3pi/4.
    Используем полярную сетку для красивых линий сетки.
    """
    # Радиусы от 0.1 до 2.0
    rs = np.linspace(0.1, 2.0, 15)
    # Углы от pi/4 до 3pi/4
    thetas = np.linspace(np.pi / 4, 3 * np.pi / 4, 30)

    # 1. Радиальные линии (лучи)
    lines = []
    for t in thetas:
        r_line = np.linspace(0, 2.0, 100)
        lines.append(r_line * np.exp(1j * t))

    # 2. Дуговые линии (окружности)
    for r in rs:
        t_arc = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        lines.append(r * np.exp(1j * t_arc))

    # Преобразуем список массивов в один длинный массив для анимации
    # Добавляем NaN разделители, чтобы линии не соединялись
    Z_lines = []
    for line in lines:
        Z_lines.extend(line)
        Z_lines.append(np.nan + 1j * np.nan)

    return np.array(Z_lines)


# === 2. ФУНКЦИЯ ОТОБРАЖЕНИЯ ===
def mapping(z):
    # z_1 = -i * z^2
    return -1j * (z ** 2)


# === 3. ПОДГОТОВКА ДАННЫХ ===
Z = get_grid_points()
Z1 = mapping(Z)


# === 4. СТАТИЧЕСКАЯ КАРТИНКА (ДЛЯ ОТЧЕТА) ===
def save_static_report_image():
    # Генерируем плотное облако точек для красивой картинки в отчет
    # (в отличие от линий сетки выше)
    num_pts = 10000
    r_rand = np.sqrt(np.random.uniform(0, 4, num_pts))  # sqrt для равномерности круга
    t_rand = np.random.uniform(np.pi / 4, 3 * np.pi / 4, num_pts)
    Z_cloud = r_rand * np.exp(1j * t_rand)
    Z1_cloud = mapping(Z_cloud)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Раскраска по углу (чтобы видеть, куда переходят границы)
    colors = np.angle(Z_cloud)

    # Plot 1: Source
    ax[0].scatter(Z_cloud.real, Z_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[0].set_title("Исходная область $D$\n($\\pi/4 < \\arg z < 3\\pi/4$)")
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].axvline(0, color='k', lw=0.8)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_aspect('equal')

    # Plot 2: Target
    ax[1].scatter(Z1_cloud.real, Z1_cloud.imag, c=colors, cmap='hsv', s=1, alpha=0.5)
    ax[1].set_title("Результат отображения $H$\n($z_1 = -i z^2$)")
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].axvline(0, color='k', lw=0.8)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "static_mapping1.png"), dpi=200)
    print("Картинка 'static_mapping1.png' сохранена.")
    plt.close()


save_static_report_image()

# === 5. АНИМАЦИЯ ===
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-1, 4.5)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_aspect('equal')

line_plot, = ax.plot([], [], 'b-', lw=1, alpha=0.6)
title = ax.set_title("Conformal Mapping")


def update(frame):
    t = frame  # t меняется от 0 до 1

    # Линейная интерполяция между Z и W
    # Z_curr = (1-t)*Z + t*Z1

    Z_curr = (1 - t) * Z + t * Z1

    line_plot.set_data(Z_curr.real, Z_curr.imag)

    # Меняем цвет заголовка или текст
    if t < 0.01:
        title.set_text("Начало: Данное изображение сектора")
    elif t > 0.99:
        title.set_text("Конец: Верхняя полуплоскость")
    else:
        title.set_text(f"Изменение... t={t:.2f}")

    return line_plot, title


# Кадры: 10 пауз в начале, 60 кадров движения, 20 пауз в конце
frames = np.concatenate([np.zeros(10), np.linspace(0, 1, 80), np.ones(20)])

ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)

try:
    ani.save(os.path.join(gif_dir, "conformal_animation1.gif"), writer=PillowWriter(fps=25))
    print("Анимация 'conformal_animation1.gif' сохранена.")
except Exception as e:
    print(f"Не удалось сохранить GIF: {e}")