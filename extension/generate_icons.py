"""Script to generate ReviewLens browser extension icons (16x16, 48x48, 128x128).

Uses Pillow to render a clean magnifying lens / shield graphic in ReviewLens brand colors.
"""

from pathlib import Path
from PIL import Image, ImageDraw


def generate_icon(size: int, output_path: Path) -> None:
    """Generate a high-quality icon of given square size and save as PNG.

    Args:
        size: Width and height of the icon in pixels.
        output_path: Path where the PNG icon will be saved.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle or circle
    padding = max(1, int(size * 0.05))
    bg_box = [padding, padding, size - padding, size - padding]
    corner_radius = max(2, int(size * 0.22))
    
    # Modern deep indigo / violet background
    bg_color = (30, 41, 59, 255)  # Slate-800
    draw.rounded_rectangle(bg_box, radius=corner_radius, fill=bg_color)

    # Draw magnifying lens
    # Circle for lens
    cx = int(size * 0.44)
    cy = int(size * 0.44)
    radius = int(size * 0.26)
    stroke = max(1, int(size * 0.09))

    # Outer lens circle in vibrant cyan
    cyan_color = (14, 165, 233, 255)  # Sky-500
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=cyan_color,
        width=stroke,
    )

    # Lens handle
    hx_start = cx + int(radius * 0.7)
    hy_start = cy + int(radius * 0.7)
    hx_end = size - int(size * 0.16)
    hy_end = size - int(size * 0.16)
    draw.line([hx_start, hy_start, hx_end, hy_end], fill=cyan_color, width=stroke)

    # Small center dot / spark in white/cyan
    dot_r = max(1, int(size * 0.07))
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(255, 255, 255, 230),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def main() -> None:
    """Generate all extension icon sizes."""
    base_dir = Path(__file__).resolve().parent
    icons_dir = base_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    for size in (16, 48, 128):
        icon_path = icons_dir / f"icon{size}.png"
        generate_icon(size, icon_path)
        print(f"Generated {icon_path.name} ({size}x{size})")


if __name__ == "__main__":
    main()
