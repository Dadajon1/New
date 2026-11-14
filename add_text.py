import os
import json
import re
import requests
import tempfile
import copy
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path, pdfinfo_from_path
from pypdf import PdfWriter, PdfReader
import gc
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from utils import get_logger

logger = get_logger()

Image.MAX_IMAGE_PIXELS = 300_000_000

FONT_DIR="fonts"
FONT_CACHE: dict = {}
FONTS_DIR  = Path(FONT_DIR)
FONTS_DIR.mkdir(parents=True, exist_ok=True)



def _download_font(font_name: str, font_weight: int, dest_path: Path) -> bool:
    """
    Downloads a font from Google Fonts and saves it to dest_path.
    Returns True on success, False on failure.
    """
    try:
        api_url = (
            f"https://fonts.googleapis.com/css2?family="
            f"{font_name.replace(' ', '+')}:wght@{font_weight}&display=swap"
        )
        css_response = requests.get(
            api_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        css_response.raise_for_status()

        font_url = None
        for line in css_response.text.split("\n"):
            if "url(" in line and ".ttf" in line:
                start    = line.find("url(") + 4
                end      = line.find(")", start)
                font_url = line[start:end].strip("\"'")
                break

        if not font_url:
            logger.error(
                f"Font CSS not found:  {font_name} w{font_weight}"
            )
            return False

        font_response = requests.get(font_url, timeout=15)
        font_response.raise_for_status()

        dest_path.write_bytes(font_response.content)
        logger.info(
            f"Downloaded font: {dest_path}",
            extra={"font_name": font_name, "weight": font_weight},
        )
        return True

    except Exception as e:
        return False


def load_local_font(font_name: str, font_weight: int, size: int = 65) -> ImageFont.FreeTypeFont:
    key = (font_name, font_weight, size)
    if key in FONT_CACHE:
        return FONT_CACHE[key]

    if font_weight >= 700:
        weight_suffix = "Bold"
    elif font_weight >= 600:
        weight_suffix = "SemiBold"
    else:
        weight_suffix = "Regular"

    filename  = f"{font_name.replace(' ', '')}-{weight_suffix}.ttf"
    font_path = FONTS_DIR / filename

    # Download if not on disk yet
    if not font_path.exists():
        success = _download_font(font_name, font_weight, font_path)
        if not success:
            return ImageFont.load_default()

    font = ImageFont.truetype(str(font_path), size)
    FONT_CACHE[key] = font
    return font


_CMAP_CACHE: dict = {}


def font_covers(font_name: str, font_weight: int, text: str) -> bool:
    """True if the font file has a glyph for every visible character of text."""
    font = load_local_font(font_name, font_weight, 20)
    path = getattr(font, "path", None)
    if not path:
        return True
    if path not in _CMAP_CACHE:
        try:
            from fontTools.ttLib import TTFont
            _CMAP_CACHE[path] = set(TTFont(path).getBestCmap().keys())
        except Exception:
            _CMAP_CACHE[path] = None
    cmap = _CMAP_CACHE[path]
    if cmap is None:
        return True
    return all(ord(c) in cmap for c in text if not c.isspace() and c not in "*")


def warm_font_cache(full_config: list) -> None:
    """
    Pre-load every unique (font, weight, size) combo found in the JSON config
    before page processing begins.
    Any missing fonts are downloaded and saved to fonts/ here,
    so the page loop has zero I/O or network calls.
    """
    seen = set()
    for page_cfg in full_config:
        for obj in page_cfg:
            font_name   = obj.get("font", "Roboto")
            font_weight = obj.get("font_weight", 400)
            font_size   = obj.get("font_size", 40)

            combos = [
                (font_name, font_weight, font_size),
                (font_name, 700,         font_size),
                (font_name, 700,         font_size + 15),
            ]
            for combo in combos:
                if combo not in seen:
                    seen.add(combo)
                    load_local_font(*combo)

    logger.info("Font cache warmed", extra={"cached_fonts": len(FONT_CACHE)})



def parse_color(color_str, default=(255, 255, 255)):
    """Parses hex color strings to RGB tuples."""
    if not color_str or not isinstance(color_str, str):
        return default
    color_str = color_str.strip().lower()
    try:
        if color_str.startswith('#'):
            hex_val = color_str[1:]
            if len(hex_val) == 6:
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                return (r, g, b)
    except:
        pass
    return default



def get_text_length(draw, text, font):
    if not text:
        return 0
    return draw.textlength(text, font=font)


def wrap_text_into_lines(draw, text, regular_font, bold_font_normal, max_width, actual_name=None):
    """Wraps text into lines and identifies segments for larger, bolder name rendering."""
    lines = []
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append([])
            continue

        current_line  = []
        current_width = 0.0
        tokens = re.findall(r'\*.*?\*|\S+|\s+', paragraph)

        for token in tokens:
            if not token.strip():
                if current_line:
                    space_w = get_text_length(draw, token, regular_font)
                    if current_width + space_w <= max_width:
                        current_line.append((token, regular_font))
                        current_width += space_w
                continue

            is_emphasis = token.startswith('*') and token.endswith('*') and len(token) > 2
            if is_emphasis:
                inner_text = token[1:-1]
                is_name    = "{name}" in inner_text or (actual_name and actual_name in inner_text)

                if is_name:
                    font = load_local_font(
                        regular_font.getname()[0],
                        700,
                        regular_font.size + 15,
                    )
                else:
                    font = bold_font_normal
                display_text = inner_text
            else:
                display_text = token
                font         = regular_font

            w = get_text_length(draw, display_text, font)

            if current_line and current_width + w > max_width:
                lines.append(current_line)
                current_line  = []
                current_width = 0.0

            current_line.append((display_text, font))
            current_width += w

        if current_line:
            lines.append(current_line)

    return lines


def draw_aligned_line(draw, x, y, segments, color, align, max_width, base_size, shadow_cfg=None):
    """Draws text segments with customizable shadow/glow logic."""
    if not segments:
        return

    content_width = sum(get_text_length(draw, text, font) for text, font in segments)

    start_x = x
    if align == "center":
        start_x = x + (max_width - content_width) / 2
    elif align == "right":
        start_x = x + max_width - content_width

    space_per_gap = 0
    if align == "justify" and len(segments) > 1 and (content_width / max_width) > 0.75:
        space_per_gap = (max_width - content_width) / (len(segments) - 1)

    if not shadow_cfg:
        shadow_cfg = {"offset": [2, 2], "color": "#000000", "opacity": 110, "layers": 3, "mode": "drop"}

    s_color      = parse_color(shadow_cfg.get("color", "#000000"))
    s_opacity    = shadow_cfg.get("opacity", 110)
    layers       = shadow_cfg.get("layers", 3)
    mode         = shadow_cfg.get("mode", "drop")
    off_x, off_y = shadow_cfg.get("offset", [2, 2])
    shadow_fill  = (*s_color, s_opacity)

    current_x = start_x
    for i, (text, font) in enumerate(segments):
        y_offset = 0
        if font.size > base_size:
            y_offset = -(font.size - base_size) // 1.2

        # Shadow Rendering 
        if layers > 0:
            for layer in range(1, layers + 1):
                ratio  = layer / layers
                sx, sy = off_x * ratio, off_y * ratio

                if mode == "glow":
                    for dx, dy in [(sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy), (sx, 0), (-sx, 0), (0, sy), (0, -sy)]:
                        draw.text((current_x + dx, y + y_offset + dy), text, fill=shadow_fill, font=font)
                else:
                    draw.text((current_x + sx, y + y_offset + sy), text, fill=shadow_fill, font=font)

        # Main Text 
        draw.text((current_x, y + y_offset), text, fill=color, font=font)

        current_x += get_text_length(draw, text, font)
        if i < len(segments) - 1:
            current_x += space_per_gap



def apply_texts_to_image(image: Image.Image, texts: list, actual_name=None) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    draw = ImageDraw.Draw(image)
    img_width, _ = image.size

    for text_obj in texts:
        content = text_obj.get("content", "").strip()
        if not content:
            continue
        if text_obj.get("upper"):
            content = content.upper()
        # e.g. "Malika" (princess) above the name: skip it when the child is called Malika.
        if text_obj.get("hide_if_name") and actual_name and content.lower() == actual_name.strip().lower():
            continue

        pos_x        = text_obj.get("position_x", 100)
        pos_y        = text_obj.get("position_y", 100)
        size         = text_obj.get("font_size", 40)
        font_name    = text_obj.get("font", "Roboto")
        weight       = text_obj.get("font_weight", 400)
        color        = parse_color(text_obj.get("color"), (255, 255, 255))
        align        = text_obj.get("align", "left").lower()
        line_spacing = text_obj.get("line_spacing", 0.95)
        max_width    = text_obj.get("max_width", img_width - pos_x - 150)
        shadow_cfg   = text_obj.get("shadow", None)

        # Customer names may be Cyrillic: fall back to a font that has the glyphs.
        if not font_covers(font_name, weight, content):
            font_name = text_obj.get("fallback_font", "Rubik")
            weight    = text_obj.get("fallback_weight", 700)

        # "fit": shrink a one-line title (e.g. a long name) until it fits max_width.
        if text_obj.get("fit"):
            while size > 40 and get_text_length(draw, content, load_local_font(font_name, weight, size)) > max_width * 0.92:
                size = int(size * 0.92)

        regular_font     = load_local_font(font_name, weight, size)
        bold_font_normal = load_local_font(font_name, 700, size)

        wrapped_lines = wrap_text_into_lines(draw, content, regular_font, bold_font_normal, max_width, actual_name)

        current_y = pos_y
        for line_segments in wrapped_lines:
            if not line_segments:
                current_y += int(size * 1.2)
                continue

            draw_aligned_line(
                draw, pos_x, current_y, line_segments, color,
                align, max_width, size, shadow_cfg=shadow_cfg
            )
            current_y += int(size * line_spacing + 16)

    return image.convert('RGB')



def add_text_to_pdf(input_pdf, json_path, output_pdf, lang,target, replacements=None, dpi=200):
   
    replacements = replacements or {}
    actual_name  = replacements.get("name")

    # Resolve language-specific JSON path
    book_base_name  = json_path.replace(".json", "")
    lang_map        = {"ru": "_russian", "uz": "_uzbek", "en": "_english"}
    suffix          = lang_map.get(lang, "_english")
    final_json_path = f"{book_base_name}{suffix}.json"

    if not os.path.exists(final_json_path):
        raise FileNotFoundError(f"CRITICAL: JSON config not found at {os.path.abspath(final_json_path)}")

    with open(final_json_path, encoding='utf-8') as f:
        full_config = json.load(f)

    warm_font_cache(full_config)

    reader      = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    TARGET = target
    page_cfgs = []
    for page_num in range(1, total_pages + 1):
        page_cfg = full_config[page_num - 1] if page_num - 1 < len(full_config) else []
        if replacements:
            updated_cfg = copy.deepcopy(page_cfg)
            for obj in updated_cfg:
                if "content" in obj:
                    for k, v in replacements.items():
                        obj["content"] = obj["content"].replace("{" + k + "}", v)
            page_cfg = updated_cfg

        if page_num == 1 and actual_name and not any(o.get("align") == "center" for o in page_cfg):
            if len(actual_name) > 14:
                for obj in page_cfg:
                    if actual_name in obj.get("content", ""):
                        obj["position_x"] = obj.get("position_x", 100) - 200
            elif len(actual_name) < 10:
                for obj in page_cfg:
                    if actual_name in obj.get("content", ""):
                        obj["position_x"] = obj.get("position_x", 100) + 150
        page_cfgs.append(page_cfg)

    def _target_size(page):
        # Same size the old path produced: render at `dpi`, then thumbnail into TARGET box.
        w_pts = float(page.mediabox.width)
        h_pts = float(page.mediabox.height)
        rw, rh = max(1, round(w_pts * dpi / 72)), max(1, round(h_pts * dpi / 72))
        if rw > TARGET or rh > TARGET:
            scale = min(TARGET / rw, TARGET / rh)
            rw, rh = max(1, round(rw * scale)), max(1, round(rh * scale))
        return rw, rh

    def _load_page(idx):
        """Fast path: the swapped PDF has one embedded image per page, so decode it
        directly instead of re-rendering the page with pdftoppm."""
        page = reader.pages[idx]
        try:
            imgs = page.images
            if len(imgs) == 1:
                img = imgs[0].image
                img.load()
                size = _target_size(page)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                if img.size != size:
                    img = img.resize(size, Image.Resampling.LANCZOS)
                return img
        except Exception as e:
            logger.warning(f"Direct image extract failed on page {idx + 1}, rendering instead: {e}")
        images = convert_from_path(
            input_pdf, dpi=dpi, first_page=idx + 1, last_page=idx + 1, thread_count=2,
        )
        if not images:
            return None
        img = images[0]
        img.thumbnail((TARGET, TARGET), Image.Resampling.LANCZOS)
        return img

    draw_lock = threading.Lock()

    def _process(idx):
        img = _load_page(idx)
        if img is None:
            return None
        cfg = page_cfgs[idx]
        if cfg:
            with draw_lock:  # FreeType faces are shared; draw one page at a time
                img = apply_texts_to_image(img, cfg, actual_name=actual_name)
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        try:
            # q95 + 4:4:4 chroma: visually lossless (PSNR ~47 dB), ~2.4x smaller than q100
            img.save(buf, "PDF", resolution=dpi, quality=95, subsampling=0)
        finally:
            img.close()
        return buf.getvalue()

    workers = max(1, min(4, os.cpu_count() or 2))
    logger.info(f"Adding text: {total_pages} pages, {workers} threads, DPI={dpi}")
    writer = PdfWriter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, pdf_bytes in enumerate(pool.map(_process, range(total_pages))):
            if pdf_bytes is None:
                continue
            writer.append(PdfReader(BytesIO(pdf_bytes)))
    gc.collect()

    with open(output_pdf, "wb") as f:
        writer.write(f)

    logger.info(f"Successfully created: {output_pdf}")
