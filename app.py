import streamlit as st
from openai import OpenAI
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import base64
import os

# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()

# ---------------------------
# Utility: Fonts and Styling
# ---------------------------

def find_bold_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\impact.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

BOLD_FONT_PATH = find_bold_font()

STYLE_PRESETS = {
    "Minimal": {
        "stroke": 4,
        "glow": False,
        "shadow": True,
        "pad_ratio": 0.06,
    },
    "Bold": {
        "stroke": 6,
        "glow": True,
        "shadow": True,
        "pad_ratio": 0.08,
    },
    "Neon": {
        "stroke": 5,
        "glow": True,
        "shadow": False,
        "pad_ratio": 0.07,
    },
    "Clean": {
        "stroke": 3,
        "glow": False,
        "shadow": True,
        "pad_ratio": 0.06,
    },
}

# ---------------------------
# OpenAI Helpers
# ---------------------------

def ai_shorten_title(long_title: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following video title into a short, punchy 3–6 word phrase suitable for a YouTube thumbnail. "
                        "Keep the original meaning. Avoid quotation marks and punctuation-heavy output. "
                        f"Title: {long_title}\n\nOutput only the phrase."
                    ),
                },
            ],
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        st.warning(f"Title shortener failed: {e}")
        return long_title

def ai_background_concept(topic: str, style_name: str, accent_hex: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Create a vivid 1–2 sentence visual concept for a YouTube thumbnail background based on this topic and style. "
                        "Do NOT include any readable text within the scene. Focus on subjects, setting, lighting, color palette, mood, and composition. "
                        "Avoid typographic instructions. Keep it concise but descriptive.\n\n"
                        f"Topic: {topic}\nStyle: {style_name}\nAccent color: {accent_hex}\n\n"
                        "Output only the visual concept."
                    ),
                },
            ],
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Concept generation failed: {e}")
        return f"A cinematic, high-contrast background that matches the topic: {topic}. No readable text in the scene."

def generate_background_image(concept_prompt: str, seed: int | None = None) -> Image.Image:
    # Request a square image and crop to 16:9 later
    prompt = (
        "Design a detailed, cinematic background for a YouTube thumbnail. "
        "Do not render any readable text or letters. "
        "Strong focal point, depth, dramatic lighting, high contrast, clean composition. "
        f"Concept: {concept_prompt}"
    )
    extra = f" Seed: {seed}" if seed is not None else ""
    try:
        with st.spinner("Generating background image..."):
            result = client.images.generate(
                model="gpt-image-1",
                prompt=prompt + extra,
                size="1024x1024",
            )
        b64 = result.data[0].b64_json
        img_bytes = base64.b64decode(b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        return img
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        raise

# ---------------------------
# Image Composition
# ---------------------------

def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    w, h = img.size
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 1e-3:
        return img
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)

def add_vignette(img: Image.Image, intensity: float = 0.35) -> Image.Image:
    # Soft vignette for readability
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    # Elliptical gradient: darker edges
    margin = int(min(w, h) * 0.08)
    draw.ellipse([margin, margin, w - margin, h - margin], fill=int(255 * (1 - intensity)))
    blur = overlay.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.12)))
    vignette_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vignette_rgba.putalpha(blur)
    # Multiply-like effect by compositing with alpha
    out = Image.alpha_composite(img, vignette_rgba)
    return out

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    if lv == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    return (255, 255, 255)

def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if BOLD_FONT_PATH:
        try:
            return ImageFont.truetype(BOLD_FONT_PATH, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int = 3):
    words = text.split()
    lines = []
    current = []
    for word in words:
        test_line = (" ".join(current + [word])).strip()
        w = draw.textlength(test_line, font=font)
        if w <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines - 1:
                # Force remainder into last line
                break
    if current:
        lines.append(" ".join(current + words[len(" ".join(lines).split()):]))
    # If exceeded max lines, merge tail into last line
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines[:max_lines]

def fit_text(draw: ImageDraw.ImageDraw, text: str, img_w: int, img_h: int, max_lines: int, pad_ratio: float):
    # Try sizes descending until fit
    max_text_width = int(img_w * (1 - 2 * pad_ratio))
    base_size = int(img_h * 0.18)
    for size in range(base_size, 22, -2):
        font = get_font(size)
        lines = wrap_text_to_width(draw, text, font, max_text_width, max_lines=max_lines)
        spacing = int(size * 0.18)
        # Measure
        line_heights = []
        max_line_w = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font, stroke_width=0)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_heights.append(h)
            max_line_w = max(max_line_w, w)
        total_h = sum(line_heights) + spacing * (len(lines) - 1)
        if max_line_w <= max_text_width and total_h <= int(img_h * (1 - 2 * pad_ratio)):
            return font, lines, spacing
    # Fallback minimal size
    size = 24
    font = get_font(size)
    lines = wrap_text_to_width(draw, text, font, max_text_width, max_lines=max_lines)
    spacing = int(size * 0.18)
    return font, lines, spacing

def draw_text_with_effects(
    img: Image.Image,
    title_text: str,
    accent_hex: str = "#00E5FF",
    style_name: str = "Bold",
    anchor: str = "center",
):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["Bold"])
    pad_ratio = preset.get("pad_ratio", 0.07)
    max_lines = 3
    font, lines, spacing = fit_text(draw, title_text, w, h, max_lines=max_lines, pad_ratio=pad_ratio)
    text_color = (255, 255, 255)
    accent = hex_to_rgb(accent_hex)
    stroke = preset.get("stroke", 4)

    # Compute text block bbox
    line_sizes = [draw.textbbox((0, 0), ln, font=font, stroke_width=stroke) for ln in lines]
    line_heights = [(bbox[3] - bbox[1]) for bbox in line_sizes]
    max_line_w = max([(bbox[2] - bbox[0]) for bbox in line_sizes]) if lines else 0
    total_h = sum(line_heights) + spacing * (len(lines) - 1)

    # Position near center
    x = w // 2
    y = h // 2

    # Optional dark panel for readability
    pad_x = int(max_line_w * 0.06)
    pad_y = int(total_h * 0.20)
    panel_left = x - max_line_w // 2 - pad_x
    panel_top = y - total_h // 2 - pad_y
    panel_right = x + max_line_w // 2 + pad_x
    panel_bottom = y + total_h // 2 + pad_y

    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(
        [panel_left, panel_top, panel_right, panel_bottom],
        radius=int(min(w, h) * 0.02),
        fill=(0, 0, 0, 120),
    )
    panel = panel.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.005)))
    img.alpha_composite(panel)

    # Draw shadow/glow layers
    if STYLE_PRESETS[style_name].get("glow", False):
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        offset_range = [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 0)]
        for i, ln in enumerate(lines):
            line_y = y - total_h // 2 + sum(line_heights[:i]) + (i * spacing)
            for ox, oy in offset_range:
                glow_draw.text((x + ox, line_y + oy), ln, font=font, fill=accent + (255,), anchor="mm", stroke_width=0)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.02)))
        img.alpha_composite(glow)

    # Draw main text with stroke
    for i, ln in enumerate(lines):
        line_y = y - total_h // 2 + sum(line_heights[:i]) + (i * spacing)
        draw.text(
            (x, line_y),
            ln,
            font=font,
            fill=text_color,
            anchor="mm",
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )

    # Accent underline or bar
    underline_h = max(4, int(h * 0.008))
    bar_w = int(max_line_w * 0.9)
    bar_left = x - bar_w // 2
    bar_right = x + bar_w // 2
    bar_top = y + total_h // 2 + int(h * 0.02)
    bar_bottom = bar_top + underline_h
    draw.rounded_rectangle([bar_left, bar_top, bar_right, bar_bottom], radius=underline_h // 2, fill=accent)

    return img

def paste_logo(img: Image.Image, logo_bytes: bytes):
    try:
        logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")
        w, h = img.size
        target_w = int(w * 0.13)
        scale = target_w / logo.width
        target_h = int(logo.height * scale)
        logo = logo.resize((target_w, target_h), Image.LANCZOS)

        # Add a subtle background circle
        pad = int(target_w * 0.15)
        bg = Image.new("RGBA", (logo.width + pad * 2, logo.height + pad * 2), (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.rounded_rectangle([0, 0, bg.width, bg.height], radius=int(bg.height * 0.25), fill=(0, 0, 0, 120))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.005)))
        bg.paste(logo, (pad, pad), logo)

        # Position top-right with margin
        margin = int(w * 0.02)
        pos = (w - bg.width - margin, margin)
        img.alpha_composite(bg, dest=pos)
        return img
    except Exception:
        return img

# ---------------------------
# Streamlit App
# ---------------------------

def build_concept_prompt(user_text: str, style: str, accent_hex: str, use_ai_concept: bool) -> str:
    if use_ai_concept:
        return ai_background_concept(user_text, style, accent_hex)
    # Handcrafted fallback
    return (
        f"A clean, high-contrast background for a YouTube thumbnail about: {user_text}. "
        f"Use a color palette that harmonizes with accent color {accent_hex}. "
        "Avoid any readable text or letters. Strong lighting, shallow depth of field, bold composition, professional look."
    )

def compose_thumbnail(user_text: str, style: str, accent_hex: str, logo_bytes: bytes | None, use_ai_concept: bool, seed: int | None):
    # 1) Background
    concept_prompt = build_concept_prompt(user_text, style, accent_hex, use_ai_concept)
    bg_img = generate_background_image(concept_prompt, seed=seed)

    # 2) Crop to 16:9 and resize to 1280x720
    bg_img = center_crop_to_aspect(bg_img, 16 / 9)
    bg_img = bg_img.resize((1280, 720), Image.LANCZOS)

    # 3) Vignette for readability
    bg_img = add_vignette(bg_img, intensity=0.35)

    # 4) Text overlay
    thumb = draw_text_with_effects(bg_img, user_text, accent_hex=accent_hex, style_name=style)

    # 5) Optional logo
    if logo_bytes:
        thumb = paste_logo(thumb, logo_bytes)

    return thumb

def image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()

def main():
    st.set_page_config(page_title="YouTube Thumbnail Generator", page_icon="🎨", layout="wide")
    st.title("🎨 AI YouTube Thumbnail Generator")
    st.write("Enter your video topic or title, pick a style, and generate a crisp, clickable YouTube thumbnail.")

    # Sidebar Controls
    with st.sidebar:
        st.header("Settings")
        style = st.selectbox("Style preset", list(STYLE_PRESETS.keys()), index=1)
        accent = st.color_picker("Accent color", "#22D3EE")
        use_ai_title = st.checkbox("Auto-shorten title with AI", value=True)
        use_ai_concept = st.checkbox("Auto-generate background concept with AI", value=True)
        seed_input = st.text_input("Seed (optional, for reproducibility)", value="")
        seed = None
        if seed_input.strip().isdigit():
            seed = int(seed_input.strip())
        logo = st.file_uploader("Optional logo (PNG with transparency preferred)", type=["png", "jpg", "jpeg", "webp"])

        st.markdown("---")
        st.caption("This app uses the OpenAI API. Set OPENAI_API_KEY in your environment before running.")

    # Main input
    default_title = "10 Python Tips You Need to Know"
    user_text = st.text_input("Video Title or Topic", value=default_title, help="This will be the text rendered on the thumbnail.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Thumbnail", type="primary"):
            if not user_text.strip():
                st.warning("Please enter a title or topic.")
                st.stop()

            final_title = user_text.strip()
            if use_ai_title:
                final_title = ai_shorten_title(final_title) or user_text.strip()

            try:
                thumbnail = compose_thumbnail(
                    user_text=final_title,
                    style=style,
                    accent_hex=accent,
                    logo_bytes=(logo.read() if logo else None),
                    use_ai_concept=use_ai_concept,
                    seed=seed,
                )
                st.success("Thumbnail generated!")
                st.image(thumbnail, caption="Generated Thumbnail (1280x720)", use_column_width=True)

                img_bytes = image_to_bytes(thumbnail, fmt="PNG")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="thumbnail.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"Failed to generate thumbnail: {e}")

    with col2:
        st.subheader("Tips for better results")
        st.markdown(
            "- Keep titles short and punchy (3–6 words works best).\n"
            "- Use numbers or power words to increase clickability.\n"
            "- Pick a strong accent color for contrast.\n"
            "- Consider uploading a logo for brand consistency.\n"
            "- Re-generate with different seeds for variety."
        )

if __name__ == "__main__":
    main()