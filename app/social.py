import json
import logging
import os
import re
import subprocess
from pathlib import Path

from .schemas import Project, SocialPlatform
from .transcription import gemini_chat

log = logging.getLogger(__name__)


def generate_social_posts(project: Project) -> dict[SocialPlatform, str]:
    if not project.transcript:
        return {}

    system = """Sən peşəkar bir SMM və kopiraytersən. Sənə verilən mətnə əsasən 3 fərqli platforma üçün paylaşım mətni (caption) hazırlamalısan.
Tələblər:
- Instagram: Peşəkar, amma cəlbedici. ÇOX CÜZİ emoji (maksimum 1-2 ədəd). Bir neçə uyğun hashtag.
- YouTube: Qısa, axtarışa uyğun (SEO) və izləyicini videonu bəyənməyə səsləyən.
- LinkedIn: Çox rəsmi, öyrədici, oxşar peşəkarlara xitab edən. Heç bir emoji istifadə etmə.
Format olaraq yalnız və yalnız düzgün JSON qaytar. Heç bir markdown formatting (```json) istifadə etmə.
Açar sözlər: "instagram", "youtube", "linkedin".
Dəyərlər: Hazırladığın mətnlər."""

    user = f"Aşağıdakı videonun mətninə əsasən postları hazırla:\n\n{project.transcript.text}"

    response = gemini_chat(
        system=system,
        user=user,
        what="social posts generation",
        temperature=0.7,
        max_tokens=2048,
    )

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        log.error("Could not parse JSON from social generation: %s", response)
        return {}

    try:
        data = json.loads(match.group(0))
        return {
            SocialPlatform.instagram: data.get("instagram", ""),
            SocialPlatform.youtube: data.get("youtube", ""),
            SocialPlatform.linkedin: data.get("linkedin", ""),
        }
    except json.JSONDecodeError:
        log.error("JSON decode error from social generation: %s", response)
        return {}


def generate_cover_image(project: Project, title: str, source_path: Path, output_dir: Path) -> str | None:
    """
    Extracts a frame from the middle of the source video and overlays the Canva transparent template + title.
    Returns the filename of the generated cover.
    """
    if not project.source_file:
        return None
        
    # Clean up old covers to avoid accumulating files
    for old_cover in output_dir.glob(f"cover_{project.source_file.split('.')[0]}*.png"):
        old_cover.unlink(missing_ok=True)
        
    import time
    cover_name = f"cover_{project.source_file.split('.')[0]}_{int(time.time())}.png"
    out_path = output_dir / cover_name

    target_time = project.media.duration / 2 if project.media else 0

    # Extract the frame
    cmd = [
        "ffmpeg", "-y", "-ss", str(target_time),
        "-i", str(source_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg failed to extract cover frame: %s", e.stderr.decode())
        return None

def generate_cover_image(project: Project, title: str, source_path: Path, output_dir: Path) -> str | None:
    """
    Extracts a frame from the middle of the source video and overlays the Canva transparent template + title.
    Returns the filename of the generated cover.
    """
    if not project.source_file:
        return None
        
    cover_name = "cover_" + project.source_file.split('.')[0] + ".png"
    out_path = output_dir / cover_name

    target_time = project.media.duration / 2 if project.media else 0

    # Extract the frame
    cmd = [
        "ffmpeg", "-y", "-ss", str(target_time),
        "-i", str(source_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg failed to extract cover frame: %s", e.stderr.decode())
        return None

    # Try to add text overlay using Pillow to mimic the Canva design exactly
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        import textwrap
        
        img = Image.open(out_path)
        
        # 1. Resize and crop video frame to Reel dimensions (1080x1920)
        target_size = (1080, 1920)
        img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS).convert('RGBA')
        
        # 2. Overlay the transparent Canva template (contains gradient and brush stroke)
        template_path = Path(__file__).parent / "assets" / "cover_template.png"
        if template_path.exists():
            template = Image.open(template_path).convert('RGBA')
            # The template might be a different size (e.g. 576x1024), resize to 1080x1920
            template = template.resize(target_size, Image.Resampling.LANCZOS)
            img = Image.alpha_composite(img, template)
            
        img = img.convert('RGB')
        
        # 3. Load Montserrat ExtraBold
        font_path = Path(__file__).parent / "fonts" / "Montserrat-ExtraBold.ttf"
        font_size = 75
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except OSError:
            font = ImageFont.load_default()
            
        draw = ImageDraw.Draw(img)
        
        # 4. Wrap text properly to fit 1080px width
        max_width = 1080 - 120  # 60px padding on each side
        
        # Process words (handle *yellow* text)
        words = title.split()
        lines = []
        current_line = []
        
        def get_word_display_text(w):
            return w.replace('*', '')

        for word in words:
            current_line.append(word)
            test_line = " ".join([get_word_display_text(w) for w in current_line])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            lw = bbox[2] - bbox[0]
            if lw > max_width:
                current_line.pop()
                if current_line:
                    lines.append(current_line)
                current_line = [word]
        if current_line:
            lines.append(current_line)
            
        # 5. Draw text
        line_height = int(font_size * 1.3)
        total_text_height = len(lines) * line_height
        
        # Bottom padding 550px (adjusted so brush stroke acts as an underline)
        y_text = 1920 - 550 - total_text_height
        
        for line_words in lines:
            # Measure total width of this line
            display_line = " ".join([get_word_display_text(w) for w in line_words])
            bbox = draw.textbbox((0, 0), display_line, font=font)
            lw = bbox[2] - bbox[0]
            
            x_text = (1080 - lw) / 2
            
            for word in line_words:
                display_word = get_word_display_text(word)
                is_yellow = word.startswith('*') and word.endswith('*')
                text_color = (252, 219, 3) if is_yellow else (255, 255, 255)
                
                # Draw shadow
                shadow_offset = 5
                draw.text((x_text + shadow_offset, y_text + shadow_offset), display_word, font=font, fill=(0, 0, 0, 180))
                
                # Draw main text
                draw.text((x_text, y_text), display_word, font=font, fill=text_color)
                
                # Advance x
                word_bbox = draw.textbbox((0, 0), display_word + " ", font=font)
                x_text += word_bbox[2] - word_bbox[0]
                
            y_text += line_height

        # 6. Save final image
        img.save(out_path, quality=95)
    except Exception as e:
        log.warning("Could not apply Pillow overlay for cover: %s", e)

    return cover_name
