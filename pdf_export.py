# https://www.reportlab.com/opensource/

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def export_to_pdf(text: str, output_path: str, title: str | None = None) -> str:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    if title:
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        safe_block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_block, styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    return str(Path(output_path).resolve())