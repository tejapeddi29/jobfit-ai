"""
Resume Generator module — produces a professional .docx resume file
from the TailoredResume Pydantic model using docx-js via Node.js subprocess.

Falls back to a clean .txt file if Node.js or docx is unavailable.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

from src.models import TailoredResume


def generate_resume_docx(tailored: TailoredResume, output_path: str) -> tuple[str, str | None]:
    """
    Generate a professional .docx resume file from a TailoredResume object.

    Args:
        tailored: The TailoredResume Pydantic model.
        output_path: Where to write the .docx file.

    Returns:
        tuple: (file_path, error_message). error_message is None on success.
    """
    # Serialize the tailored resume data for the Node.js script
    data = tailored.model_dump()

    # Write data to a temp JSON file for the Node script to read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    # Node.js script that builds the .docx
    node_script = _build_node_script(json_path, output_path)

    script_path = tempfile.mktemp(suffix=".js")
    with open(script_path, "w") as f:
        f.write(node_script)

    try:
        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Fall back to text file
            txt_path = output_path.replace(".docx", ".txt")
            _generate_text_fallback(tailored, txt_path)
            return txt_path, f"DOCX generation failed (Node.js error), saved as .txt instead."

        return output_path, None

    except FileNotFoundError:
        # Node.js not installed — fall back to text
        txt_path = output_path.replace(".docx", ".txt")
        _generate_text_fallback(tailored, txt_path)
        return txt_path, "Node.js not found. Resume saved as .txt — install Node.js for .docx output."

    except subprocess.TimeoutExpired:
        txt_path = output_path.replace(".docx", ".txt")
        _generate_text_fallback(tailored, txt_path)
        return txt_path, "DOCX generation timed out, saved as .txt instead."

    finally:
        # Clean up temp files
        for p in [json_path, script_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def _build_node_script(json_path: str, output_path: str) -> str:
    """Build the Node.js script that creates the .docx file using docx-js."""
    return f"""
const {{ Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
         BorderStyle, LevelFormat, TabStopType, TabStopPosition }} = require('docx');
const fs = require('fs');

const data = JSON.parse(fs.readFileSync('{json_path}', 'utf-8'));

// Build document sections
const children = [];

// ── Name / Title Header ────────────────────────────────────────
children.push(new Paragraph({{
    alignment: AlignmentType.CENTER,
    spacing: {{ after: 60 }},
    children: [new TextRun({{ text: "CANDIDATE NAME", bold: true, size: 28, font: "Arial" }})]
}}));
children.push(new Paragraph({{
    alignment: AlignmentType.CENTER,
    spacing: {{ after: 200 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "2E75B6", space: 4 }} }},
    children: [new TextRun({{ text: `Target: ${{data.target_title}}`, size: 20, color: "666666", font: "Arial" }})]
}}));

// ── Professional Summary ───────────────────────────────────────
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 200, after: 100 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
    children: [new TextRun({{ text: "PROFESSIONAL SUMMARY", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
}}));
children.push(new Paragraph({{
    spacing: {{ after: 200 }},
    children: [new TextRun({{ text: data.summary, size: 20, font: "Arial" }})]
}}));

// ── Skills ─────────────────────────────────────────────────────
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 200, after: 100 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
    children: [new TextRun({{ text: "TECHNICAL SKILLS", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
}}));
children.push(new Paragraph({{
    spacing: {{ after: 200 }},
    children: [new TextRun({{ text: data.skills.join("  \\u2022  "), size: 20, font: "Arial" }})]
}}));

// ── Experience ─────────────────────────────────────────────────
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 200, after: 100 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
    children: [new TextRun({{ text: "EXPERIENCE", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
}}));

for (const exp of data.experience) {{
    children.push(new Paragraph({{
        spacing: {{ before: 120, after: 40 }},
        children: [
            new TextRun({{ text: exp.title, bold: true, size: 21, font: "Arial" }}),
            new TextRun({{ text: `  |  ${{exp.company}}  |  ${{exp.dates}}`, size: 20, font: "Arial", color: "555555" }})
        ]
    }}));
    for (const bullet of exp.bullets) {{
        children.push(new Paragraph({{
            spacing: {{ after: 40 }},
            indent: {{ left: 360 }},
            children: [
                new TextRun({{ text: "\\u2022  ", size: 20, font: "Arial" }}),
                new TextRun({{ text: bullet.rewritten, size: 20, font: "Arial" }})
            ]
        }}));
    }}
}}

// ── Projects ───────────────────────────────────────────────────
if (data.projects && data.projects.length > 0) {{
    children.push(new Paragraph({{
        heading: HeadingLevel.HEADING_1,
        spacing: {{ before: 200, after: 100 }},
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
        children: [new TextRun({{ text: "PROJECTS", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
    }}));
    for (const proj of data.projects) {{
        children.push(new Paragraph({{
            spacing: {{ after: 40 }},
            indent: {{ left: 360 }},
            children: [
                new TextRun({{ text: "\\u2022  ", size: 20, font: "Arial" }}),
                new TextRun({{ text: proj, size: 20, font: "Arial" }})
            ]
        }}));
    }}
}}

// ── Education ──────────────────────────────────────────────────
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 200, after: 100 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
    children: [new TextRun({{ text: "EDUCATION", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
}}));
for (const edu of data.education) {{
    children.push(new Paragraph({{
        spacing: {{ after: 40 }},
        children: [new TextRun({{ text: edu, size: 20, font: "Arial" }})]
    }}));
}}

// ── Certifications ─────────────────────────────────────────────
if (data.certifications && data.certifications.length > 0) {{
    children.push(new Paragraph({{
        heading: HeadingLevel.HEADING_1,
        spacing: {{ before: 200, after: 100 }},
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 2 }} }},
        children: [new TextRun({{ text: "CERTIFICATIONS", bold: true, size: 22, font: "Arial", color: "2E75B6" }})]
    }}));
    for (const cert of data.certifications) {{
        children.push(new Paragraph({{
            spacing: {{ after: 40 }},
            children: [new TextRun({{ text: cert, size: 20, font: "Arial" }})]
        }}));
    }}
}}

// ── Build Document ─────────────────────────────────────────────
const doc = new Document({{
    styles: {{
        default: {{ document: {{ run: {{ font: "Arial", size: 20 }} }} }},
        paragraphStyles: [
            {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
               run: {{ size: 22, bold: true, font: "Arial", color: "2E75B6" }},
               paragraph: {{ spacing: {{ before: 200, after: 100 }}, outlineLevel: 0 }} }}
        ]
    }},
    sections: [{{
        properties: {{
            page: {{
                size: {{ width: 12240, height: 15840 }},
                margin: {{ top: 1080, right: 1080, bottom: 1080, left: 1080 }}
            }}
        }},
        children: children
    }}]
}});

Packer.toBuffer(doc).then(buffer => {{
    fs.writeFileSync('{output_path}', buffer);
    process.exit(0);
}}).catch(err => {{
    console.error(err);
    process.exit(1);
}});
"""


def _generate_text_fallback(tailored: TailoredResume, output_path: str):
    """Generate a clean .txt resume as fallback when .docx generation fails."""
    lines = []

    lines.append("=" * 60)
    lines.append("CANDIDATE NAME")
    lines.append(f"Target: {tailored.target_title}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("PROFESSIONAL SUMMARY")
    lines.append("-" * 40)
    lines.append(tailored.summary)
    lines.append("")

    lines.append("TECHNICAL SKILLS")
    lines.append("-" * 40)
    lines.append("  •  ".join(tailored.skills))
    lines.append("")

    lines.append("EXPERIENCE")
    lines.append("-" * 40)
    for exp in tailored.experience:
        lines.append(f"{exp.title}  |  {exp.company}  |  {exp.dates}")
        for bullet in exp.bullets:
            lines.append(f"  • {bullet.rewritten}")
        lines.append("")

    if tailored.projects:
        lines.append("PROJECTS")
        lines.append("-" * 40)
        for proj in tailored.projects:
            lines.append(f"  • {proj}")
        lines.append("")

    lines.append("EDUCATION")
    lines.append("-" * 40)
    for edu in tailored.education:
        lines.append(edu)
    lines.append("")

    if tailored.certifications:
        lines.append("CERTIFICATIONS")
        lines.append("-" * 40)
        for cert in tailored.certifications:
            lines.append(cert)
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_markdown_resume(tailored: TailoredResume) -> str:
    """Generate a Markdown version of the tailored resume for in-app preview."""
    lines = []

    lines.append(f"# Target: {tailored.target_title}")
    lines.append("")
    lines.append("## Professional Summary")
    lines.append(tailored.summary)
    lines.append("")

    lines.append("## Technical Skills")
    lines.append("  •  ".join(tailored.skills))
    lines.append("")

    lines.append("## Experience")
    for exp in tailored.experience:
        lines.append(f"**{exp.title}** | {exp.company} | {exp.dates}")
        for bullet in exp.bullets:
            lines.append(f"- {bullet.rewritten}")
        lines.append("")

    if tailored.projects:
        lines.append("## Projects")
        for proj in tailored.projects:
            lines.append(f"- {proj}")
        lines.append("")

    lines.append("## Education")
    for edu in tailored.education:
        lines.append(f"- {edu}")
    lines.append("")

    if tailored.certifications:
        lines.append("## Certifications")
        for cert in tailored.certifications:
            lines.append(f"- {cert}")

    return "\n".join(lines)


def generate_resume_pdf(tailored: TailoredResume, output_path: str) -> tuple[str, str | None]:
    """Generate a PDF version of the tailored resume using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return "", "fpdf2 is not installed."
        
    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # Helper to sanitize text for Latin-1 standard fonts
    def s(text: str) -> str:
        if not text: return ""
        text = text.replace("•", "-").replace("–", "-").replace("—", "-")
        text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
        return text.encode('latin-1', 'ignore').decode('latin-1')

    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Name & Target
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, s("CANDIDATE NAME"), ln=True, align="C")
        
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, s(f"Target: {tailored.target_title}"), ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_draw_color(46, 117, 182)
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)
        
        def section_header(title):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(46, 117, 182)
            pdf.cell(0, 8, s(title), ln=True)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(3)
            pdf.set_text_color(0, 0, 0)

        # Summary
        section_header("PROFESSIONAL SUMMARY")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, s(tailored.summary))
        pdf.ln(5)
        
        # Skills
        section_header("TECHNICAL SKILLS")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, s("  |  ".join(tailored.skills)))
        pdf.ln(5)
        
        # Experience
        section_header("EXPERIENCE")
        for exp in tailored.experience:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 6, s(exp.title), ln=True)
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(85, 85, 85)
            pdf.cell(0, 6, s(f"{exp.company}  |  {exp.dates}"), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            for bullet in exp.bullets:
                x = pdf.get_x()
                y = pdf.get_y()
                pdf.set_xy(x + 5, y)
                pdf.multi_cell(0, 6, s(f"- {bullet.rewritten}"))
            pdf.ln(3)
            
        # Projects
        if tailored.projects:
            section_header("PROJECTS")
            pdf.set_font("Helvetica", "", 11)
            for proj in tailored.projects:
                x = pdf.get_x()
                y = pdf.get_y()
                pdf.set_xy(x + 5, y)
                pdf.multi_cell(0, 6, s(f"- {proj}"))
            pdf.ln(3)
            
        # Education
        if tailored.education:
            section_header("EDUCATION")
            pdf.set_font("Helvetica", "", 11)
            for edu in tailored.education:
                pdf.multi_cell(0, 6, s(edu))
            pdf.ln(3)
            
        # Certifications
        if tailored.certifications:
            section_header("CERTIFICATIONS")
            pdf.set_font("Helvetica", "", 11)
            for cert in tailored.certifications:
                pdf.multi_cell(0, 6, s(cert))
            pdf.ln(3)
            
        pdf.output(output_path)
        return output_path, None
    except Exception as e:
        return "", f"PDF generation failed: {str(e)}"
