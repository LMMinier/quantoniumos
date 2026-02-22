# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Convert Markdown to PDF using fpdf2."""

import sys
import re
from fpdf import FPDF


class MarkdownPDF(FPDF):
    def header(self):
        pass
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def clean_line(line):
    """Remove markdown formatting and normalize Unicode."""
    # Remove bold/italic
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    line = re.sub(r'\*(.+?)\*', r'\1', line)
    # Remove inline code
    line = re.sub(r'`(.+?)`', r'\1', line)
    # Remove links
    line = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', line)
    # Replace Unicode characters with ASCII equivalents
    replacements = {
        '\u2014': '-',  # em dash
        '\u2013': '-',  # en dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2026': '...',  # ellipsis
        '\u2192': '->',  # right arrow
        '\u2190': '<-',  # left arrow
        '\u2194': '<->',  # left-right arrow
        '\u2022': '-',  # bullet
        '\u2713': '[x]',  # check mark
        '\u2717': '[ ]',  # x mark
        '\u03c6': 'phi',  # phi
        '\u03a6': 'Phi',  # Phi
        '\u03c3': 'sigma',  # sigma
        '\u03b2': 'beta',  # beta
        '\u03b8': 'theta',  # theta
        '\u03b1': 'alpha',  # alpha
        '\u2248': '~=',  # approximately
        '\u2260': '!=',  # not equal
        '\u2264': '<=',  # less than or equal
        '\u2265': '>=',  # greater than or equal
        '\u221e': 'inf',  # infinity
        '\u03c0': 'pi',  # pi
        '\u2211': 'Sum',  # summation
        '\u220f': 'Prod',  # product
        '\u221a': 'sqrt',  # square root
        '\u2208': 'in',  # element of
        '\u2209': 'not in',  # not element of
        '\u2229': 'AND',  # intersection
        '\u222a': 'OR',  # union
        '\u2282': 'subset',  # subset
        '\u2286': 'subseteq',  # subset or equal
        '\u2295': '+',  # circled plus
        '\u2297': 'x',  # circled times
        '\u22c5': '*',  # dot operator
        '\u00d7': 'x',  # multiplication sign
        '\u00f7': '/',  # division sign
        '\u00b1': '+/-',  # plus-minus
        '\u00b2': '^2',  # superscript 2
        '\u00b3': '^3',  # superscript 3
        '\u2070': '^0',  # superscript 0
        '\u00b9': '^1',  # superscript 1
        '\u2074': '^4',  # superscript 4
        '\u2075': '^5',  # superscript 5
        '\u2076': '^6',  # superscript 6
        '\u2077': '^7',  # superscript 7
        '\u2078': '^8',  # superscript 8
        '\u2079': '^9',  # superscript 9
        '\u2081': '_1',  # subscript 1
        '\u2082': '_2',  # subscript 2
        '\u2083': '_3',  # subscript 3
        '\u2084': '_4',  # subscript 4
        '\u2085': '_5',  # subscript 5
        '\u207b': '-',  # superscript minus
        '\u2080': '_0',  # subscript 0
        '\u25b6': '>',  # play button
        '\u25bc': 'v',  # down arrow
        '\u2502': '|',  # box drawing vertical
        '\u2500': '-',  # box drawing horizontal
        '\u250c': '+',  # box drawing corner
        '\u2510': '+',  # box drawing corner
        '\u2514': '+',  # box drawing corner
        '\u2518': '+',  # box drawing corner
        '\u251c': '+',  # box drawing T
        '\u2524': '+',  # box drawing T
        '\u252c': '+',  # box drawing T
        '\u2534': '+',  # box drawing T
        '\u253c': '+',  # box drawing cross
        '\u2551': '|',  # double vertical
        '\u2550': '=',  # double horizontal
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    # Final fallback: replace any remaining non-ASCII
    line = line.encode('ascii', 'replace').decode('ascii')
    return line


def convert_md_to_pdf(md_path, pdf_path):
    """Convert markdown file to PDF."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean entire content first
    content = clean_line(content)
    
    pdf = MarkdownPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    in_code_block = False
    code_lines = []
    
    for line in content.split('\n'):
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                pdf.set_font('Courier', '', 8)
                for code_line in code_lines:
                    pdf.multi_cell(0, 4, code_line)
                pdf.ln(3)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue
        
        # Headers
        if line.startswith('# '):
            pdf.set_font('Helvetica', 'B', 18)
            pdf.multi_cell(0, 10, clean_line(line[2:]))
            pdf.ln(5)
        elif line.startswith('## '):
            pdf.set_font('Helvetica', 'B', 14)
            pdf.multi_cell(0, 8, clean_line(line[3:]))
            pdf.ln(3)
        elif line.startswith('### '):
            pdf.set_font('Helvetica', 'B', 12)
            pdf.multi_cell(0, 7, clean_line(line[4:]))
            pdf.ln(2)
        elif line.startswith('#### '):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.multi_cell(0, 6, clean_line(line[5:]))
            pdf.ln(2)
        elif line.startswith('---'):
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        elif line.startswith('|'):
            # Table row
            pdf.set_font('Courier', '', 9)
            pdf.multi_cell(0, 5, clean_line(line))
        elif line.strip().startswith('- '):
            # Bullet point
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(5, 5, '')
            pdf.multi_cell(0, 5, '* ' + clean_line(line.strip()[2:]))
        elif line.strip().startswith('$$'):
            # Math block marker - skip
            continue
        elif line.strip().startswith('$'):
            # Inline math - just print as-is
            pdf.set_font('Courier', '', 10)
            pdf.multi_cell(0, 5, line)
        elif line.strip():
            # Regular paragraph
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(0, 5, clean_line(line))
        else:
            # Empty line
            pdf.ln(2)
    
    pdf.output(pdf_path)
    print(f'PDF saved: {pdf_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python md_to_pdf.py <input.md> [output.pdf]')
        sys.exit(1)
    
    md_path = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_path = sys.argv[2]
    else:
        pdf_path = md_path.replace('.md', '.pdf')
    
    convert_md_to_pdf(md_path, pdf_path)
