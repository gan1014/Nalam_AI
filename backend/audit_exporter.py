from fpdf import FPDF
import pandas as pd
from datetime import datetime

class AuditPDF(FPDF):
    def header(self):
        # Government Header
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'GOVERNMENT OF TAMIL NADU', 0, 1, 'C')
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, 'HEALTH & FAMILY WELFARE DEPARTMENT', 0, 1, 'C')
        self.set_font('helvetica', 'B', 11)
        self.cell(0, 10, 'NALAM AI — DISTRICT SURVEILLANCE AUDIT TRAIL', 0, 1, 'C')
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} · Nalam AI Automated Audit Log · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

def generate_audit_pdf(df, filename, date_range=None):
    pdf = AuditPDF()
    pdf.alias_nb_pages()
    pdf.add_page(orientation='L') # Landscape for many columns
    
    # Selection Metadata
    pdf.set_font('helvetica', '', 10)
    if date_range:
        pdf.cell(0, 10, f'Audit Period: {date_range}', 0, 1)
    pdf.cell(0, 10, f'Total Records: {len(df)}', 0, 1)
    pdf.ln(5)

    # Table Header
    pdf.set_font('helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    cols = ['Timestamp', 'Username', 'Role', 'Session', 'Action', 'Target', 'Details']
    col_widths = [35, 25, 20, 15, 45, 30, 105] # Total ~275 for Landscape A4

    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 10, col, 1, 0, 'C', True)
    pdf.ln()

    # Table Rows
    pdf.set_font('helvetica', '', 8)
    for _, row in df.iterrows():
        # Truncate strings if too long
        ts = str(row['timestamp'])
        user = str(row['username'])[:15]
        role = str(row['role'])[:12]
        sess = str(row['session_id'])[:8]
        act = str(row['action'])[:25]
        tgt = f"{row['target_type']}:{row['target_id']}"[:15]
        det = str(row['details'])[:60]

        pdf.cell(col_widths[0], 8, ts, 1)
        pdf.cell(col_widths[1], 8, user, 1)
        pdf.cell(col_widths[2], 8, role, 1)
        pdf.cell(col_widths[3], 8, sess, 1)
        pdf.cell(col_widths[4], 8, act, 1)
        pdf.cell(col_widths[5], 8, tgt, 1)
        pdf.cell(col_widths[6], 8, det, 1)
        pdf.ln()

    pdf.output(filename)
    return filename
