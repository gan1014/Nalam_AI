
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys

# ── Resolve project root & add to path ───────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.dirname(_HERE)
_ENV_PATH   = os.path.join(_PROJ_ROOT, '.env')

if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from backend import db

# ── Tamil lookup tables ───────────────────────────────────────────────────────
TAMIL_DISTRICTS = {
    'Chennai': 'சென்னை', 'Coimbatore': 'கோயம்புத்தூர்', 'Madurai': 'மதுரை',
    'Tiruchirappalli': 'திருச்சிராப்பள்ளி', 'Salem': 'சேலம்',
    'Tirunelveli': 'திருநெல்வேலி', 'Erode': 'ஈரோடு', 'Vellore': 'வேலூர்',
    'Thoothukudi': 'தூத்துக்குடி', 'Thanjavur': 'தஞ்சாவூர்',
    'Dindigul': 'திண்டுக்கல்', 'Virudhunagar': 'விருதுநகர்',
    'Cuddalore': 'கடலூர்', 'Kanyakumari': 'கன்னியாகுமரி',
    'Tiruppur': 'திருப்பூர்',
}

TAMIL_DISEASES = {
    'Dengue': 'டெங்கு', 'Cholera': 'காலரா',
    'Leptospirosis': 'எலிக்காய்ச்சல்', 'Malaria': 'மலேரியா',
    'Chikungunya': 'சிக்கன்குனியா',
}

# ─────────────────────────────────────────────────────────────────────────────
def _load_credentials():
    """
    Always load credentials fresh from the .env file.
    Returns (gmail_user, gmail_app_password) as stripped strings.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.abspath(_ENV_PATH), override=True)
    except ImportError:
        pass  # python-dotenv not installed; rely on system env vars

    user = os.getenv("GMAIL_USER", "").strip()
    pwd  = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    return user, pwd


def send_alert(district, disease, risk_level, notes="", triggered_by="Government Authorized User", resource_data=None, verification_proof=None):
    """
    Sends clinical alert only if valid face verification proof is provided.
    verification_proof = {
        'face_verified': bool,
        'admin_id': int,
        'timestamp': datetime string (IST),
        'alert_id': str (UUID)
    }
    """
    print(f"🔔 send_alert triggered for {district} | Verification: {bool(verification_proof)}")

    # 1. Mandatory Face Verification Gate
    if not verification_proof or not verification_proof.get('face_verified'):
        print("❌ BLOCK: Attempted dispatch without face verification.")
        db.log_event("SYSTEM", "SEC-GATE", "N/A", "ALERT_BLOCK", "Security", district, "Bypass attempt: Missing face verification")
        return False
    
    # 2. Freshness Check (< 60 seconds)
    try:
        from datetime import datetime
        proof_time = datetime.fromisoformat(verification_proof['timestamp'])
        # Handle timezone-aware vs naive if needed, but assuming IST as per app standard
        now = datetime.now()
        age_seconds = (now - proof_time).total_seconds()
        
        if age_seconds > 60:
            print(f"❌ BLOCK: Face verification expired ({age_seconds:.1f}s ago).")
            db.log_event("SYSTEM", "SEC-GATE", "N/A", "ALERT_BLOCK", "Security", district, f"Verification expired: {age_seconds:.1f}s")
            return False
    except Exception as e:
        print(f"❌ BLOCK: Invalid verification timestamp: {e}")
        return False

    # 3. Data Prep
    alert_id = verification_proof.get('alert_id', 'Unknown')
    admin_id = verification_proof.get('admin_id', '0')
    ist_ts   = verification_proof.get('timestamp', 'N/A')

    # Government Standard Recommendations
    recs = [
        "Deploy vector control teams immediately (பூச்சி கட்டுப்பாட்டுக் குழு)",
        "Check water stagnation in affected areas (நீர் தேக்கம் சரிபார்)",
        "Alert local hospitals for potential admissions (மருத்துவமனை எச்சரிக்கை)"
    ]
    recs_str = "|".join(recs)

    # Reload credentials
    gmail_user, gmail_pwd = _load_credentials()

    # Log to DB (Audit Trail)
    db.log_alert(district, disease, risk_level, "Processing", triggered_by, recs_str)

    if not gmail_user or not gmail_pwd:
        print("⚠️ Gmail credentials missing — alert logged to DB only.")
        db.log_alert(district, disease, risk_level, "Logged Only", triggered_by, recs_str)
        return False

    # 4. Build Email with Verification Metadata
    tamil_dist    = TAMIL_DISTRICTS.get(district, district)
    tamil_disease = TAMIL_DISEASES.get(disease, disease)
    risk_color    = {"HIGH": "#dc2626", "MEDIUM": "#f97316", "LOW": "#22c55e", "CRITICAL": "#b91c1c"}.get(risk_level, "#dc2626")
    risk_emoji    = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢", "CRITICAL": "☣️"}.get(risk_level, "🔴")

    resource_html = ""
    if resource_data:
        resource_html = f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-bottom:20px;">
          <h4 style="margin:0 0 15px; color:#1e293b; font-size:1rem;">🛡️ Resource Planning Projection</h4>
          <table style="width:100%; border-collapse:collapse;">
            <tr>
              <td style="padding:10px; border-bottom:1px solid #f1f5f9;">
                <div style="font-size:0.75rem; color:#64748b;">Field Teams</div>
                <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">{resource_data.get('teams', 0)} Units</div>
              </td>
              <td style="padding:10px; border-bottom:1px solid #f1f5f9;">
                <div style="font-size:0.75rem; color:#64748b;">Hospital Beds</div>
                <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">{resource_data.get('beds', 0)} Assigned</div>
              </td>
              <td style="padding:10px; border-bottom:1px solid #f1f5f9;">
                <div style="font-size:0.75rem; color:#64748b;">Spray Capacity</div>
                <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">{resource_data.get('spray', 0)} Ltrs/Day</div>
              </td>
            </tr>
          </table>
        </div>
        """

    verification_html = f"""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:15px; margin-bottom:20px;">
        <div style="color:#166534; font-weight:700; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">🔐 Biometric Verification Token</div>
        <table style="width:100%; margin-top:10px; font-size:0.8rem; color:#166534;">
            <tr><td><b>Alert ID:</b> {alert_id}</td><td><b>Admin ID:</b> {admin_id}</td></tr>
            <tr><td><b>Status:</b> <span style="background:#16a34a; color:white; padding:2px 6px; border-radius:4px;">VERIFIED</span></td><td><b>Timestamp:</b> {ist_ts}</td></tr>
        </table>
    </div>
    """

    html_body = f"""
    <html>
    <body style="font-family:'Segoe UI',Arial,sans-serif;background:#f4f4f4;margin:0;padding:0;">
      <div style="max-width:620px;margin:30px auto;background:#fff;border-radius:12px; overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.1);">
        <div style="background:{risk_color};padding:28px 32px;text-align:center;">
          <div style="font-size:2.5rem;">{risk_emoji}</div>
          <h1 style="color:#fff;margin:8px 0;font-size:1.5rem;letter-spacing:1px;">{risk_level} RISK ALERT</h1>
          <p style="color:rgba(255,255,255,.85);margin:0;font-size:.9rem;">நலம் AI Clinical Surveillance System</p>
        </div>
        <div style="padding:28px 32px;">
          {verification_html}
          <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
            <tr>
              <td style="padding:10px;background:#f8f9fa;border-radius:8px;width:50%;">
                <div style="font-size:.7rem;color:#6b7280;text-transform:uppercase;">District</div>
                <div style="font-size:1.1rem;font-weight:700;">{district}</div>
                <div style="font-size:.8rem;color:#6b7280;">{tamil_dist}</div>
              </td>
              <td style="padding:10px;width:10%;"></td>
              <td style="padding:10px;background:#f8f9fa;border-radius:8px;width:50%;">
                <div style="font-size:.7rem;color:#6b7280;text-transform:uppercase;">Disease</div>
                <div style="font-size:1.1rem;font-weight:700;">{disease}</div>
                <div style="font-size:.8rem;color:#6b7280;">{tamil_disease}</div>
              </td>
            </tr>
          </table>
          {resource_html}
          {f'<div style="background:#f0f9ff;border-left:4px solid #0ea5e9;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:20px;"><div style="font-size:.7rem;color:#6b7280;text-transform:uppercase;">Clinical Notes</div><p style="margin:0;">{notes}</p></div>' if notes else ''}
          <div style="background:#fff7ed;border-left:4px solid {risk_color}; padding:14px 16px;border-radius:0 8px 8px 0;margin-bottom:20px;">
            <h4 style="margin:0 0 10px;color:#92400e;">🚀 Recommended Actions</h4>
            <ul style="margin:0;padding-left:18px;color:#374151;line-height:1.8;">
              <li>Deploy vector control teams immediately</li>
              <li>Check water stagnation in affected areas</li>
              <li>Alert local hospitals for potential admissions</li>
            </ul>
          </div>
          <p style="font-size:.8rem;color:#9ca3af;">Generated by Nalam AI · Tamil Nadu Health Department</p>
        </div>
        <div style="background:#f9fafb;padding:16px 32px;text-align:center; border-top:1px solid #e5e7eb;">
          <p style="margin:0;font-size:.75rem;color:#9ca3af;">🩺 Nalam AI · Automated Outbreak Alert System</p>
        </div>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart('alternative')
    msg['From']    = gmail_user
    msg['To']      = gmail_user
    msg['Subject'] = f"🚨 Nalam AI Verified Alert: {risk_level} — {district} | {disease}"
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        server.ehlo(); server.starttls(); server.ehlo()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(gmail_user, gmail_user, msg.as_string())
        server.quit()

        print(f"✅ Verified alert sent successfully for {district}")
        db.log_alert(district, disease, risk_level, "Sent", triggered_by, recs_str)
        return True
    except Exception as e:
        print(f"❌ SMTP Error: {e}")
        db.log_alert(district, disease, risk_level, "Failed", triggered_by, str(e))
        return False


if __name__ == "__main__":
    # Quick test
    result = send_alert("Chennai", "Dengue", "HIGH", "Test alert from CLI")
    print("Result:", result)
