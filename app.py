from flask import Flask, request, send_from_directory, render_template, redirect, url_for, render_template_string
from io import BytesIO
import zipfile
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import numpy as np
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from models import db, User
import os, io, shutil
from functools import wraps
import tempfile
from sqlalchemy import create_engine
import random
from sqlalchemy.sql import text
from flask import make_response
import smtplib
import requests
from email.mime.text import MIMEText
from flask import abort
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, abort
from email.mime.multipart import MIMEMultipart
from flask_sqlalchemy import SQLAlchemy
from configparser import ConfigParser
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from models import db, User  # ✅ this now works because User is defined in models.py
import os
import shutil
from flask import Flask, render_template, request, send_file, jsonify,session
from math import ceil
from sqlalchemy import text
import html
from functools import wraps
from flask import session, redirect, url_for, flash
from zoneinfo import ZoneInfo
from datetime import datetime, date
from flask import request
import re, pickle 
from sqlalchemy import text, inspect

IST = ZoneInfo("Asia/Kolkata")  # <-- Add this
def now_ist():
    return datetime.now(IST) 


app = Flask(__name__)
session_data = {}  # ✅ Add this line
TEMP_FOLDER = os.path.join(tempfile.gettempdir(), 'ranking_outputs')
#TEMP_FOLDER = tempfile.gettempdir()
os.makedirs(TEMP_FOLDER, exist_ok=True)


# DB Config
DB_USER = 'appsadmin'
DB_PASS = 'appsadmin2025'
DB_HOST = '34.93.75.171'
DB_PORT = '3306'
DB_NAME = 'elicita'
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

app.secret_key = 'vasanth'

# MySQL Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://appsadmin:appsadmin2025@34.93.75.171:3306/elicita'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
SMTP_SERVER = "smtp.datasolve-analytics.com"
SMTP_PORT = 587
WEBMAIL_USER = "apps.admin@datasolve-analytics.com"
WEBMAIL_PASSWORD = "datasolve@2025"


SMTP_USER = WEBMAIL_USER
SMTP_PASS = WEBMAIL_PASSWORD
EXTRA_ALERT_EMAILS = [
     "megha.m@datasolve-analytics.com",
]

# Initialize db
db.init_app(app)

# ── PROFILE MODEL (lives in another schema: mainapp) ─────────────
class UserProfile(db.Model):
    __tablename__  = "User_Profiles"
    __table_args__ = {"extend_existing": True, "schema": "mainapp"}  # <- important
    # If the table has no PK, Email_ID is a safe choice
    Email_ID  = db.Column(db.String(255), primary_key=True)
    Image_URL = db.Column(db.Text)
    Designation  = db.Column(db.String(200))
    Team         = db.Column(db.String(100))


def get_mapped_results_date_col(engine) -> str:
    """
    Returns the DATE column name in Mapped_Results.
    Prefers 'dispatch_date', falls back to 'despatched_date', default 'dispatch_date'.
    """
    sql = text("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
          AND TABLE_NAME = 'Mapped_Results'
          AND COLUMN_NAME IN ('dispatch_date','despatched_date')
        ORDER BY FIELD(COLUMN_NAME,'dispatch_date','despatched_date')
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"db": DB_NAME}).fetchone()
    return (row[0] if row else 'dispatch_date')



def _ajax_wants_json() -> bool:
    """
    True if the client likely expects JSON (AJAX/fetch).
    Uses Accept and X-Requested-With headers.
    """
    accept = (request.headers.get("Accept") or "").lower()
    xhr    = (request.headers.get("X-Requested-With") or "").lower()
    return ("json" in accept) or ("xmlhttprequest" in xhr)

def _all_blank(series) -> bool:
    """
    Return True if a pandas Series is effectively empty
    (NaN or only whitespace in all cells).
    """
    try:
        return (series.isna() | (series.astype(str).str.strip() == '')).all()
    except Exception:
        return False
    
def log_row(*, username, email, abstract, upload_filename, downloadfile_name,
            action, start_dt, end_dt, status, project_code,
            initial_rows=None, final_rows=None):
    start_time  = start_dt.strftime('%H:%M:%S')
    end_time    = end_dt.strftime('%H:%M:%S')
    runtime_hms = str(end_dt - start_dt).split('.')[0]
    start_date  = start_dt.strftime('%Y-%m-%d')

    data = {
        'name'             : username,
        'email'            : email,
        'abstract'         : abstract,
        'upload_filename'  : upload_filename,
        'downloadfile_name': downloadfile_name,
        'action'           : action,
        'start_time'       : start_time,
        'end_time'         : end_time,
        'status'           : status,
        'results'          : None,
        'project_code'     : project_code,
        'execution_time'   : runtime_hms,
        "date"             : start_date,
    }
    if initial_rows is not None:
        data['initial_rows'] = int(initial_rows)
    if final_rows is not None:
        data['final_rows'] = int(final_rows)

    try:
        pd.DataFrame([data]).to_sql('prioriq_log', con=engine, if_exists='append', index=False)
    except Exception as e:
        print(f"[LOGGING ERROR] Could not write to prioriq_log: {e}")


def _safe_code(code: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', str(code or ''))

def _stash_paths(code: str):
    safe = _safe_code(code)
    base = os.path.join(TEMP_FOLDER, f"{safe}__to_push")
    return base + ".parquet", base + ".pkl"

def save_df_stash(df: pd.DataFrame, code: str) -> str:
    pq, pk = _stash_paths(code)
    try:
        df.to_parquet(pq, index=False)   # needs pyarrow or fastparquet (recommended)
        return pq
    except Exception:
        with open(pk, "wb") as f:
            pickle.dump(df, f)           # fallback if parquet not available
        return pk

def load_df_stash(code: str) -> pd.DataFrame | None:
    pq, pk = _stash_paths(code)
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if os.path.exists(pk):
        try:
            with open(pk, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

def cleanup_stash(code: str):
    for p in _stash_paths(code):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _date_where_args():
    """
    Reads ?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    Returns (extra_sql, params_dict, sd, ed)

    If dispatch_date is DATETIME in DB: keep DATE(dispatch_date)
    If it's DATE, you may change to just 'dispatch_date' in the strings below.
    """
    sd = _parse_date((request.args.get("start_date") or "").strip())
    ed = _parse_date((request.args.get("end_date") or "").strip())

    where, args = [], {}
    if sd:
        where.append("DATE(dispatch_date) >= :sd")
        args["sd"] = sd
    if ed:
        where.append("DATE(dispatch_date) <= :ed")
        args["ed"] = ed

    extra = (" AND " + " AND ".join(where)) if where else ""
    return extra, args, sd, ed


def _date_filter_from_args():
    """
    Reads ?start_date=YYYY-MM-DD & ?end_date=YYYY-MM-DD from the query string.
    Returns (sql_snippet, params_dict). If neither is valid → ("", {}).
    """
    sd_raw = (request.args.get("start_date") or "").strip()
    ed_raw = (request.args.get("end_date") or "").strip()

    def _valid_ymd(s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except Exception:
            return False

    sd = sd_raw if sd_raw and _valid_ymd(sd_raw) else None
    ed = ed_raw if ed_raw and _valid_ymd(ed_raw) else None

    if sd and ed:
        return "AND DATE(dispatch_date) BETWEEN :sd AND :ed", {"sd": sd, "ed": ed}
    if sd:
        return "AND DATE(dispatch_date) >= :sd", {"sd": sd}
    if ed:
        return "AND DATE(dispatch_date) <= :ed", {"ed": ed}
    return "", {}

# Prevent browser caching
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in first.")
            return redirect('/sign-in')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return redirect('/sign-in')



@app.route('/otpverification', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        otp = request.form['otp']

        if 'verification_code' in session and str(session['verification_code']) == otp:
            email = session.get('email')
            user = db.session.query(User).filter_by(email=email).first()
            if user:
                user.verified = True
                user.verification_code = None  # Clear the OTP after successful verification
                db.session.commit()

                session.pop('email', None)
                session.pop('verification_code', None)

                return render_template('login.html', success="Your account has been verified successfully!")
            return render_template('verify.html', error="User not found or verification error.")

    return render_template('verify.html')

@app.route('/sign-up', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # ✅ Only allow emails from @datasolve-analytics.com
        allowed_domain = "datasolve-analytics.com"
        if not email.endswith(f"@{allowed_domain}"):
            return render_template('register.html', error="Only datasolve-analytics.com emails are allowed.")

        hashed_password = generate_password_hash(password)
        verification_code = random.randint(100000, 999999)

        # Check if username or email already exists
        existing_user = db.session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            return render_template('register.html', error="Username or email already exists.")

        # Create new user
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            verification_code=verification_code,
            verified=False
        )

        db.session.add(new_user)
        db.session.commit()

        # Send OTP for verification
        send_otp_email(email, verification_code)

        # Store verification details in session
        session['email'] = email
        session['verification_code'] = verification_code

        return redirect(url_for('verify'))

    return render_template('register.html', error="Only datasolve-analytics.com emails are allowed.")

                           
def send_otp_email(email, otp):
    try:
        otp_str = str(otp)
        subject = "Email Verification OTP"
        plain_text = f"Your OTP is: {otp_str}"
        html_content = f"""
        <html>
            <body>
                <h1>Email Verification</h1>
                <p>Your OTP is: <strong>{otp_str}</strong></p>
            </body>
        </html>
        """
        msg = MIMEMultipart("alternative")
        msg["From"] = f"From PriorIQ<{WEBMAIL_USER}>"
        msg["To"] = email
        msg["Subject"] = subject
        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(WEBMAIL_USER, WEBMAIL_PASSWORD)
            server.sendmail(WEBMAIL_USER, email, msg.as_string())

    except Exception as error:
        print("Error sending OTP email:", error)

### ── HELPERS image ───────────────────────────────────────────────────────────────
import hashlib
def get_profile_for_email(email: str):
    """Return (role_from_profile, team_from_profile, image_url) for an email."""
    if not email:
        return None, None, None
    rec = (db.session.query(UserProfile.Designation, UserProfile.Team, UserProfile.Image_URL)
           .filter(UserProfile.Email_ID == email)
           .first())
    if not rec:
        return None, None, None
    return rec[0], rec[1], rec[2]
def gravatar_url(email: str, size=64, default="identicon"):
    if not email:
        return ""
    h = hashlib.md5(email.strip().lower().encode("utf-8")).hexdigest()
    return f"https://www.gravatar.com/avatar/{h}?s={size}&d={default}&r=g"
@app.context_processor
def inject_gravatar():
    return dict(gravatar_url=gravatar_url)
@app.context_processor
def inject_profile_image():
    """
    Make user_email + profile_image_url available in ALL templates.
    Uses session['email'] first; falls back to DB lookup by username.
    Also fetches Image_URL from mainapp.User_Profiles for the avatar.
    """
    img_url = None
    display_name = session.get("username")
    email = session.get("email")
    try:
        # fallback: get email from DB if not in session
        if not email and display_name:
            u = User.query.filter_by(username=display_name).first()
            email = u.email if u else None
        # lookup profile image by email
        if email:
            rec = (db.session.query(UserProfile.Image_URL)
                   .filter(UserProfile.Email_ID == email)
                   .first())
            if rec and rec[0]:
                img_url = rec[0]
    except Exception as e:
        app.logger.exception("Profile inject failed: %s", e)
    return {
        "user_email": email,            # :point_left: now available everywhere
        "profile_image_url": img_url,
        "profile_name": display_name,
    }

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        role = session.get("role")
        if role not in ("admin", "superadmin"):
            flash("⛔ Access denied! Admins only.")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


@app.route('/sign-in', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['password']

        user = User.query.filter_by(email=email, verified=True).first()
        if user and check_password_hash(user.password, pwd):
            session['username'] = user.username
            session['role'] = user.role
            session['user_name'] = user.username
            session['user_email'] = user.email

            if user.role in ('admin', 'superadmin'):
                return redirect(url_for('logs'))  # Admin -> Logs Page
            return redirect('/welcome')             # Normal users -> File Page

        flash("❌ Invalid credentials or not verified.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/sign-in')



@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = db.session.query(User).filter_by(email=email, verified=True).first()

        if user:
            reset_code = random.randint(100000, 999999)
            user.verification_code = reset_code
            db.session.commit()

            send_otp_email(email, reset_code)
            session['reset_email'] = email

            flash("An OTP has been sent to your email to reset your password.", "info")
            return redirect(url_for('reset_password'))
        else:
            return render_template('forgot_password.html', error="No verified account found with this email.")

    return render_template('forgot_password.html')

        
# Reset Password Route
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        otp = request.form['otp']
        new_password = request.form['new_password']

        if 'reset_email' in session:
            email = session['reset_email']
            user = db.session.query(User).filter_by(email=email).first()

            if user and str(user.verification_code) == otp:
                user.password = generate_password_hash(new_password)
                user.verification_code = None
                db.session.commit()

                session.pop('reset_email', None)
                flash("Your password has been reset. Please log in.", "success")
                return redirect(url_for('login'))
            else:
                return render_template('reset_password.html', error="Invalid OTP or email.")
    
    return render_template('reset_password.html')

@app.route('/welcome')
def landing():
    return render_template("landing.html")

#_____________________________________________________________________________________________________________

def get_admin_emails():
    """
    Get all verified admin emails from elicita.rankifylogin.
    """
    try:
        sql = """
            SELECT email
            FROM elicita.rankifylogin
            WHERE Role = 'admin' AND verified = 1
        """
        df = pd.read_sql(sql, con=engine)
        emails = (
            df["email"]
            .dropna()
            .map(lambda x: x.strip())
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        if not emails:
            print("[AdminNotify] No admin emails found in rankifylogin.")
        return emails
    except Exception as e:
        print(f"[AdminNotify] Failed to fetch admin emails: {e}")
        return []


def get_extra_notify_emails():
    """
    Returns cleaned list of extra alert emails from EXTRA_ALERT_EMAILS.
    """
    try:
        emails = [
            e.strip()
            for e in EXTRA_ALERT_EMAILS
            if isinstance(e, str) and e.strip() != ""
        ]
        return list(set(emails))
    except Exception as e:
        print(f"[AdminNotify] Failed to get extra notify emails: {e}")
        return []


def send_admin_email(subject: str, body: str, extra_emails=None):
    """
    Send email to:
      - all admins (Role='admin', verified=1)
      - plus any extra_emails passed
      - plus global EXTRA_ALERT_EMAILS
    Best-effort only (won't crash routes).
    """
    admin_emails = get_admin_emails()
    global_extra = get_extra_notify_emails()
    passed_extra = extra_emails or []

    # Combine all recipients and de-duplicate
    recipients = set(admin_emails)
    for e in global_extra + passed_extra:
        if isinstance(e, str) and e.strip():
            recipients.add(e.strip())

    recipients = list(recipients)

    if not recipients:
        print("[AdminNotify] No recipients found (admins + extra).")
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT)) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipients, msg.as_string())

        print(f"[AdminNotify] Mail sent to: {recipients}")
    except Exception as e:
        print(f"[AdminNotify] Error sending admin email: {e}")


def notify_admins_initial(tool_label: str,
                          user_name: str,
                          user_email: str,
                          project_code: str,
                          upload_filename: str,
                          extra_emails=None):
    """
    Notify admins + extra emails when an initial tool run happens.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"[PriorIQ] Initial tool run – {tool_label}"

    body = f"""
Hi Admin,

User        : {user_name or '-'} ({user_email or '-'})
Tool        : {tool_label} (Initial Dataset)
Project Code: {project_code or '-'}
Input File  : {upload_filename or '-'}
Date        : {now_str}

Regards,
PriorIQ App (Datasolve Analytics)
""".strip()

    # This will send to:
    # - Admins from DB
    # - EXTRA_ALERT_EMAILS (global)
    # - extra_emails (optional per-call)
    send_admin_email(subject, body, extra_emails=extra_emails)


####################################################################
#               ROUTES
####################################################################

@app.route('/initial-dataset', methods=['GET'])
def page_a():
    return render_template("initial.html")

@app.route('/prioriq', methods=['POST'])
def prioriq_route():
    username = session.get('user_name', 'Anonymous')
    email    = session.get("user_email", "Unknown")

    ExecutionStartTime = datetime.now()
    upload_filename = ""
    project_code = ""
    excel_name = ""
    zip_name = ""
    success = False
    err_msg = ""

    try:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        os.makedirs(TEMP_FOLDER, exist_ok=True)

        if 'initial_file' not in request.files:
            err_msg = "No file part 'initial_file' in request."
            return jsonify({"status": "error", "message": err_msg}), 400

        abstract = (request.form.get('abstract') or "").strip()
        if not abstract:
            err_msg = "Abstract text is required."
            return jsonify({"status": "error", "message": err_msg}), 400

        file = request.files['initial_file']
        if not file or file.filename.strip() == "":
            err_msg = "No file selected."
            return jsonify({"status": "error", "message": err_msg}), 400

        upload_filename = secure_filename(file.filename)
        project_code = (request.form.get('project_code') or "").strip() or upload_filename.split('_')[0].strip()

        try:
            df = pd.read_excel(file)
        except Exception as ex_read:
            err_msg = f"Failed to read Excel: {ex_read}"
            return jsonify({"status": "error", "message": err_msg}), 400

        # Required columns (same as before)
        required_cols = ['Title', 'Abstract', 'Claims', 'Description']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            err_msg = f"Missing columns in Input: {', '.join(missing)}"
            return jsonify({"status": "error", "message": err_msg}), 400

        # Optional new column
        fan_col = 'Questel unique family ID (FAN)'

        # Build combined text for similarity
        df['All'] = df[required_cols].fillna('').agg(' '.join, axis=1)

        model_path = os.path.join(os.getcwd(), 'Static', 'trained_model_fasttriplet_3')
        if not os.path.isdir(model_path):
            err_msg = f"Model folder not found at: {model_path}"
            return jsonify({"status": "error", "message": err_msg}), 500

        model = SentenceTransformer(model_path)
        query_embedding = model.encode(abstract, convert_to_tensor=True)
        doc_embeddings  = model.encode(df['All'].tolist(), convert_to_tensor=True)

        df['Similarity'] = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        df_sorted = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

        # Insert project code at first column
        df_sorted.insert(0, 'Project_code', project_code)

        # --- Reorder so that FAN column (if present) is LAST ---
        if fan_col in df_sorted.columns:
            # take all columns except FAN, then append FAN at the end
            cols_except_fan = [c for c in df_sorted.columns if c != fan_col]
            df_sorted = df_sorted[cols_except_fan + [fan_col]]

        excel_name = f"{project_code}_Initial_PriorIQ_Output.xlsx"
        zip_name   = f"{project_code}_PriorIQ_Output.zip"
        excel_path = os.path.join(TEMP_FOLDER, excel_name)
        zip_path   = os.path.join(TEMP_FOLDER, zip_name)

        df_sorted.to_excel(excel_path, index=False)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(excel_path, arcname=excel_name)
        os.remove(excel_path)

        if not os.path.exists(zip_path):
            err_msg = "ZIP creation failed unexpectedly."
            return jsonify({"status": "error", "message": err_msg}), 500

        success = True

        # 🔔 Notify admins
        try:
            notify_admins_initial(
                tool_label="PriorIQ",
                user_name=username,
                user_email=email,
                project_code=project_code,
                upload_filename=upload_filename,
                extra_emails=None  # or ["someone@datasolve-analytics.com"]
            )
        except Exception as _notify_err:
            print(f"[AdminNotify] PriorIQ notify error: {_notify_err}")

        return jsonify({"status": "success", "output_zip": zip_name})

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return jsonify({"status": "error", "message": err_msg}), 500

    finally:
        ExecutionEndTime = datetime.now()
        start_time  = ExecutionStartTime.strftime('%H:%M:%S')
        end_time    = ExecutionEndTime.strftime('%H:%M:%S')
        runtime_hms = str(ExecutionEndTime - ExecutionStartTime).split('.')[0]
        start_date  = ExecutionStartTime.strftime('%Y-%m-%d')
        status_msg = "Success" if success else (f"Error: {err_msg[:240]}" if err_msg else "Error: Unknown")

        log_data = {
            'name'              : username,
            'email'             : email,
            'abstract'          : locals().get("abstract",""),
            'upload_filename'   : locals().get("upload_filename",""),
            'downloadfile_name' : (locals().get("zip_name","") or locals().get("excel_name","")),
            'action'            : "PriorIQ Initial Run",
            'start_time'        : start_time,
            'end_time'          : end_time,
            'execution_time'    : runtime_hms,
            "date": start_date,
            'status'            : status_msg,
            'results'           : None,
            'project_code'      : locals().get("project_code","")
        }
        try:
            pd.DataFrame([log_data]).to_sql('prioriq_log', con=engine, if_exists='append', index=False)
        except Exception as e:
            print(f"Error logging execution details to MySQL: {e}")


###########################################################################################################################

@app.route('/patentryx', methods=['POST'])
def patentryx_route():
    username = session.get('user_name', 'Anonymous')
    email    = session.get("user_email", "Unknown")

    ExecutionStartTime = datetime.now()
    upload_filename = ""
    project_code = ""
    excel_name = ""
    zip_name = ""
    success = False
    err_msg = ""

    try:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        os.makedirs(TEMP_FOLDER, exist_ok=True)

        if 'initial_file' not in request.files:
            err_msg = "No file part 'initial_file' in request."
            return jsonify({"status": "error", "message": err_msg}), 400

        abstract = (request.form.get('abstract') or "").strip()
        if not abstract:
            err_msg = "Abstract text is required."
            return jsonify({"status": "error", "message": err_msg}), 400

        file = request.files['initial_file']
        if not file or file.filename.strip() == "":
            err_msg = "No file selected."
            return jsonify({"status": "error", "message": err_msg}), 400

        upload_filename = secure_filename(file.filename)
        project_code = (request.form.get('project_code') or "").strip() or upload_filename.split('_')[0].strip()

        try:
            df = pd.read_excel(file)
        except Exception as ex_read:
            err_msg = f"Failed to read Excel: {ex_read}"
            return jsonify({"status": "error", "message": err_msg}), 400

        # Required columns
        required_cols = ['Abstract', 'Claims']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            err_msg = f"Missing columns in Input: {', '.join(missing)}"
            return jsonify({"status": "error", "message": err_msg}), 400

        # Optional input columns
        pub_col = 'Publication numbers with kind code'
        fan_col = 'Questel unique family ID (FAN)'  # NEW COLUMN

        df['Full'] = df['Abstract'].fillna('') + ' ' + df['Claims'].fillna('')
        if len(df) > 0:
            df.at[0, 'Full'] = abstract

        def _clean_text(x):
            words = []
            for w in str(x).split():
                w = re.sub(r'[^a-zA-Z]', '', w).lower()
                if w:
                    words.append(w)
            return " ".join(words)

        df['doc_clean'] = df['Full'].apply(_clean_text)

        tfv = TfidfVectorizer()
        tfv_vectors = tfv.fit_transform(df['doc_clean'])
        similarity = np.dot(tfv_vectors.toarray(), tfv_vectors.toarray().T)

        results = []
        for i in np.argsort(similarity[0])[::-1]:
            if i == 0:
                continue
            score = float(similarity[0][i])
            if score > 0:
                row = df.iloc[i]
                results.append({
                    'Project_code': project_code,
                    'Number': row.get(pub_col, ''),
                    'Score': score,
                    fan_col: row.get(fan_col, '')   # NEW: include FAN column
                })

        df_sorted = pd.DataFrame(results)

        # Final column ordering
        if not df_sorted.empty:
            df_sorted['Patent_No'] = (
                df_sorted['Number']
                .astype(str)
                .str.replace(r'\s+', '', regex=True)
            )

            # FAN must be LAST
            final_order = ['Project_code', 'Number', 'Score', 'Patent_No', fan_col]
            df_sorted = df_sorted[[c for c in final_order if c in df_sorted.columns]]
        else:
            df_sorted = pd.DataFrame(columns=['Project_code', 'Number', 'Score', 'Patent_No', fan_col])

        excel_name = f"{project_code}_Initial_Patentryx_Output.xlsx"
        zip_name   = f"{project_code}_Patentryx_Output.zip"
        excel_path = os.path.join(TEMP_FOLDER, excel_name)
        zip_path   = os.path.join(TEMP_FOLDER, zip_name)

        df_sorted.to_excel(excel_path, index=False)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(excel_path, arcname=excel_name)

        os.remove(excel_path)

        if not os.path.exists(zip_path):
            err_msg = "ZIP creation failed unexpectedly."
            return jsonify({"status": "error", "message": err_msg}), 500

        success = True

        # Admin Notify
        try:
            notify_admins_initial(
                tool_label="Patentryx",
                user_name=username,
                user_email=email,
                project_code=project_code,
                upload_filename=upload_filename,
                extra_emails=None
            )
        except Exception as _notify_err:
            print(f"[AdminNotify] Patentryx notify error: {_notify_err}")

        return jsonify({"status": "success", "output_zip": zip_name})

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return jsonify({"status": "error", "message": err_msg}), 500

    finally:
        ExecutionEndTime = datetime.now()
        start_time  = ExecutionStartTime.strftime('%H:%M:%S')
        end_time    = ExecutionEndTime.strftime('%H:%M:%S')
        runtime_hms = str(ExecutionEndTime - ExecutionStartTime).split('.')[0]
        start_date  = ExecutionStartTime.strftime('%Y-%m-%d')
        status_msg = "Success" if success else (f"Error: {err_msg[:240]}" if err_msg else "Error: Unknown")

        log_data = {
            'name'              : username,
            'email'             : email,
            'abstract'          : locals().get("abstract",""),
            'upload_filename'   : locals().get("upload_filename",""),
            'downloadfile_name' : (locals().get("zip_name","") or locals().get("excel_name","")),
            'action'            : "Patentryx Initial Run",
            'start_time'        : start_time,
            'end_time'          : end_time,
            'execution_time'    : runtime_hms,
            "date": start_date,
            'status'            : status_msg,
            'results'           : None,
            'project_code'      : locals().get("project_code","")
        }
        try:
            pd.DataFrame([log_data]).to_sql('prioriq_log', con=engine, if_exists='append', index=False)
        except Exception as e:
            print(f"Error logging execution details to MySQL: {e}")



######################################################################################################################

@app.route('/bothranking', methods=['POST'])
def bothranking_route():
    username = session.get('user_name', 'Anonymous')
    email    = session.get("user_email", "Unknown")

    ExecutionStartTime = datetime.now()
    upload_filename = ""
    project_code = ""
    zip_name = ""
    success = False
    err_msg = ""

    try:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        os.makedirs(TEMP_FOLDER, exist_ok=True)

        if 'initial_file' not in request.files:
            err_msg = "No file part 'initial_file' in request."
            return jsonify({"status": "error", "message": err_msg}), 400

        abstract = (request.form.get('abstract') or "").strip()
        if not abstract:
            err_msg = "Abstract text is required."
            return jsonify({"status": "error", "message": err_msg}), 400

        file = request.files['initial_file']
        if not file or file.filename.strip() == "":
            err_msg = "No file selected."
            return jsonify({"status": "error", "message": err_msg}), 400

        upload_filename = secure_filename(file.filename)
        project_code = (request.form.get('project_code') or "").strip() or upload_filename.split('_')[0].strip()

        try:
            df = pd.read_excel(file)
        except Exception as ex_read:
            err_msg = f"Failed to read Excel: {ex_read}"
            return jsonify({"status": "error", "message": err_msg}), 400

        # Optional new column (used in both outputs, last column)
        fan_col = 'Questel unique family ID (FAN)'

        # ----------------------------
        # --- Patentryx (TF-IDF)  ---
        # ----------------------------
        if not {'Abstract','Claims'}.issubset(df.columns):
            missing = [c for c in ['Abstract','Claims'] if c not in df.columns]
            err_msg = f"Missing columns in Excel for Patentryx: {', '.join(missing)}"
            return jsonify({"status": "error", "message": err_msg}), 400

        df_pat = df.copy()
        df_pat['Full'] = df_pat['Abstract'].fillna('') + ' ' + df_pat['Claims'].fillna('')
        if len(df_pat) > 0:
            df_pat.at[0, 'Full'] = abstract

        def _clean_doc(x):
            return " ".join(re.sub(r'[^a-zA-Z]', '', w).lower() for w in str(x).split())

        df_pat['doc_clean'] = df_pat['Full'].apply(_clean_doc)
        tfv = TfidfVectorizer()
        tfv_vectors = tfv.fit_transform(df_pat['doc_clean'])
        similarity = np.dot(tfv_vectors.toarray(), tfv_vectors.toarray().T)

        pat_results = []
        pub_col = 'Publication numbers with kind code'
        for i in np.argsort(similarity[0])[::-1]:
            if i == 0:
                continue
            score = float(similarity[0][i])
            if score > 0:
                row = df_pat.iloc[i]
                pat_results.append({
                    'Project_code': project_code,
                    'Number'      : row.get(pub_col, ''),
                    'Score'       : score,
                    fan_col       : row.get(fan_col, '')  # FAN value if column exists
                })

        df_patentryx = pd.DataFrame(pat_results)
        if not df_patentryx.empty:
            df_patentryx['Patent_No'] = df_patentryx['Number'].astype(str).str.replace(r'\s+', '', regex=True)

            # Ensure FAN is last column
            final_pat_cols = ['Project_code', 'Number', 'Score', 'Patent_No', fan_col]
            df_patentryx = df_patentryx[[c for c in final_pat_cols if c in df_patentryx.columns]]
        else:
            df_patentryx = pd.DataFrame(columns=['Project_code','Number','Score','Patent_No', fan_col])

        # -------------------------------
        # --- PriorIQ (SentenceTransformer)
        # -------------------------------
        for col in ['Title','Abstract','Claims','Description']:
            if col not in df.columns:
                err_msg = f"Missing columns in Excel for PriorIQ: {col}"
                return jsonify({"status": "error", "message": err_msg}), 400

        df_prior = df.copy()
        df_prior['All'] = df_prior[['Title','Abstract','Claims','Description']].fillna('').agg(' '.join, axis=1)

        model_path = os.path.join(os.getcwd(), 'Static', 'trained_model_fasttriplet_3')
        if not os.path.isdir(model_path):
            err_msg = f"Model folder not found at: {model_path}"
            return jsonify({"status": "error", "message": err_msg}), 500

        model = SentenceTransformer(model_path)
        query_embedding = model.encode(abstract, convert_to_tensor=True)
        doc_embeddings  = model.encode(df_prior['All'].tolist(), convert_to_tensor=True)
        df_prior['Similarity'] = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

        df_prioriq = df_prior.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        df_prioriq.insert(0, 'Project_code', project_code)

        # Reorder so FAN (if present) is LAST
        if fan_col in df_prioriq.columns:
            prior_cols_except_fan = [c for c in df_prioriq.columns if c != fan_col]
            df_prioriq = df_prioriq[prior_cols_except_fan + [fan_col]]

        # -------------------------
        # Save both and zip them
        # -------------------------
        excel_pat   = f"{project_code}_Patentryx_Output.xlsx"
        excel_prior = f"{project_code}_PriorIQ_Output.xlsx"
        zip_name    = f"{project_code}_Combined_Output.zip"

        path_pat   = os.path.join(TEMP_FOLDER, excel_pat)
        path_prior = os.path.join(TEMP_FOLDER, excel_prior)
        zip_path   = os.path.join(TEMP_FOLDER, zip_name)

        df_patentryx.to_excel(path_pat, index=False)
        df_prioriq.to_excel(path_prior, index=False)

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(path_pat,   arcname=excel_pat)
            zipf.write(path_prior, arcname=excel_prior)

        try:
            os.remove(path_pat)
            os.remove(path_prior)
        except Exception:
            pass

        if not os.path.exists(zip_path):
            err_msg = "ZIP creation failed unexpectedly."
            return jsonify({"status": "error", "message": err_msg}), 500

        success = True

        # 🔔 Notify
        try:
            notify_admins_initial(
                tool_label="PriorIQ + Patentryx (Both Ranking)",
                user_name=username,
                user_email=email,
                project_code=project_code,
                upload_filename=upload_filename,
                extra_emails=None  # or specific list
            )
        except Exception as _notify_err:
            print(f"[AdminNotify] BothRanking notify error: {_notify_err}")

        return jsonify({"status": "success", "output_zip": zip_name})

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return jsonify({"status": "error", "message": err_msg}), 500

    finally:
        ExecutionEndTime = datetime.now()
        start_time  = ExecutionStartTime.strftime('%H:%M:%S')
        end_time    = ExecutionEndTime.strftime('%H:%M:%S')
        runtime_hms = str(ExecutionEndTime - ExecutionStartTime).split('.')[0]
        start_date  = ExecutionStartTime.strftime('%Y-%m-%d')
        status_msg = "Success" if success else (f"Error: {err_msg[:240]}" if err_msg else "Error: Unknown")

        log_data = {
            'name'              : username,
            'email'             : email,
            'abstract'          : locals().get("abstract",""),
            'upload_filename'   : locals().get("upload_filename",""),
            'downloadfile_name' : locals().get("zip_name",""),
            'action'            : "Patentryx & PriorIQ Initial Run",
            'start_time'        : start_time,
            'end_time'          : end_time,
            'execution_time'    : runtime_hms,
            "date": start_date,
            'status'            : status_msg,
            'results'           : None,
            'project_code'      : locals().get("project_code","")
        }
        try:
            pd.DataFrame([log_data]).to_sql('prioriq_log', con=engine, if_exists='append', index=False)
        except Exception as e:
            print(f"Error logging execution details to MySQL: {e}")


#########################################################################################################################

#final
from datetime import datetime, date
from models import User  # make sure this import exists

import os, re, io, zipfile, shutil, tempfile, traceback
from io import BytesIO

import numpy as np
import pandas as pd
from flask import request, render_template, jsonify, session
from sqlalchemy import text, inspect
from werkzeug.utils import secure_filename

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Assumed available in your app:
# app, engine, TEMP_FOLDER, session_data
# _ajax_wants_json(), _all_blank(), log_row(), get_mapped_results_date_col(), save_df_stash()
# send_admin_email(...)  <-- already defined earlier in your app

# (We *use* load_df_stash / cleanup_stash below; if they’re not present in your app,
#  the no-op fallbacks defined here will keep things running.)

# ---- Optional safe fallbacks if not globally available ----
try:
    load_df_stash  # noqa
except NameError:
    def load_df_stash(project_code: str):
        # If your app already provides this, remove this fallback.
        return None

try:
    cleanup_stash  # noqa
except NameError:
    def cleanup_stash(project_code: str):
        # If your app already provides this, remove this fallback.
        pass

# ---------------- Helpers ----------------
def _get_verified_usernames():
    """Return a list of verified usernames, sorted ascending (fallback safe)."""
    try:
        from models import User
        users = User.query.filter_by(verified=True).order_by(User.username.asc()).all()
        return [u.username for u in users if getattr(u, "username", None)]
    except Exception:
        return []

def _user_can_pick_analyst():
    return session.get("role", "user") in ("admin", "superadmin")

def _parse_dispatch_date(s: str) -> str:
    """Accept 'MM/DD/YYYY' or 'YYYY-MM-DD'; return 'YYYY-MM-DD'."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Dispatch date is required (MM/DD/YYYY).")
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    raise ValueError("Invalid date format. Use MM/DD/YYYY.")

# --------------- Route -------------------
@app.route('/final-dataset', methods=['GET', 'POST'])
def page_b():
    # who is logged in (from session)
    session_username = session.get('user_name', 'Anonymous')
    session_email    = session.get("user_email", "Unknown")
    session_role     = session.get("role", "user")

    # ---------- GET: render form ----------
    if request.method == 'GET':
        return render_template(
            "final.html",
            analysts=_get_verified_usernames(),
            selected_analyst=session_username,
            can_pick_analyst=_user_can_pick_analyst(),
            final_table=None,
            download_token=None,
            project_code=""
        )

    # ---------- POST ----------
    upload_filename = ""
    zip_name        = ""
    project_code    = ""

    # read chosen analyst from form (fallback to logged-in user)
    chosen_analyst = (request.form.get('analyst') or session_username).strip() or session_username

    try:
        # FAN column name (optional in input)
        fan_col = 'Questel unique family ID (FAN)'

        # ───────── inputs ─────────
        abstract         = request.form['abstract']
        initial_file     = request.files['initial_file']
        final_file       = request.files['final_file']
        initial_filename = secure_filename(initial_file.filename)
        final_filename   = secure_filename(final_file.filename)
        upload_filename  = f"{initial_filename}, {final_filename}"

        # ───────── dispatch date normalize ─────────
        dispatch_date_in = (request.form.get('dispatch_date') or '').strip()
        try:
            dispatch_date_str = _parse_dispatch_date(dispatch_date_in)
        except ValueError as _e:
            err_msg = f"Dispatch date error: {str(_e)}"
            return (jsonify({"status":"error","message":err_msg}), 400) if _ajax_wants_json() else (err_msg, 400)

        project_code = (request.form.get('project_code') or '').strip() or final_file.filename.split('_')[0]
        zip_name     = f"{project_code}_final_outputs.zip"
        zip_path     = os.path.join(TEMP_FOLDER, zip_name)

        # fresh temp + zip
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            initial_path = os.path.join(temp_dir, initial_file.filename)
            final_path   = os.path.join(temp_dir, final_file.filename)
            initial_file.save(initial_path)
            final_file.save(final_path)

            # ───────── read inputs ─────────
            df_initial = pd.read_excel(initial_path)
            initial_rows = int(df_initial.shape[0])

            # ===== INITIAL validation =====
            _init_has_pub = 'Publication numbers with kind code' in df_initial.columns
            _init_has_pat = 'Patent_No' in df_initial.columns
            if not (_init_has_pub or _init_has_pat):
                err_msg = "Missing columns in Initial: need 'Publication numbers with kind code'"
                return (jsonify({"status":"error","message":err_msg}), 400) if _ajax_wants_json() else (err_msg, 400)

            if _init_has_pub and not _init_has_pat:
                df_initial.rename(columns={'Publication numbers with kind code': 'Patent_No'}, inplace=True)
            df_initial['Patent_No'] = df_initial['Patent_No'].astype(str).str.replace(r'\s+', '', regex=True)

            if _all_blank(df_initial['Patent_No']):
                err_msg = "Initial file error: 'Patent_No' column has no values."
                return (jsonify({"status":"error","message":err_msg}), 400) if _ajax_wants_json() else (err_msg, 400)

            # ───────── final file read & validation ─────────
            df_final = pd.read_excel(final_path)
            final_rows = int(df_final.shape[0])

            required_final = ['Title', 'Abstract', 'Claims', 'Description']
            missing_final = [c for c in required_final if c not in df_final.columns]
            empty_final   = [c for c in required_final if c in df_final.columns and _all_blank(df_final[c])]
            if missing_final or empty_final:
                bits = []
                if missing_final: bits.append("Missing: " + ", ".join(missing_final))
                if empty_final:   bits.append("Empty: "   + ", ".join(empty_final))
                err_msg = "Final file error — " + " | ".join(bits)
                return (jsonify({"status":"error","message":err_msg}), 400) if _ajax_wants_json() else (err_msg, 400)

            # combine (now guaranteed present)
            df_final['All'] = df_final[['Title', 'Abstract', 'Claims', 'Description']].fillna('').agg(' '.join, axis=1)

            # =========================
            # BLOCK 1: Final_New_Ranking (SentenceTransformer)
            # =========================
            block1_start = datetime.now()
            try:
                model           = SentenceTransformer(os.path.join(os.getcwd(), 'Static', 'trained_model_fasttriplet_3'))
                query_embedding = model.encode(abstract, convert_to_tensor=True)
                doc_embeddings  = model.encode(df_final['All'].tolist(), convert_to_tensor=True)
                df_final['Similarity'] = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

                df_new_sorted = (
                    df_final.sort_values(by='Similarity', ascending=False)
                            .reset_index(drop=True)
                            .copy()
                )
                if 'Publication numbers with kind code' in df_new_sorted.columns:
                    df_new_sorted['Patent_No'] = (
                        df_new_sorted['Publication numbers with kind code'].astype(str)
                        .str.replace(r'\s+', '', regex=True)
                    )
                else:
                    df_new_sorted['Patent_No'] = ""

                df_new_sorted.insert(0, 'Project_code', project_code)

                # Ensure FAN (if present) is last column in this output
                if fan_col in df_new_sorted.columns:
                    cols_except_fan = [c for c in df_new_sorted.columns if c != fan_col]
                    df_new_sorted = df_new_sorted[cols_except_fan + [fan_col]]

                with BytesIO() as new_out:
                    with pd.ExcelWriter(new_out, engine='xlsxwriter') as writer:
                        df_new_sorted.to_excel(writer, index=False)
                    new_bytes = new_out.getvalue()
                with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr(f"{project_code}_Final_New_Ranking.xlsx", new_bytes)

                block1_end = datetime.now()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Prion', start_dt=block1_start, end_dt=block1_end,
                    status='Success', project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows,
                )
            except Exception as be1:
                block1_end = datetime.now()
                err_msg = f"Internal Error in Block 1 (Final_New_Ranking): {str(be1)}"
                traceback.print_exc()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Prion', start_dt=block1_start, end_dt=block1_end,
                    status=err_msg, project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows
                )
                return (jsonify({"status":"error","message": err_msg}), 500) if _ajax_wants_json() else (err_msg, 500)

            # =========================
            # BLOCK 2: Final_Old_Ranking (TF-IDF)
            # =========================
            block2_start = datetime.now()
            try:
                df_final['Full'] = df_final['Abstract'].fillna('') + ' ' + df_final['Claims'].fillna('')
                if len(df_final) > 0:
                    df_final.at[0, 'Full'] = abstract

                def _clean_doc(x):
                    return " ".join(re.sub(r'[^a-zA-Z]', '', w).lower() for w in str(x).split())

                tfv_text    = df_final['Full'].apply(_clean_doc)
                tfv         = TfidfVectorizer()
                tfv_vectors = tfv.fit_transform(tfv_text)
                similarity  = np.dot(tfv_vectors.toarray(), tfv_vectors.toarray().T)

                final_old = []
                pub_col   = 'Publication numbers with kind code'
                for i in np.argsort(similarity[0])[::-1]:
                    if i == 0:
                        continue
                    score = similarity[0][i]
                    if score > 0:
                        row = df_final.iloc[i]
                        number_val = row[pub_col] if pub_col in df_final.columns else ""
                        final_old.append({
                            'Project_code': project_code,
                            'Number'      : number_val,
                            'Score'       : score,
                            fan_col       : row.get(fan_col, '')  # FAN from final file (if exists)
                        })

                df_old_sorted = pd.DataFrame(final_old)
                if not df_old_sorted.empty:
                    df_old_sorted['Patent_No'] = df_old_sorted['Number'].astype(str).str.replace(r'\s+', '', regex=True)

                    # Ensure FAN last in this output
                    final_old_cols = ['Project_code', 'Number', 'Score', 'Patent_No', fan_col]
                    df_old_sorted = df_old_sorted[[c for c in final_old_cols if c in df_old_sorted.columns]]
                else:
                    df_old_sorted = pd.DataFrame(columns=['Project_code','Number','Score','Patent_No', fan_col])

                with BytesIO() as old_out:
                    with pd.ExcelWriter(old_out, engine='xlsxwriter') as writer:
                        df_old_sorted.to_excel(writer, index=False)
                    old_bytes = old_out.getvalue()
                with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr(f"{project_code}_Final_Old_Ranking.xlsx", old_bytes)

                block2_end = datetime.now()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Patentryx', start_dt=block2_start, end_dt=block2_end,
                    status='Success', project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows
                )
            except Exception as be2:
                block2_end = datetime.now()
                err_msg = f"Internal Error in Block 2 (Final_Old_Ranking): {str(be2)}"
                traceback.print_exc()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Patentryx', start_dt=block2_start, end_dt=block2_end,
                    status=err_msg, project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows
                )
                return (jsonify({"status":"error","message": err_msg}), 500) if _ajax_wants_json() else (err_msg, 500)

            # =========================
            # BLOCK 3: Final_Rank_Order_Output
            # =========================
            block3_start = datetime.now()
            try:
                pub_col = 'Publication numbers with kind code'

                df_rank_input = pd.read_excel(final_path).copy()
                if 'Patent_No' not in df_rank_input.columns and pub_col in df_rank_input.columns:
                    df_rank_input.rename(columns={pub_col: 'Patent_No'}, inplace=True)
                if 'Patent_No' in df_rank_input.columns:
                    df_rank_input['Patent_No'] = df_rank_input['Patent_No'].astype(str).str.replace(r'\s+', '', regex=True)
                else:
                    df_rank_input['Patent_No'] = ""

                def get_rank(pn, ranked_list):
                    for idx, val in enumerate(ranked_list):
                        if pn and val and (pn in val or val in pn):
                            return idx + 1
                    return None

                def present_in_initial(pn, init_list):
                    for val in init_list:
                        if pn and val and (pn in val or val in pn):
                            return "Yes"
                    return "No"

                df_rank_input['Project_code']   = project_code
                df_rank_input['Old_Rank_Order'] = df_rank_input['Patent_No'].apply(
                    lambda x: get_rank(x, df_old_sorted['Patent_No'].tolist())
                )
                df_rank_input['New_Rank_Order'] = df_rank_input['Patent_No'].apply(
                    lambda x: get_rank(x, df_new_sorted['Patent_No'].tolist())
                )
                df_rank_input['Yes_No'] = df_rank_input['Patent_No'].apply(
                    lambda x: present_in_initial(x, df_initial['Patent_No'].tolist())
                )

                df_rank_input.rename(columns={
                    'Order of the results in report'     : 'Result_Order',
                    'Individual rating of the references': 'Individual_rating'
                }, inplace=True)

                df_rank_input['Version'] = ''

                # base final columns
                base_final_cols = [
                    'Project_code', 'Patent_No', 'Result_Order', 'Individual_rating',
                    'Old_Rank_Order', 'New_Rank_Order', 'Yes_No', 'Version'
                ]
                # add FAN if present in rank_input
                if fan_col in df_rank_input.columns:
                    final_cols = base_final_cols + [fan_col]
                else:
                    final_cols = base_final_cols

                if 'Result_Order' in df_rank_input.columns:
                    df_final_ranked = (
                        df_rank_input[final_cols]
                        .dropna(subset=['Result_Order'])
                        .sort_values(by='Result_Order', ascending=True)
                        .copy()
                    )
                else:
                    df_final_ranked = pd.DataFrame(columns=final_cols)

                for col in ['Result_Order', 'Old_Rank_Order', 'New_Rank_Order']:
                    if col in df_final_ranked.columns:
                        df_final_ranked[col] = pd.to_numeric(df_final_ranked[col], errors='coerce').astype('Int64')

                # attach normalized dispatch date AND Analyst for preview/export
                df_final_ranked['Dispatch_Date'] = dispatch_date_str
                df_final_ranked['Analyst'] = chosen_analyst

                # write preview file
                with BytesIO() as final_out:
                    with pd.ExcelWriter(final_out, engine='xlsxwriter') as writer:
                        df_final_ranked.to_excel(writer, index=False)
                    final_bytes = final_out.getvalue()
                with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.writestr(f"{project_code}_Final_Rank_Order_Output.xlsx", final_bytes)

                # ── Map to DB column names (including date & analyst) ──
                date_col = get_mapped_results_date_col(engine)  # actual DB date column
                db_cols_map = {
                    'Project_code'     : 'project_code',
                    'Patent_No'        : 'patent_no',
                    'Result_Order'     : 'result_order',
                    'Individual_rating': 'individual_rating',
                    'Old_Rank_Order'   : 'old_rank_order',
                    'New_Rank_Order'   : 'new_rank_order',
                    'Yes_No'           : 'yes_no',
                    'Version'          : 'Version',
                    'Dispatch_Date'    : date_col,
                    'Analyst'          : 'analyst',   # ensure analyst goes to DB
                    # fan_col NOT mapped → not pushed to DB
                }
                df_for_db = df_final_ranked.rename(columns=db_cols_map)

                # ---- Inject Analyst (auto-detect existing DB column) ----
                try:
                    insp = inspect(engine)
                    cols_info = insp.get_columns('Mapped_Results')
                    existing_cols = {c['name'] for c in cols_info}
                except Exception:
                    existing_cols = set(df_for_db.columns)

                # If your table uses a different analyst-like column, map it.
                analyst_candidates = ['analyst', 'Analyst', 'pushed_by', 'created_by', 'Created_By', 'username']
                analyst_col = next((c for c in analyst_candidates if c in existing_cols), None)
                if analyst_col:
                    # If DB uses a different column name than 'analyst', move data accordingly
                    if analyst_col != 'analyst' and 'analyst' in df_for_db.columns:
                        df_for_db[analyst_col] = df_for_db['analyst']
                        df_for_db.drop(columns=['analyst'], inplace=True, errors='ignore')
                else:
                    # If table doesn’t have any, keep 'analyst' (to_sql will append the column)
                    if 'analyst' not in df_for_db.columns:
                        df_for_db['analyst'] = chosen_analyst

                # write order for DB
                db_order = [
                    'project_code', 'patent_no', 'result_order', 'individual_rating',
                    'old_rank_order', 'new_rank_order', 'yes_no'
                ]
                # include analyst column (whatever the chosen one is)
                if analyst_col and analyst_col in df_for_db.columns and analyst_col not in db_order:
                    db_order.append(analyst_col)
                elif 'analyst' in df_for_db.columns and 'analyst' not in db_order:
                    db_order.append('analyst')

                db_order.extend(['Version', date_col])

                df_for_db = df_for_db[[c for c in db_order if c in df_for_db.columns]].copy()

                for c in ['result_order', 'old_rank_order', 'new_rank_order']:
                    if c in df_for_db.columns:
                        df_for_db[c] = pd.to_numeric(df_for_db[c], errors='coerce').astype('Int64')

                try:
                    df_for_db[date_col] = pd.to_datetime(df_for_db[date_col], errors='coerce').dt.date
                except Exception:
                    pass

                # stash for /push_to_db
                session_data[project_code] = df_for_db
                save_df_stash(df_for_db, project_code)

                block3_end = datetime.now()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Comparison', start_dt=block3_start, end_dt=block3_end,
                    status='Success', project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows
                )

                # ───────── render ─────────
                return render_template(
                    "final.html",
                    download_token=zip_name,
                    final_table=df_final_ranked.head(20).to_html(index=False),
                    project_code=project_code,
                    analysts=_get_verified_usernames(),
                    selected_analyst=chosen_analyst,
                    can_pick_analyst=_user_can_pick_analyst()
                )

            except Exception as be3:
                block3_end = datetime.now()
                err_msg = f"Internal Error in Block 3 (Final_Rank_Order_Output): {str(be3)}"
                traceback.print_exc()
                log_row(
                    username=chosen_analyst, email=session_email, abstract=abstract,
                    upload_filename=upload_filename, downloadfile_name=zip_name,
                    action='Final Comparison', start_dt=block3_start, end_dt=block3_end,
                    status=err_msg, project_code=project_code,
                    initial_rows=initial_rows, final_rows=final_rows
                )
                return (jsonify({"status":"error","message": err_msg}), 500) if _ajax_wants_json() else (err_msg, 500)

    except Exception as e:
        traceback.print_exc()
        err_msg = f"Internal Error: {str(e)}"
        return (jsonify({"status":"error","message": err_msg}), 500) if _ajax_wants_json() else (err_msg, 500)

@app.route("/push_to_db", methods=["POST"])
def push_to_db():
    # session context
    session_username = session.get('user_name', 'Anonymous')
    session_email    = session.get("user_email", "Unknown")

    wants_json = "application/json" in (request.headers.get("Accept","").lower())
    start_dt = datetime.now()
    results = 0

    # analyst from form (fallback to logged-in user)
    chosen_analyst = (request.form.get("analyst") or session_username).strip() or session_username

    try:
        project_code = (request.form.get("project_code") or "").strip()
        if not project_code:
            msg = "⚠️ Project code is required."
            return (jsonify({"status":"error","message": msg}), 400) if wants_json else (msg, 400)

        # Attempt memory stash first
        df = session_data.get(project_code)

        # Fallback: disk stash
        if df is None or getattr(df, "empty", True):
            df = load_df_stash(project_code)

        if df is None or getattr(df, "empty", True):
            msg = "⚠️ No data available to push for this project code."
            return (jsonify({"status":"error","message": msg}), 400) if wants_json else (msg, 400)

        # Ensure we have a valid date column mapped to the DB's date col
        date_col = get_mapped_results_date_col(engine)  # e.g., 'dispatch_date' / 'date'
        if date_col not in df.columns:
            df[date_col] = pd.Timestamp('today').date()
        else:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
                if df[date_col].isna().all():
                    df[date_col] = pd.Timestamp('today').date()
            except Exception:
                df[date_col] = pd.Timestamp('today').date()

        # Determine next Version (max 3)
        chk = text("""
            SELECT COUNT(DISTINCT Version) AS version_count
            FROM Mapped_Results
            WHERE Project_code = :project_code
        """)
        with engine.connect() as conn:
            row = conn.execute(chk, {"project_code": project_code}).fetchone()
            version_count = (row[0] if row else 0) or 0

        if version_count >= 3:
            msg = "⚠️ Maximum of 3 versions have already been pushed for this project code."
            return (jsonify({"status":"error","message": msg}), 400) if wants_json else (msg, 400)

        next_version = f"V{version_count + 1}"

        # Prepare push frame
        df_to_push = df.copy()
        df_to_push['Version'] = next_version

        # If the table has any analyst-like column, populate it; else keep 'analyst'
        try:
            insp = inspect(engine)
            cols_info = insp.get_columns('Mapped_Results')  # schema can be added if needed
            existing_cols = {c['name'] for c in cols_info}
        except Exception:
            existing_cols = set(df_to_push.columns)

        possible_analyst_cols = ['analyst', 'Analyst', 'pushed_by', 'created_by', 'Created_By', 'username']
        set_any = False
        for col in possible_analyst_cols:
            if col in existing_cols:
                df_to_push[col] = chosen_analyst
                set_any = True
                break
        if not set_any:
            if 'analyst' not in df_to_push.columns:
                df_to_push['analyst'] = chosen_analyst

        # --------- STATS FOR EMAIL SUMMARY ---------
        total_results = len(df_to_push)

        # Yes/No column for "Not in Initial Dataset"
        yn_col = None
        for c in ['yes_no', 'Yes_No', 'YES_NO']:
            if c in df_to_push.columns:
                yn_col = c
                break

        if yn_col:
            not_in_initial = int(
                df_to_push[yn_col].astype(str).str.strip().str.lower().eq('no').sum()
            )
        else:
            not_in_initial = 0

        # Patentryx / Prion columns
        old_col = 'old_rank_order' if 'old_rank_order' in df_to_push.columns else None
        new_col = 'new_rank_order' if 'new_rank_order' in df_to_push.columns else None

        patentryx_count = int(df_to_push[old_col].notna().sum()) if old_col else 0
        prion_count     = int(df_to_push[new_col].notna().sum()) if new_col else 0
        both_count      = int(
            (df_to_push[old_col].notna() & df_to_push[new_col].notna()).sum()
        ) if (old_col and new_col) else 0

        # Dispatch date text (from date_col)
        dispatch_str = ""
        if date_col in df_to_push.columns and len(df_to_push) > 0:
            first_date = df_to_push[date_col].iloc[0]
            if pd.notna(first_date):
                if isinstance(first_date, (datetime, date)):
                    dispatch_str = first_date.strftime('%Y-%m-%d')
                else:
                    dispatch_str = str(first_date)
        if not dispatch_str:
            dispatch_str = datetime.now().strftime('%Y-%m-%d')

        # Actually push
        df_to_push.to_sql("Mapped_Results", con=engine, if_exists="append", index=False)
        results = len(df_to_push)

        # 🔔 Notify admins on successful push (summary format)
        try:
            subject = f"[PriorIQ] Final results pushed – {project_code}"

            body = f"""
Hi Admin,

{project_code} completed by {chosen_analyst} dispatched on {dispatch_str}
Final Dataset successfully run and mapped results pushed to DB
{not_in_initial} out of {total_results} results Not in Initial Dataset
{patentryx_count} out of {total_results} results is identified by Patentryx
{prion_count} out of {total_results} results is identified by Prion
{both_count} out of {total_results} results is identified by Patentryx & Prion

Regards,
PriorIQ App (Datasolve Analytics)
""".strip()

            # Uses your existing helper
            send_admin_email(subject, body)
        except Exception as mail_err:
            print(f"[AdminNotify] PushToDB notify error: {mail_err}")

        # Cleanup on success
        cleanup_stash(project_code)

        ok_msg = f"✅ {results} records for project code '{project_code}' successfully pushed as version {next_version}."
        return jsonify({"status":"ok","message": ok_msg, "results": results, "version": next_version}) if wants_json else ok_msg

    except Exception as e:
        traceback.print_exc()
        err_msg = f"Error: {str(e)}"
        return (jsonify({"status":"error","message": f"❌ {err_msg}"}), 500) if wants_json else (f"❌ {err_msg}", 500)

    finally:
        end_dt = datetime.now()
        runtime_hms = str(end_dt - start_dt).split('.')[0]
        start_date = start_dt.strftime('%Y-%m-%d')
        status_msg = "Success"
        if 'e' in locals():
            status_msg = f"Error: {str(e)}"

        # Write execution log
        log_data = {
            "name"             : chosen_analyst,  # use selected analyst in logs
            "email"            : session_email,
            "abstract"         : "",
            "upload_filename"  : "",
            "downloadfile_name": "",
            "action"           : "PushToDB",
            "start_time"       : start_dt.strftime('%H:%M:%S'),
            "end_time"         : end_dt.strftime('%H:%M:%S'),
            "date"             : start_date,
            "status"           : status_msg,
            "results"          : results or 0,
            "project_code"     : locals().get("project_code", ""),
            "execution_time"   : runtime_hms
        }
        try:
            pd.DataFrame([log_data]).to_sql("prioriq_log", con=engine, if_exists="append", index=False)
        except Exception as e2:
            print(f"Error logging execution details to MySQL: {e2}")


######################################################################################################################
            
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(TEMP_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

#_____________________________________________________________________________________________________

@app.route('/mapped-results', methods=['GET', 'POST'])
def page_c():
    try:
       
        message = None
        # --- 1) Suggestions for autocomplete ---
        suggestion_query = """
            SELECT DISTINCT project_code
            FROM Mapped_Results
            WHERE project_code IS NOT NULL
            ORDER BY project_code
        """
        suggestion_df = pd.read_sql(suggestion_query, con=engine)
        suggestions = suggestion_df['project_code'].astype(str).tolist()
        # --- 2) Load data for table ---
        if request.method == 'POST':
            project_code = (request.form.get('project_code') or '').strip()
            if not project_code:
                message = ":warning: Please provide a project code to search."
                df = pd.DataFrame()
            else:
                query = """
                    SELECT *
                    FROM Mapped_Results
                    WHERE project_code = %s
                    ORDER BY result_order ASC
                """
                df = pd.read_sql(query, con=engine, params=(project_code,))
                if df.empty:
                    message = f"No data found for project code: {project_code}"
        else:
            # Default: show a small recent sample to keep page light
            query = """
                SELECT *
                FROM Mapped_Results
                ORDER BY project_code DESC, result_order ASC
                LIMIT 50
            """
            df = pd.read_sql(query, con=engine)
        # --- 3) Tidy up columns & hide 'id' ---
        if not df.empty:
            # strip accidental whitespace in column names
            df.columns = [c.strip() for c in df.columns]
            # drop 'id' if present
            drop_cols = [c for c in df.columns if c.lower() == 'id']
            if drop_cols:
                df = df.drop(columns=drop_cols)
            # 🎉 NEW CODE: Rename columns for display
            # *************************************************************
            rename_map = {
                'old_rank_order': 'Patentryx',
                'new_rank_order': 'Prion',
                'yes_no': 'Available in Initial Dataset'
            }
            df = df.rename(columns=rename_map)
            # **************************************************************

            # Optional: ensure display order (put project fields first)
            preferred_order = [
                'project_code', 'patent_no', 'result_order',
                'individual_rating', 'Patentryx', 'Prion',
                'Available in Initial Dataset', 'Version'
            ]
            # keep only those that exist, then append the rest in original order
            front = [c for c in preferred_order if c in df.columns]
            rest  = [c for c in df.columns if c not in front]
            df = df[front + rest]
            # Optional: convert obvious numeric cols (so they render clean)
            for c in ['result_order', 'old_rank_order', 'new_rank_order']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            # Render with the class used in your template CSS/JS
            table_html = df.to_html(index=False, classes='results')
        else:
            table_html = ""
        return render_template(
            "search.html",
            table=table_html,
            message=message,
            suggestions=suggestions
        )
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", 500
#_____________________________________________________________________________________________________

from datetime import datetime
from flask import request, render_template, jsonify, redirect, url_for, flash
from sqlalchemy import text
import pandas as pd

# ---------- shared helpers (dates + analyst) ----------
def _parse_date(s):
    """Accepts yyyy-mm-dd, dd-mm-yyyy, yyyy/mm/dd, mm/dd/yyyy."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def _filters_where_args(date_func=False):
    """
    Returns (extra_sql, params_dict) for optional filters:
      - dispatch_date (start_date/end_date)
      - analyst (exact match)
    Set date_func=True to wrap as DATE(dispatch_date) (useful where needed).
    """
    sd = _parse_date(request.args.get("start_date"))
    ed = _parse_date(request.args.get("end_date"))
    analyst = (request.args.get("analyst") or "").strip()

    col = "DATE(dispatch_date)" if date_func else "dispatch_date"

    clauses = []
    args = {}

    if sd and ed:
        clauses.append(f"{col} BETWEEN :sd AND :ed")
        args.update({"sd": sd, "ed": ed})
    elif sd:
        clauses.append(f"{col} >= :sd")
        args.update({"sd": sd})
    elif ed:
        clauses.append(f"{col} <= :ed")
        args.update({"ed": ed})

    if analyst:
        clauses.append("analyst = :analyst")
        args["analyst"] = analyst

    extra = ""
    if clauses:
        extra = " AND " + " AND ".join(clauses)

    return extra, args


# ========= DASHBOARD (CTE-based, consistent denominator) =========
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Keep your original semantics (no DATE() wrapper here)
        extra, params = _filters_where_args(date_func=False)

        sql = f"""
        WITH base AS (
          SELECT *
          FROM elicita.Mapped_Results
          WHERE 1=1{extra}
        )
        SELECT
          -- denominator
          (SELECT COUNT(*) FROM base)                                                   AS total_rows,
          -- projects
          (SELECT COUNT(DISTINCT project_code) FROM base)                               AS total_projects,
          -- unfiltered thresholds (same base)
          (SELECT COUNT(*) FROM base WHERE old_rank_order < 250)                        AS old_lt_250,
          (SELECT COUNT(*) FROM base WHERE new_rank_order < 250)                        AS new_lt_250,
          (SELECT COUNT(*) FROM base WHERE old_rank_order < 250 AND new_rank_order < 250) AS both_lt_250,
          (SELECT COUNT(*) FROM base WHERE (old_rank_order < 250 OR new_rank_order < 250)) AS either_lt_250,
          -- yes_no='NO' totals + buckets (same base)
          (SELECT COUNT(*) FROM base WHERE UPPER(yes_no)='NO')                          AS yes_no_no_total,
          (SELECT COUNT(*) FROM base WHERE UPPER(yes_no)='NO' AND old_rank_order < 250) AS old_lt_250_no,
          (SELECT COUNT(*) FROM base WHERE UPPER(yes_no)='NO' AND new_rank_order < 250) AS new_lt_250_no,
          (SELECT COUNT(*) FROM base WHERE UPPER(yes_no)='NO' AND old_rank_order < 250 AND new_rank_order < 250) AS both_lt_250_no,
          (SELECT COUNT(*) FROM base WHERE UPPER(yes_no)='NO' AND (old_rank_order < 250 OR new_rank_order < 250)) AS either_lt_250_no
        ;
        """

        with engine.connect() as conn:
            row = conn.execute(text(sql), params).mappings().first()

        # pull counts
        total_rows               = row["total_rows"] or 0
        total_projects           = row["total_projects"] or 0
        old_lt_250               = row["old_lt_250"] or 0
        new_lt_250               = row["new_lt_250"] or 0
        both_lt_250              = row["both_lt_250"] or 0
        either_lt_250            = row["either_lt_250"] or 0
        yes_no_no_total          = row["yes_no_no_total"] or 0
        old_lt_250_no            = row["old_lt_250_no"] or 0
        new_lt_250_no            = row["new_lt_250_no"] or 0
        both_lt_250_no           = row["both_lt_250_no"] or 0
        either_lt_250_no         = row["either_lt_250_no"] or 0

        # percentages (≤ 100 because same base)
        old_rank_percentage      = (old_lt_250   / total_rows) * 100 if total_rows else 0
        new_rank_percentage      = (new_lt_250   / total_rows) * 100 if total_rows else 0
        both_lt_250_percentage   = (both_lt_250  / total_rows) * 100 if total_rows else 0
        either_lt_250_percentage = (either_lt_250/ total_rows) * 100 if total_rows else 0

        yes_no_no_percentage                 = (yes_no_no_total / total_rows) * 100 if total_rows else 0
        old_lt_250_yes_no_percentage         = (old_lt_250_no   / yes_no_no_total) * 100 if yes_no_no_total else 0
        new_lt_250_yes_no_percentage         = (new_lt_250_no   / yes_no_no_total) * 100 if yes_no_no_total else 0
        both_lt_250_yes_no_percentage        = (both_lt_250_no  / yes_no_no_total) * 100 if yes_no_no_total else 0
        either_lt_250_yes_no_percentage      = (either_lt_250_no/ yes_no_no_total) * 100 if yes_no_no_total else 0

        return render_template(
            'dashboard.html',
            total_projects=total_projects,
            patents_count=total_rows,  # keep template var name the same
            old_lt_250=old_lt_250,
            new_lt_250=new_lt_250,
            old_rank_percentage=round(old_rank_percentage, 2),
            new_rank_percentage=round(new_rank_percentage, 2),
            both_lt_250=both_lt_250,
            both_lt_250_percentage=round(both_lt_250_percentage, 2),
            either_lt_250=either_lt_250,
            either_lt_250_percentage=round(either_lt_250_percentage, 2),
            old_lt_250_no=old_lt_250_no,
            new_lt_250_no=new_lt_250_no,
            both_lt_250_no=both_lt_250_no,
            either_lt_250_no=either_lt_250_no,
            yes_no_no_total=yes_no_no_total,
            yes_no_no_percentage=round(yes_no_no_percentage, 2),
            old_lt_250_yes_no_percentage=round(old_lt_250_yes_no_percentage, 2),
            new_lt_250_yes_no_percentage=round(new_lt_250_yes_no_percentage, 2),
            both_lt_250_yes_no_percentage=round(both_lt_250_yes_no_percentage, 2),
            either_lt_250_yes_no_percentage=round(either_lt_250_yes_no_percentage, 2)
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Could not load dashboard data: {e}", "error")
        return redirect(url_for('landing'))


@app.route('/dashboard_data')
@login_required
def dashboard_data():
    """
    Returns JSON for the requested metric.
    Optional query params:
      - start_date=YYYY-MM-DD
      - end_date=YYYY-MM-DD
      - analyst=<name>
    """
    metric = (request.args.get('metric') or '').strip().lower()
    extra, params = _filters_where_args(date_func=False)

    queries = {
        # distinct projects
        'projects': f"""
            SELECT DISTINCT project_code AS project
            FROM elicita.Mapped_Results
            WHERE 1=1{extra}
            ORDER BY project_code
        """,

        # all patents (full rows)
        'patents': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE 1=1{extra}
            ORDER BY patent_no
        """,

        # unfiltered thresholds
        'old_lt_250': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE old_rank_order < 250{extra}
            ORDER BY old_rank_order ASC
        """,
        'new_lt_250': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE new_rank_order < 250{extra}
            ORDER BY new_rank_order ASC
        """,
        'both_lt_250': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE old_rank_order < 250 AND new_rank_order < 250{extra}
            ORDER BY LEAST(old_rank_order, new_rank_order) ASC
        """,
        'either_lt_250': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE (old_rank_order < 250 OR new_rank_order < 250){extra}
            ORDER BY LEAST(
                COALESCE(old_rank_order, 999999),
                COALESCE(new_rank_order, 999999)
            ) ASC
        """,

        # yes_no='NO' filtered
        'old_lt_250_no': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE UPPER(yes_no)='NO' AND old_rank_order < 250{extra}
            ORDER BY old_rank_order ASC
        """,
        'new_lt_250_no': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE UPPER(yes_no)='NO' AND new_rank_order < 250{extra}
            ORDER BY new_rank_order ASC
        """,
        'both_lt_250_no': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE UPPER(yes_no)='NO'
              AND old_rank_order < 250 AND new_rank_order < 250{extra}
            ORDER BY LEAST(old_rank_order, new_rank_order) ASC
        """,
        'either_lt_250_no': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE UPPER(yes_no)='NO'
              AND (old_rank_order < 250 OR new_rank_order < 250){extra}
            ORDER BY LEAST(
                COALESCE(old_rank_order, 999999),
                COALESCE(new_rank_order, 999999)
            ) ASC
        """,

        # full rows for yes_no = 'NO'
        'yes_no_no': f"""
            SELECT *
            FROM elicita.Mapped_Results
            WHERE UPPER(yes_no)='NO'{extra}
            ORDER BY project_code, patent_no
        """
    }

    if metric not in queries:
        return jsonify({"error": "unknown metric"}), 400

    with engine.connect() as conn:
        df = pd.read_sql(text(queries[metric]), con=conn, params=params)

    # Drop Version/version if present (unchanged)
    to_drop = [c for c in df.columns if c.lower() == 'version']
    if to_drop:
        df = df.drop(columns=to_drop)

    return jsonify({
        "columns": list(df.columns),
        "rows": df.to_dict(orient='records'),
        "count": int(len(df)),
        "metric": metric
    })


# ========= RATING =========
@app.route('/rating')
@login_required
def rating():
    """Rating page with topline KPIs + 3 rank-filter sections (Explicit, Implicit, Not Disclosed).
       Adds optional date + analyst filtering via ?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&analyst=<name>
    """
    # Build a reusable SQL snippet + bind params (wrap DATE() here like before)
    filt_sql, binds = _filters_where_args(date_func=True)

    try:
        with engine.connect() as conn:
            # Topline
            total_projects = conn.execute(
                text(f"""
                    SELECT COUNT(DISTINCT project_code)
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            patents_count = conn.execute(
                text(f"""
                    SELECT COUNT(patent_no)
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            # ----- Disclosed Explicitly -----
            disclosed_explicitly_total = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating = 'Disclosed Explicitly' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            disclosed_explicitly_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating = 'Disclosed Explicitly'
                      AND UPPER(yes_no)='NO' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            de_old_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            de_old_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            de_new_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            de_new_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND UPPER(yes_no)='NO'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            de_both_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            de_both_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            de_either_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            de_either_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Explicitly'
                      AND UPPER(yes_no)='NO'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            # ----- Disclosed Implicitly -----
            disclosed_implicitly_total = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating = 'Disclosed Implicitly' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            disclosed_implicitly_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND UPPER(yes_no)='NO' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            di_old_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            di_old_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            di_new_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            di_new_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND UPPER(yes_no)='NO'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            di_both_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            di_both_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            di_either_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            di_either_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Disclosed Implicitly'
                      AND UPPER(yes_no)='NO'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            # ----- Not Disclosed -----
            not_disclosed_total = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating = 'Not Disclosed' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            not_disclosed_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND UPPER(yes_no)='NO' {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            nd_old_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            nd_old_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            nd_new_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            nd_new_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND UPPER(yes_no)='NO'
                      AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            nd_both_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            nd_both_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND UPPER(yes_no)='NO'
                      AND old_rank_order < 250 AND new_rank_order < 250 {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

            nd_either_lt_250 = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0
            nd_either_lt_250_no = conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM elicita.Mapped_Results
                    WHERE individual_rating='Not Disclosed'
                      AND UPPER(yes_no)='NO'
                      AND (old_rank_order < 250 OR new_rank_order < 250) {filt_sql}
                """), binds
            ).scalar_one_or_none() or 0

        return render_template(
            'rating.html',
            # topline
            total_projects=total_projects,
            patents_count=patents_count,
            disclosed_explicitly_total=disclosed_explicitly_total,
            disclosed_explicitly_no=disclosed_explicitly_no,
            disclosed_implicitly_total=disclosed_implicitly_total,
            disclosed_implicitly_no=disclosed_implicitly_no,
            not_disclosed_total=not_disclosed_total,
            not_disclosed_no=not_disclosed_no,
            # explicit
            de_old_lt_250=de_old_lt_250,         de_old_lt_250_no=de_old_lt_250_no,
            de_new_lt_250=de_new_lt_250,         de_new_lt_250_no=de_new_lt_250_no,
            de_both_lt_250=de_both_lt_250,       de_both_lt_250_no=de_both_lt_250_no,
            de_either_lt_250=de_either_lt_250,   de_either_lt_250_no=de_either_lt_250_no,
            # implicit
            di_old_lt_250=di_old_lt_250,         di_old_lt_250_no=di_old_lt_250_no,
            di_new_lt_250=di_new_lt_250,         di_new_lt_250_no=di_new_lt_250_no,
            di_both_lt_250=di_both_lt_250,       di_both_lt_250_no=di_both_lt_250_no,
            di_either_lt_250=di_either_lt_250,   di_either_lt_250_no=di_either_lt_250_no,
            # not disclosed
            nd_old_lt_250=nd_old_lt_250,         nd_old_lt_250_no=nd_old_lt_250_no,
            nd_new_lt_250=nd_new_lt_250,         nd_new_lt_250_no=nd_new_lt_250_no,
            nd_both_lt_250=nd_both_lt_250,       nd_both_lt_250_no=nd_both_lt_250_no,
            nd_either_lt_250=nd_either_lt_250,   nd_either_lt_250_no=nd_either_lt_250_no,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Could not load rating page: {e}", "error")
        return redirect(url_for('landing'))


# ========= RESULT =========
from datetime import datetime
from flask import request, render_template, jsonify, redirect, url_for, flash
from sqlalchemy import text
import pandas as pd

# ---------- shared helpers ----------
def _parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def _filters_where_args(date_func: bool = True):
    """
    Build a reusable filter snippet + params for:
      - dispatch_date (optionally wrapped with DATE())
      - analyst (exact match)
    Returns: (filt_sql_str, params_dict)
    filt_sql_str starts with a leading space, e.g. " AND DATE(dispatch_date) BETWEEN :sd AND :ed AND analyst=:analyst"
    """
    sd = _parse_date(request.args.get("start_date"))
    ed = _parse_date(request.args.get("end_date"))
    analyst = (request.args.get("analyst") or "").strip()

    # choose column expression for date filter
    date_col = "DATE(dispatch_date)" if date_func else "dispatch_date"

    parts = []
    params = {}

    if sd and ed:
        parts.append(f"{date_col} BETWEEN :sd AND :ed")
        params["sd"] = sd
        params["ed"] = ed
    elif sd:
        parts.append(f"{date_col} >= :sd")
        params["sd"] = sd
    elif ed:
        parts.append(f"{date_col} <= :ed")
        params["ed"] = ed

    if analyst:
        parts.append("analyst = :analyst")
        params["analyst"] = analyst

    filt_sql = ""
    if parts:
        filt_sql = " AND " + " AND ".join(parts)

    return filt_sql, params


# =========================================
#                 RESULT
# =========================================
@app.route('/result')
@login_required
def result():
    """
    Result Dashboard — grouped by result_order (1..4, 5+).
    Optional query params apply globally:
      ?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&analyst=<name>
    """
    try:
        # Uniform filter (DATE() on dispatch_date + analyst)
        filt_sql, date_params = _filters_where_args(date_func=True)

        buckets = [
            ("1",     "result_order = 1"),
            ("2",     "result_order = 2"),
            ("3",     "result_order = 3"),
            ("4",     "result_order = 4"),
            ("5plus", "result_order >= 5"),
        ]

        with engine.connect() as conn:
            # Topline
            total_projects = conn.execute(
                text(f"""
                    SELECT COUNT(DISTINCT project_code)
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                """),
                date_params
            ).scalar_one_or_none() or 0

            total_patents = conn.execute(
                text(f"""
                    SELECT COUNT(patent_no)
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                """),
                date_params
            ).scalar_one_or_none() or 0

            # Buckets
            result_summary = {}
            for label, cond in buckets:
                total = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                total_no = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE UPPER(yes_no)='NO' AND {cond} {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                pat_lt = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND old_rank_order < 250 {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0
                pat_lt_no = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND old_rank_order < 250
                              AND UPPER(yes_no)='NO' {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                pri_lt = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND new_rank_order < 250 {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0
                pri_lt_no = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND new_rank_order < 250
                              AND UPPER(yes_no)='NO' {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                both_lt = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND old_rank_order < 250
                              AND new_rank_order < 250 {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0
                both_lt_no = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond} AND old_rank_order < 250
                              AND new_rank_order < 250
                              AND UPPER(yes_no)='NO' {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                # Either < 250 (no f-string + .format() mixing)
                either_lt = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond}
                          AND (old_rank_order < 250 OR new_rank_order < 250)
                          {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                either_lt_no = conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM elicita.Mapped_Results
                        WHERE {cond}
                          AND (old_rank_order < 250 OR new_rank_order < 250)
                          AND UPPER(yes_no)='NO'
                          {filt_sql}
                    """),
                    date_params
                ).scalar_one_or_none() or 0

                result_summary[label] = {
                    "total": total, "total_no": total_no,
                    "pat_lt": pat_lt,       "pat_lt_no": pat_lt_no,
                    "pri_lt": pri_lt,       "pri_lt_no": pri_lt_no,
                    "both_lt": both_lt,     "both_lt_no": both_lt_no,
                    "either_lt": either_lt, "either_lt_no": either_lt_no,
                }

        return render_template(
            'result.html',
            total_projects=total_projects,
            total_patents=total_patents,
            result_summary=result_summary,
            start_date=request.args.get("start_date", ""),
            end_date=request.args.get("end_date", ""),
            analyst=request.args.get("analyst", "")
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Could not load Result dashboard: {e}", "error")
        return redirect(url_for('landing'))


@app.route('/result_data')
@login_required
def result_data():
    """
    Drill-down data for tables.

    Modes:
      ?metric=projects        -> project_code, patent_count
      ?metric=patents         -> all patents (entire table)

    Or result-order mode:
      ?order=1|2|3|4|5plus
      &filter=total|old_lt_250|new_lt_250|both_lt_250|either_lt_250
      &no=1 (only 'Not in Initial') | 0 (all)

    Optional everywhere:
      &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&analyst=<name>
    """
    filt_sql, date_params = _filters_where_args(date_func=True)

    metric = (request.args.get('metric') or '').strip().lower()
    if metric:
        with engine.connect() as conn:
            if metric == 'projects':
                sql = f"""
                    SELECT project_code, COUNT(DISTINCT patent_no) AS patent_count
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                    GROUP BY project_code
                    ORDER BY patent_count DESC, project_code
                """
                df = pd.read_sql(text(sql), con=conn, params=date_params)

            elif metric == 'patents':
                sql = f"""
                    SELECT *
                    FROM elicita.Mapped_Results
                    WHERE 1=1 {filt_sql}
                    ORDER BY project_code, patent_no
                """
                df = pd.read_sql(text(sql), con=conn, params=date_params)
                drop_cols = [c for c in df.columns if c.lower() == 'version']
                if drop_cols:
                    df = df.drop(columns=drop_cols)
            else:
                return jsonify({"error": "Invalid 'metric'"}), 400

        return jsonify({
            "columns": list(df.columns),
            "rows": df.to_dict(orient='records'),
            "count": int(len(df)),
            "metric": metric
        })

    # ------- result-order mode -------
    order = (request.args.get('order') or '').strip().lower()
    filter_key = (request.args.get('filter') or 'total').strip().lower()
    only_no = (request.args.get('no') or '0').strip() == '1'

    order_map = {
        "1": "result_order = 1",
        "2": "result_order = 2",
        "3": "result_order = 3",
        "4": "result_order = 4",
        "5plus": "result_order >= 5",
    }
    order_cond = order_map.get(order)
    if not order_cond:
        return jsonify({"error": "Invalid 'order'"}), 400

    filter_map = {
        "total":         "1=1",
        "old_lt_250":    "old_rank_order < 250",
        "new_lt_250":    "new_rank_order < 250",
        "both_lt_250":   "(old_rank_order < 250 AND new_rank_order < 250)",
        "either_lt_250": "(old_rank_order < 250 OR new_rank_order < 250)",
    }
    rank_cond = filter_map.get(filter_key)
    if not rank_cond:
        return jsonify({"error": "Invalid 'filter'"}), 400

    no_cond = "AND UPPER(yes_no)='NO'" if only_no else ""

    sql = f"""
        SELECT project_code, patent_no, result_order,
               old_rank_order, new_rank_order, yes_no,
               dispatch_date, analyst
        FROM elicita.Mapped_Results
        WHERE {order_cond} AND {rank_cond} {no_cond} {filt_sql}
        ORDER BY project_code, patent_no
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), con=conn, params=date_params)

    drop_cols = [c for c in df.columns if c.lower() == 'version']
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return jsonify({
        "columns": list(df.columns),
        "rows": df.to_dict(orient='records'),
        "count": int(len(df)),
        "order": order,
        "filter": filter_key,
        "only_no": only_no
    })


#_______________________________________________________________________________________________________________________________

@app.route("/log", methods=["GET"])
@login_required
@admin_required
def logs():
    # ---------- GET FILTERS ----------
    project_code     = request.args.get("project_code", "").strip()
    action           = request.args.get("action", "").strip()
    status           = request.args.get("status", "").strip()
    results_selected = request.args.get("results", "").strip()
    who              = request.args.get("who", "").strip()
    date_from        = request.args.get("date_from", "").strip()
    date_to          = request.args.get("date_to", "").strip()
    page             = int(request.args.get("page", 1))
    per_page         = int(request.args.get("per_page", 20))
    offset           = (page - 1) * per_page

    # ---------- BASE QUERY ----------
    base_query = "SELECT * FROM elicita.prioriq_log WHERE 1=1"
    filters = {}

    if project_code:
        base_query += " AND project_code LIKE :project_code"
        filters["project_code"] = f"%{project_code}%"

    if action:
        base_query += " AND action = :action"
        filters["action"] = action

    if status:
        base_query += " AND status = :status"
        filters["status"] = status

    if results_selected:
        base_query += " AND results = :results"
        filters["results"] = results_selected

    if who:
        base_query += " AND (name LIKE :who OR email LIKE :who)"
        filters["who"] = f"%{who}%"

    if date_from:
        base_query += " AND date >= :date_from"
        filters["date_from"] = date_from

    if date_to:
        base_query += " AND date <= :date_to"
        filters["date_to"] = date_to

    count_query     = f"SELECT COUNT(*) FROM ({base_query}) AS sub"
    paginated_query = f"{base_query} ORDER BY id DESC LIMIT :limit OFFSET :offset"

    # ---------- EXECUTE ----------
    with db.engine.connect() as conn:
        total = conn.execute(text(count_query), filters).scalar()
        filters["limit"]  = per_page
        filters["offset"] = offset

        df = pd.read_sql(text(paginated_query), conn, params=filters)

        # --- clean/format time columns ---
        for col in ["start_time", "end_time", "execution_time"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r"0 days\s*", "", regex=True)

        # ---------- Column order & labels ----------
        desired_order = [
            "name","project_code","action",
            "start_time","end_time","execution_time",
            "date","abstract","upload_filename","downloadfile_name",
            "initial_rows","final_rows","results","status",
        ]
        existing = [c for c in desired_order if c in df.columns]
        df = df.reindex(columns=existing)

        display_names = {
            "name": "Name",
            "project_code": "Project Code",
            "action": "Action",
            "start_time": "Start Time",
            "end_time": "End Time",
            "execution_time": "Execution Time",
            "date": "Date",
            "abstract": "Abstract",
            "upload_filename": "Upload File Name",
            "downloadfile_name": "Download File Name",
            "initial_rows": "Initial Rows",
            "final_rows": "Final Rows",
            "results": "Results",
            "status": "Status",
        }
        df = df.rename(columns=display_names)

        # ---------- Dropdown lists ----------
        actions = pd.read_sql(
            text("""SELECT DISTINCT action FROM elicita.prioriq_log
                    WHERE action IS NOT NULL ORDER BY action"""), conn
        )["action"].dropna().tolist()

        statuses = pd.read_sql(
            text("""SELECT DISTINCT status FROM elicita.prioriq_log
                    WHERE status IS NOT NULL ORDER BY status"""), conn
        )["status"].dropna().tolist()

        results = pd.read_sql(
            text("""SELECT DISTINCT results FROM elicita.prioriq_log
                    WHERE results IS NOT NULL ORDER BY results"""), conn
        )["results"].dropna().tolist()

        projects = pd.read_sql(
            text("""SELECT DISTINCT project_code FROM elicita.prioriq_log
                    WHERE project_code IS NOT NULL ORDER BY project_code"""), conn
        )["project_code"].dropna().tolist()

    # ---------- Pagination ----------
    total_pages = max(ceil(total / per_page), 1)
    query_args = request.args.to_dict(flat=True)
    query_args.setdefault("per_page", str(per_page))

    q_prev = dict(query_args); q_prev["page"] = max(page - 1, 1)
    prev_url = url_for("logs", **q_prev)

    q_next = dict(query_args); q_next["page"] = min(page + 1, total_pages)
    next_url = url_for("logs", **q_next)

    # ---------- Table HTML ----------
    table_html = df.to_html(
        classes="table table-bordered logs-table table-striped mb-0",
        index=False,
        escape=False
    )

    # ---------- Render ----------
    return render_template(
        "logs.html",
        table=table_html,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        project_code=project_code,
        action=action,
        status=status,
        who=who,
        date_from=date_from,
        date_to=date_to,
        actions=actions,
        statuses=statuses,
        projects=projects,
        results=results,
        results_selected=results_selected,
        prev_url=prev_url,
        next_url=next_url
    )

##################################################################################################################

#overall
from datetime import datetime, date, timedelta
from flask import render_template, request, jsonify, flash, redirect, url_for
from sqlalchemy import text
import pandas as pd

# Assumes: engine, app, login_required exist in your app context.


# ---------- helpers (Overall) ----------
def _first_day_of_month(d: date) -> date:
    return d.replace(day=1)

def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def _parse_date(s: str):
    """Accepts yyyy-mm-dd, dd-mm-yyyy, yyyy/mm/dd, mm/dd/yyyy; returns date or None."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

def _overall_where(start_date: str, end_date: str, analyst: str):
    """
    Build WHERE clause and bind params for Overall page.
    Date filter uses DATE(dispatch_date) so it works for DATETIME columns too.
    """
    binds = {}
    clauses = []

    if start_date:
        clauses.append("DATE(dispatch_date) >= :sd")
        binds["sd"] = start_date
    if end_date:
        clauses.append("DATE(dispatch_date) <= :ed")
        binds["ed"] = end_date

    a = (analyst or "").strip()
    if a:
        clauses.append("analyst = :analyst")
        binds["analyst"] = a

    if clauses:
        return "WHERE " + " AND ".join(clauses), binds
    return "", binds


# ========== OVERALL ==========
@app.route('/dashboard-overall', defaults={'tab': 'overall'})
@app.route('/dashboard-<tab>')
@app.route('/overall', defaults={'tab': 'overall'})      # backward compatibility
@app.route('/overall/<tab>')                            # old style /overall/a etc.
@login_required
def overall(tab='overall'):
    """
    Overall — landing that links to Dashboard, Rating, Result,
    with topline KPIs (projects, patents, rating buckets, result buckets).

    URLs:
        /dashboard-overall  -> Dashboard
        /dashboard-rating   -> Rating
        /dashboard-result   -> Result
      (old still work)
        /overall, /overall/a  -> Dashboard
        /overall/b            -> Rating
        /overall/c            -> Result

    Query params:
        start_date=YYYY-MM-DD
        end_date=YYYY-MM-DD
        analyst=<exact name in `analyst` column>

    Defaults:
      - If ALL three are blank -> set start_date to first of this month and end_date to today.
      - If analyst is given but no dates -> show analyst-only filter (no date constraint).
    """
    # incoming params (raw strings from querystring)
    q_start   = (request.args.get('start_date') or '').strip()
    q_end     = (request.args.get('end_date') or '').strip()
    q_analyst = (request.args.get('analyst') or '').strip()

    # ✅ Default month only if ALL filters are empty
    if not q_start and not q_end and not q_analyst:
        today = date.today()
        q_start = _yyyymmdd(_first_day_of_month(today))
        q_end   = _yyyymmdd(today)

    # WHERE fragment for the main KPIs/cards
    where_sql, binds = _overall_where(q_start, q_end, q_analyst)

    # Map tab -> label  (supports new 'overall|rating|result' and old 'a|b|c')
    tab_key = (tab or 'overall').lower()
    tab_map = {
        'overall': 'Dashboard',
        'rating':  'Rating',
        'result':  'Result',
        'a': 'Dashboard',
        'b': 'Rating',
        'c': 'Result',
    }
    active_label = tab_map.get(tab_key, 'Dashboard')

    # navigation tiles (tab key now uses new slugs)
    sites = [
        {"label": "Dashboard", "url": url_for('dashboard'), "tab": "overall"},
        {"label": "Rating",    "url": url_for('rating'),    "tab": "rating"},
        {"label": "Result",    "url": url_for('result'),    "tab": "result"},
    ]

    try:
        with engine.connect() as conn:
            # 🔎 Analyst list filtered by the CURRENT date range (not by analyst itself)
            # This is what drives the dropdown so it shows only relevant names.
            analyst_binds = {
                "sd": (q_start or None),
                "ed": (q_end or None),
            }
            analysts = conn.execute(text("""
                SELECT DISTINCT analyst
                FROM elicita.Mapped_Results
                WHERE analyst IS NOT NULL AND analyst <> ''
                  AND (:sd IS NULL OR DATE(dispatch_date) >= :sd)
                  AND (:ed IS NULL OR DATE(dispatch_date) <= :ed)
                ORDER BY analyst
            """), analyst_binds).scalars().all()

            # Topline — Projects & patents
            total_projects = conn.execute(
                text(f"""
                    SELECT COUNT(DISTINCT project_code)
                    FROM elicita.Mapped_Results
                    {where_sql}
                """), binds
            ).scalar_one_or_none() or 0

            total_patents = conn.execute(
                text(f"""
                    SELECT COUNT(patent_no)
                    FROM elicita.Mapped_Results
                    {where_sql}
                """), binds
            ).scalar_one_or_none() or 0

            # Rating buckets
            rating_rows = conn.execute(text(f"""
                SELECT individual_rating AS rating, COUNT(*) AS cnt
                FROM elicita.Mapped_Results
                {where_sql}
                GROUP BY individual_rating
            """), binds).mappings().all()
            rating_counts = { (r['rating'] or 'Unknown'): int(r['cnt']) for r in rating_rows }
            disclosed_explicitly = rating_counts.get('Disclosed Explicitly', 0)
            disclosed_implicitly = rating_counts.get('Disclosed Implicitly', 0)
            not_disclosed        = rating_counts.get('Not Disclosed', 0)

            # Result order buckets
            result_rows = conn.execute(text(f"""
                SELECT
                    CASE
                        WHEN result_order = 1 THEN '1'
                        WHEN result_order = 2 THEN '2'
                        WHEN result_order = 3 THEN '3'
                        WHEN result_order = 4 THEN '4'
                        WHEN result_order >= 5 THEN '5plus'
                        ELSE 'Unknown'
                    END AS bucket,
                    COUNT(*) AS cnt
                FROM elicita.Mapped_Results
                {where_sql}
                GROUP BY bucket
            """), binds).mappings().all()
            result_counts = { rr['bucket']: int(rr['cnt']) for rr in result_rows }

        # render
        return render_template(
            "overall.html",
            sites=sites,
            # echo filters
            start_date=q_start,
            end_date=q_end,
            analyst=q_analyst,
            analysts=analysts,  # filtered list
            # topline
            total_projects=total_projects,
            total_patents=total_patents,
            # rating split
            disclosed_explicitly=disclosed_explicitly,
            disclosed_implicitly=disclosed_implicitly,
            not_disclosed=not_disclosed,
            # result buckets
            result_counts=result_counts,
            # 🔹 which tab is active (frontend use)
            active_label=active_label,
            active_tab=tab_key,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f"Could not load Overall page: {e}", "error")
        return redirect(url_for('landing'))


# ========== OVERALL DATA (drilldowns used by embedded views) ==========
@app.route('/overall_data')
@login_required
def overall_data():
    """
    Drill-down API for Overall page widgets.
    Optional filters (?start_date, ?end_date, ?analyst) applied to DATE(dispatch_date) & analyst.
    Modes:
      ?metric=projects
      ?metric=patents
      ?metric=by_rating   [&rating=...]
      ?metric=by_result_order [&bucket=1|2|3|4|5plus]
      &limit=N (optional)
    """
    q_start   = (request.args.get('start_date') or '').strip()
    q_end     = (request.args.get('end_date') or '').strip()
    q_analyst = (request.args.get('analyst') or '').strip()

    where_sql, binds = _overall_where(q_start, q_end, q_analyst)

    # Helper to merge base WHERE with extra conditions safely
    def _merge_where(base_where: str, extra_sql: str) -> str:
        if not extra_sql:
            return base_where
        if base_where:
            return f"{base_where} {extra_sql}"
        return f"WHERE 1=1 {extra_sql}"

    metric = (request.args.get('metric') or '').strip().lower()
    limit  = max(0, int(request.args.get('limit') or 0))  # 0 = no cap

    if metric not in {'projects', 'patents', 'by_rating', 'by_result_order'}:
        return jsonify({"error": f"invalid metric '{metric}'"}), 400

    try:
        with engine.connect() as conn:
            if metric == 'projects':
                sql = f"""
                    SELECT project_code, COUNT(DISTINCT patent_no) AS patent_count
                    FROM elicita.Mapped_Results
                    {where_sql}
                    GROUP BY project_code
                    ORDER BY patent_count DESC, project_code
                """
                df = pd.read_sql(text(sql), con=conn, params=binds)

            elif metric == 'patents':
                sql = f"""
                    SELECT *
                    FROM elicita.Mapped_Results
                    {where_sql}
                    ORDER BY project_code, patent_no
                """
                df = pd.read_sql(text(sql), con=conn, params=binds)
                drop_cols = [c for c in df.columns if c.lower() == 'version']
                if drop_cols:
                    df = df.drop(columns=drop_cols)

            elif metric == 'by_rating':
                rating_filter = (request.args.get('rating') or '').strip()
                local_binds = dict(binds)
                extra = ""
                if rating_filter:
                    extra = " AND individual_rating = :_rating "
                    local_binds["_rating"] = rating_filter
                sql = f"""
                    SELECT *
                    FROM elicita.Mapped_Results
                    {_merge_where(where_sql, extra)}
                    ORDER BY individual_rating, project_code, patent_no
                """
                df = pd.read_sql(text(sql), con=conn, params=local_binds)

            elif metric == 'by_result_order':
                bucket = (request.args.get('bucket') or '').strip().lower()
                local_binds = dict(binds)
                extra = ""
                if bucket in {'1','2','3','4'}:
                    extra = " AND result_order = :_ro "
                    local_binds["_ro"] = int(bucket)
                elif bucket == '5plus':
                    extra = " AND result_order >= 5 "
                sql = f"""
                    SELECT *
                    FROM elicita.Mapped_Results
                    {_merge_where(where_sql, extra)}
                    ORDER BY result_order, project_code, patent_no
                """
                df = pd.read_sql(text(sql), con=conn, params=local_binds)

        if limit and len(df) > limit:
            df = df.head(limit)

        return jsonify({
            "columns": list(df.columns),
            "rows": df.to_dict(orient='records'),
            "count": int(len(df)),
            "metric": metric
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"failed to fetch overall_data: {e}"}), 500


# ========== Tiny helper API for dynamic Analyst dropdown ==========
@app.route('/overall_analysts')
@login_required
def overall_analysts():
    """
    Returns analysts that have rows inside the provided date window.
    Used by overall.html to keep the Analyst dropdown in sync with date filters.
    """
    q_start = (request.args.get('start_date') or '').strip()
    q_end   = (request.args.get('end_date') or '').strip()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT analyst
            FROM elicita.Mapped_Results
            WHERE analyst IS NOT NULL AND analyst <> ''
              AND (:sd IS NULL OR DATE(dispatch_date) >= :sd)
              AND (:ed IS NULL OR DATE(dispatch_date) <= :ed)
            ORDER BY analyst
        """), {"sd": (q_start or None), "ed": (q_end or None)}).scalars().all()

    return jsonify({"analysts": rows})




# ========== OVERALL PAGES (Dashboard / Rating / Result) ==========







# @app.route('/c')
# def page_c():
#     return render_template("c.html")
@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('query', '').lower()
    project_codes = [...]  # e.g., from database or file
    matches = [code for code in project_codes if query in code.lower()]
    return jsonify(matches)

@app.route('/d')
def page_d():
    return render_template("d.html")

@app.route('/logout1')
def logout1():
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True)
