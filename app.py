"""
Shetkari Mitra website server - Powered by Viksit Minds AI Engine
"""
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
from datetime import date
from html import escape
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlparse

import database as db

# ------------------------------------------------------------------ Safe AI Engine Imports
MANDI_FEATURES = None
try:
    from predict_mandi import load_and_prepare_features, predict_for_farmer
    print("Loading Mandi ML feature matrix into memory...")
    MANDI_FEATURES = load_and_prepare_features()
    print("AI Prediction Engine ready!")
except Exception as e:
    print(f"Notice: AI engine not loaded yet ({e}). Running in fallback mode.")

# ------------------------------------------------------------------ Server Settings
BASE = Path(__file__).parent.resolve()
STATIC = (BASE / "static").resolve()
HOST, PORT = "127.0.0.1", 8000
MAX_BODY = 10_000
SESSION_TTL = 2 * 60 * 60

MIME = {
    ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".svg": "image/svg+xml", ".ico": "image/x-icon",
}

NOTICES = {
    "created":  ("Account created. Please log in.", True),
    "welcome":  ("You are logged in.", True),
    "updated":  ("Your profile was saved.", True),
    "deleted":  ("Your account was deleted.", True),
    "expired":  ("Please log in to continue.", False),
}
GENDERS = ("male", "female", "undisclosed")
MARITAL = ("married", "unmarried", "undisclosed")

SESSIONS = {}
LOCK = threading.Lock()


def new_session(email):
    sid = secrets.token_urlsafe(32)
    now = time.time()
    with LOCK:
        for k in [k for k, v in list(SESSIONS.items()) if v["expires"] < now]:
            del SESSIONS[k]
        SESSIONS[sid] = {"email": email, "csrf": secrets.token_urlsafe(16), "expires": now + SESSION_TTL}
    return sid


def read_file(path_str):
    candidates = [
        BASE / path_str,
        BASE / "static" / path_str,
        BASE / "templates" / Path(path_str).name,
        BASE / "static" / "templates" / Path(path_str).name
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Template file not found: checked {[str(c) for c in candidates]}")


def nav_html(active):
    items = [
        ("home", "/", "Home", "मुख्यपृष्ठ"),
        ("dashboard", "/dashboard", "AI Engine", "AI अंदाज"),
        ("apmc", "/apmc", "APMC's", "बाजार समित्या")
    ]
    return "".join(
        f'<a{" class=active" if key == active else ""} href="{href}" data-en="{en_lbl}" data-mr="{mr_lbl}">{en_lbl}</a>'
        for key, href, en_lbl, mr_lbl in items
    )


def account_html(user):
    if user is None:
        return (
            '<div class="header-auth-actions">'
            '<a class="btn-auth-top" href="/auth" data-en="Login / Signup" data-mr="लॉगिन / नोंदणी">Login / Signup</a>'
            '</div>'
        )
    first = escape(user["full_name"].split()[0] if user["full_name"].strip() else "Profile")
    return (
        f'<div class="header-auth-actions">'
        f'<a class="account-link-badge" href="/profile">👤 {first}</a>'
        f'<a class="logout-link-badge" href="/logout" data-en="Log out" data-mr="बाहेर पडा">Log out</a>'
        f'</div>'
    )


def flash(message, ok):
    if not message:
        return ""
    return f'<p class="message {"ok" if ok else "error"}" role="alert">{escape(message)}</p>'


def render(page, title, css, user=None, active="", message="", ok=False, raw=None, **values):
    safe = {k: escape(str(v)) for k, v in values.items()}
    safe.update(raw or {})
    content = Template(read_file(f"templates/{page}.html")).safe_substitute(
        message=flash(message, ok), **safe)
    return Template(read_file("templates/base.html")).safe_substitute(
        title=escape(title), page_css=css, nav=nav_html(active),
        account=account_html(user), content=content)


def radios(name, current, options):
    return {f"{name}_{o}": (" checked" if o == current else "") for o in options}


EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def check_mobile(m):
    return m == "" or re.fullmatch(r"\d{10}", m) is not None


def check_dob(d):
    if d == "":
        return True
    try:
        return date.fromisoformat(d) <= date.today()
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, ctype, status=200, cookies=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Access-Control-Allow-Origin", "*")
        for c in cookies:
            self.send_header("Set-Cookie", c)
        self.end_headers()
        self.wfile.write(body)

    def send_page(self, html, status=200, cookies=()):
        self.send_body(html.encode("utf-8"), "text/html; charset=utf-8", status, cookies)

    def redirect(self, location, cookies=()):
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        for c in cookies:
            self.send_header("Set-Cookie", c)
        self.end_headers()

    def not_found(self, user=None):
        self.send_page(render("notfound", "Page not found", "auth.css", user), 404)

    def sid(self):
        jar = SimpleCookie(self.headers.get("Cookie", ""))
        return jar["sid"].value if "sid" in jar else None

    def current(self):
        sid = self.sid()
        with LOCK:
            sess = SESSIONS.get(sid)
            if sess and sess["expires"] < time.time():
                del SESSIONS[sid]
                sess = None
        if not sess:
            return None, None
        try:
            return db.get_user_by_email(sess["email"]), sess
        except (db.UserNotFoundError, Exception):
            with LOCK:
                SESSIONS.pop(sid, None)
            return None, None

    @staticmethod
    def login_cookie(sid):
        return f"sid={sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_TTL}"

    @staticmethod
    def clear_cookie():
        return "sid=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"

    def serve_static(self, rel):
        candidates = [STATIC / rel, BASE / rel]
        target = None
        for c in candidates:
            if c.is_file():
                target = c
                break
        if not target:
            return self.not_found()
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        self.send_body(target.read_bytes(), ctype)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = url.path

            if path.startswith("/static/"):
                return self.serve_static(path[len("/static/"):])

            # Dropdown metadata route: returns districts, markets, and vegetables per market!
            if path == "/api/dropdown-options":
                if MANDI_FEATURES is None:
                    err = json.dumps({"error": "AI features not loaded on server."}).encode("utf-8")
                    return self.send_body(err, "application/json; charset=utf-8", 500)

                districts = sorted(MANDI_FEATURES["district"].dropna().unique().tolist())
                commodities = sorted(MANDI_FEATURES["commodity"].dropna().unique().tolist())
                
                # District -> list of APMC markets
                district_markets = (
                    MANDI_FEATURES.groupby("district")["market"]
                    .unique()
                    .apply(lambda arr: sorted(list(arr)))
                    .to_dict()
                )

                # Market -> ONLY the vegetables traded in that market!
                market_commodities = (
                    MANDI_FEATURES.groupby("market")["commodity"]
                    .unique()
                    .apply(lambda arr: sorted(list(arr)))
                    .to_dict()
                )

                payload = json.dumps({
                    "districts": districts,
                    "commodities": commodities,
                    "markets_by_district": district_markets,
                    "commodities_by_market": market_commodities
                }, default=str).encode("utf-8")
                return self.send_body(payload, "application/json; charset=utf-8")

            if path == "/api/predict":
                if MANDI_FEATURES is None:
                    err = json.dumps({"error": "AI models not loaded."}).encode("utf-8")
                    return self.send_body(err, "application/json; charset=utf-8", 500)

                district = query.get("district", [""])[0]
                market = query.get("market", [""])[0]
                commodity = query.get("commodity", [""])[0]

                result = predict_for_farmer(district, market, commodity, MANDI_FEATURES)
                payload = json.dumps(result, default=str).encode("utf-8")
                return self.send_body(payload, "application/json; charset=utf-8")

            user, sess = self.current()

            if path == "/":
                self.send_page(render("index", "Home", "home.css", user, "home"))
            elif path == "/dashboard":
                if not user:
                    return self.redirect("/auth?notice=expired")
                msg, ok = NOTICES.get(query.get("notice", [""])[0], ("", False))
                self.send_page(render("dashboard", "AI Engine", "dashboard.css", user, "dashboard", msg, ok))
            elif path == "/profile":
                if not user:
                    return self.redirect("/auth?notice=expired")
                msg, ok = NOTICES.get(query.get("notice", [""])[0], ("", False))
                self.send_page(render("profile", "Farmer Profile", "dashboard.css", user, "", msg, ok,
                                      full_name=user["full_name"], email=user["email"],
                                      mobile=user["mobile"] or "Not added", created_at=user["created_at"]))
            elif path == "/apmc":
                self.send_page(render("apmc", "APMC Mandis", "dashboard.css", user, "apmc"))
            elif path in ("/auth", "/login", "/signup"):
                if user:
                    return self.redirect("/dashboard")
                mode = "signup" if path == "/signup" or query.get("mode", [""])[0] == "signup" else "login"
                msg, ok = NOTICES.get(query.get("notice", [""])[0], ("", False))
                self.send_page(self.auth_page(mode, msg, ok))
            elif path == "/update":
                if not user:
                    return self.redirect("/auth?notice=expired")
                msg, ok = NOTICES.get(query.get("notice", [""])[0], ("", False))
                self.send_page(self.update_page(user, sess, msg, ok))
            elif path == "/logout":
                if not user and query.get("done", [""])[0] != "1":
                    return self.redirect("/auth?notice=expired")
                state = "confirm" if user else "done"
                self.send_page(render("logout", "Log out", "logout.css", user, state=state,
                                      csrf=sess["csrf"] if sess else ""))
            else:
                self.not_found(user)

        except Exception as e:
            print("\n--- SERVER ERROR IN GET REQUEST ---")
            traceback.print_exc()
            self.send_page(render("notfound", "Error", "auth.css", None,
                                  message=f"Server error: {e}"), 500)

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if length > MAX_BODY:
                return self.send_page(render("notfound", "Error", "auth.css", message="Request too large."), 413)

            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"), keep_blank_values=True)
            get = lambda name: form.get(name, [""])[0]

            if path == "/login":
                return self.handle_login(get)
            if path == "/signup":
                return self.handle_signup(get)

            user, sess = self.current()
            if not user:
                return self.redirect("/auth?notice=expired")
            if not secrets.compare_digest(get("csrf"), sess["csrf"]):
                return self.send_page(render("notfound", "Error", "auth.css", user,
                                             message="Invalid form token. Reload the page and try again."), 403)
            if path == "/update":
                self.handle_update(user, sess, get)
            elif path == "/delete":
                self.handle_delete(user, sess, get)
            elif path == "/logout":
                with LOCK:
                    SESSIONS.pop(self.sid(), None)
                self.redirect("/logout?done=1", [self.clear_cookie()])
            else:
                self.not_found(user)

        except Exception as e:
            print("\n--- SERVER ERROR IN POST REQUEST ---")
            traceback.print_exc()
            self.send_page(render("notfound", "Error", "auth.css", None,
                                  message=f"Server error: {e}"), 500)

    def auth_page(self, mode, message="", ok=False, status=200, **v):
        vals = {"login_email": "", "su_name": "", "su_mobile": "", "su_email": ""}
        vals.update(v)
        return render("auth", "Login / Sign up", "auth.css", None, "", message, ok,
                      raw={"mode": mode}, **vals)

    def update_page(self, user, sess, message="", ok=False, form=None):
        v = {"full_name": user["full_name"], "mobile": user["mobile"], "dob": user["dob"],
             "gender": user["gender"], "marital": user["marital"]}
        if form:
            v.update(form)
        raw = {**radios("gender", v["gender"], GENDERS), **radios("marital", v["marital"], MARITAL)}
        return render("update", "Update profile", "update.css", user, "", message, ok, raw=raw,
                      full_name=v["full_name"], mobile=v["mobile"], dob=v["dob"],
                      email=user["email"], csrf=sess["csrf"])

    def handle_signup(self, get):
        name, email = get("full_name").strip(), get("email").strip()
        mobile = re.sub(r"[\s-]", "", get("mobile"))
        password = get("password")
        error = None
        if not name:
            error = "Please enter your name."
        elif not EMAIL_RE.fullmatch(email):
            error = "Please enter a valid email address."
        elif not check_mobile(mobile):
            error = "Mobile number must be 10 digits."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != get("confirm_password"):
            error = "Passwords do not match."
        keep = dict(su_name=name, su_mobile=mobile, su_email=email)
        if not error:
            try:
                db.create_user(name, email, password, mobile)
            except db.DuplicateEmailError as e:
                error = str(e)
        if error:
            return self.send_page(self.auth_page("signup", error, **keep), 400)
        self.redirect("/auth?mode=login&notice=created")

    def handle_login(self, get):
        email = get("email").strip()
        try:
            user = db.authenticate_user(email, get("password"))
        except db.InvalidCredentialsError as e:
            return self.send_page(self.auth_page("login", str(e), login_email=email), 401)
        self.redirect("/dashboard?notice=welcome", [self.login_cookie(new_session(user["email"]))])

    def handle_update(self, user, sess, get):
        name = get("full_name").strip()
        mobile = re.sub(r"[\s-]", "", get("mobile"))
        dob, gender, marital = get("dob").strip(), get("gender"), get("marital")
        new_pw = get("new_password")
        error = None
        if not name:
            error = "Name cannot be empty."
        elif not check_mobile(mobile):
            error = "Mobile number must be 10 digits."
        elif not check_dob(dob):
            error = "Enter a valid date of birth."
        elif gender not in GENDERS or marital not in MARITAL:
            error = "Please choose a valid option."
        elif new_pw and len(new_pw) < 8:
            error = "New password must be at least 8 characters."
        elif new_pw != get("confirm_new_password"):
            error = "New passwords do not match."
        form = dict(full_name=name, mobile=mobile, dob=dob, gender=gender, marital=marital)
        if not error:
            try:
                db.update_user(user["email"], get("current_password"), name, mobile, dob,
                               gender, marital, new_pw or None)
            except db.InvalidCredentialsError:
                error = "Current password is incorrect."
        if error:
            return self.send_page(self.update_page(user, sess, error, form=form), 400)
        self.redirect("/profile?notice=updated")

    def handle_delete(self, user, sess, get):
        try:
            db.delete_user(user["email"], get("delete_password"))
        except db.InvalidCredentialsError:
            return self.send_page(
                self.update_page(user, sess, "Password incorrect. Account NOT deleted."), 401)
        except db.UserNotFoundError:
            pass
        with LOCK:
            SESSIONS.pop(self.sid(), None)
        self.redirect("/auth?mode=signup&notice=deleted", [self.clear_cookie()])


if __name__ == "__main__":
    db.init_db()
    print(f"Shetkari Mitra server running at http://{HOST}:{PORT}")
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")