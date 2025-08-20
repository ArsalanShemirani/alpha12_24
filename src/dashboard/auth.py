#!/usr/bin/env python3
"""
Simple authentication helpers for Streamlit dashboard.

Environment variables:
  - DASH_AUTH_ENABLED (default: "1")
  - DASH_USERNAME (required when enabled)
  - DASH_PASSWORD (plain) or DASH_PASSWORD_HASH (sha256:<hex>)

Usage (in app.py):
  from src.dashboard.auth import login_gate, render_logout_sidebar
  if not login_gate():
      return
  ...
  with st.sidebar:
      render_logout_sidebar()
"""

import os
import time
import hashlib
import streamlit as st
from datetime import datetime


def _get_env_bool(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.getenv(name, default)))
    except Exception:
        v = os.getenv(name, default).strip().lower()
        return v in {"1", "true", "yes", "y"}


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_password(input_password: str, plain: str | None, hashed: str | None) -> bool:
    if hashed:
        # expected format: sha256:<hex>
        parts = hashed.split(":", 1)
        if len(parts) == 2 and parts[0].lower() == "sha256":
            return _sha256_hex(input_password) == parts[1].strip().lower()
        # fallback: treat as raw hex
        return _sha256_hex(input_password) == hashed.strip().lower()
    if plain is not None:
        return input_password == plain
    return False


def login_gate() -> bool:
    """Render a login form and return True when authenticated.

    Stores auth state in st.session_state["is_authenticated"].
    """
    auth_enabled = _get_env_bool("DASH_AUTH_ENABLED", "1")
    if not auth_enabled:
        # Explicitly disabled, allow access
        st.session_state["is_authenticated"] = True
        return True

    # Already authenticated in this session
    if st.session_state.get("is_authenticated") is True:
        # Check for session timeout (24 hours)
        auth_time = st.session_state.get("auth_time", 0)
        current_time = int(time.time())
        session_timeout = 24 * 60 * 60  # 24 hours in seconds
        
        if current_time - auth_time > session_timeout:
            # Session expired, clear auth state
            st.session_state["is_authenticated"] = False
            st.session_state["auth_user"] = None
            st.session_state["auth_time"] = None
            st.session_state["auth_just_logged_in"] = False
        else:
            return True
    
    # Check if we're in the middle of a rerun after successful login
    if st.session_state.get("auth_just_logged_in") is True:
        st.session_state["auth_just_logged_in"] = False
        return True

    username_env = os.getenv("DASH_USERNAME", "").strip()
    password_env = os.getenv("DASH_PASSWORD")  # optional when HASH is provided
    password_hash_env = os.getenv("DASH_PASSWORD_HASH")  # sha256:<hex>

    st.title("🔐 Login Required")
    st.caption("This dashboard is protected. Please sign in.")

    if not username_env or (password_env is None and not password_hash_env):
        st.error("Dashboard credentials not configured. Set DASH_USERNAME and DASH_PASSWORD or DASH_PASSWORD_HASH.")
        with st.expander("Setup instructions"):
            st.code(
                """
export DASH_AUTH_ENABLED=1
export DASH_USERNAME=your_user
export DASH_PASSWORD=strong_password
# or use hash:
# export DASH_PASSWORD_HASH=sha256:$(python - <<'PY'\nimport hashlib; print(hashlib.sha256(b"strong_password").hexdigest())\nPY)
                """.strip(),
                language="bash",
            )
        return False

    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in", use_container_width=True)

        if submit:
            if user == username_env and _verify_password(pwd, password_env, password_hash_env):
                st.session_state["is_authenticated"] = True
                st.session_state["auth_user"] = user
                st.session_state["auth_time"] = int(time.time())
                st.session_state["auth_just_logged_in"] = True
                st.success("Authenticated. Loading dashboard...")
                st.rerun()
            else:
                st.error("Invalid credentials")

    return False


def render_logout_sidebar():
    """Render a small auth box in the sidebar with logout control."""
    if not _get_env_bool("DASH_AUTH_ENABLED", "1"):
        return
    st.markdown("---")
    st.caption("Authentication")
    user = st.session_state.get("auth_user", "user")
    st.write(f"Signed in as: **{user}**")
    
    # Show session information
    if 'session_start_time' in st.session_state:
        session_duration = datetime.now() - st.session_state['session_start_time']
        hours = int(session_duration.total_seconds() // 3600)
        minutes = int((session_duration.total_seconds() % 3600) // 60)
        st.caption(f"Session active: {hours}h {minutes}m")
    
    # Show UI override status
    try:
        from src.core.ui_config import is_ui_config_recent
        if is_ui_config_recent(hours=1):  # Check if UI config was updated in last hour
            st.success("✅ UI settings active (overriding autosignal)")
        else:
            st.info("ℹ️ Using default autosignal settings")
    except Exception:
        pass
    
    if st.button("Log out", use_container_width=True):
        # Clear all session state
        st.session_state.clear()
        st.rerun()


