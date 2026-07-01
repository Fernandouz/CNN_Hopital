"""Encarts de métriques Streamlit avec espacement stable."""

import html

import streamlit as st


def metric_card(label: str, value, help_text: str = ""):
    """Affiche une métrique dans un encart HTML paddé."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_help = html.escape(str(help_text))
    help_html = (
        f"<div style='color:#8FB3A6;font-size:0.78rem;line-height:1.3;margin-top:0.45rem;'>{safe_help}</div>"
        if help_text
        else ""
    )

    st.markdown(
        f"""
        <div style="
            background-color:#142420;
            border:1px solid #2D4A40;
            border-radius:8px;
            padding:1rem 1.1rem;
            min-height:104px;
        ">
            <div style="color:#8FB3A6;font-size:0.86rem;line-height:1.25;margin-bottom:0.45rem;">
                {safe_label}
            </div>
            <div style="color:#1D9E75;font-size:1.9rem;font-weight:700;line-height:1.15;">
                {safe_value}
            </div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
