from pathlib import Path


def test_ui_exposes_1024_default_and_2048_maximum():
    html = Path("spaces/hf_cpu/ui/index.html").read_text(encoding="utf-8")
    assert 'value="1024" selected' in html
    assert 'value="2048"' in html


def test_ui_only_shows_continue_for_length_stops():
    source = Path("spaces/hf_cpu/ui/app.js").read_text(encoding="utf-8")
    assert 'done.finish_reason === "length"' in source
    assert "continueButton.hidden = !done.can_continue" in source


def test_ui_renders_model_output_as_text_not_html():
    source = Path("spaces/hf_cpu/ui/app.js").read_text(encoding="utf-8")
    assert "assistantNode.textContent += data.text" in source
    assert "innerHTML" not in source
