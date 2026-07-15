import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "models"
    / "trained_model"
    / "notebooks"
    / "train_weights.ipynb"
)


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_notebook_supports_synthetic_source():
    source = notebook_source()
    assert '"synthetic"' in source
    assert "build_synthetic_dataset.py" in source
    assert 'return output_dir / "train.jsonl"' in source


def test_notebook_stops_on_invalid_synthetic_manifest():
    source = notebook_source()
    assert 'manifest["accepted"] != manifest["requested"]' in source
    assert 'manifest["rejected"]' in source
    assert 'manifest["family_overlap"]' in source


def test_notebook_smoke_uses_one_epoch():
    source = notebook_source()
    assert "SYNTHETIC_SMOKE_COUNT = 100" in source
    assert "config.num_epochs = 1.0" in source


def test_notebook_uses_held_out_synthetic_eval_file():
    source = notebook_source()
    assert 'config.eval_data_path = str(output_dir / "eval.jsonl")' in source
    assert "Configured train and eval files contain duplicate examples." in source


def test_all_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")


def test_notebook_installs_pascal_compatible_torch():
    source = notebook_source()
    assert '"torch==2.5.1"' in source
    assert '"torchvision==0.20.1"' in source
    assert '"https://download.pytorch.org/whl/cu121"' in source
