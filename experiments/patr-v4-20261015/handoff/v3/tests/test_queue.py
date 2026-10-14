"""The v2 repair checkout must never launch the inherited formal training queue."""
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import run_queue


def test_formal_queue_refuses_and_does_not_write_or_launch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Queue launched a process'))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('Queue launched a command'))
    with pytest.raises(SystemExit, match='Formal queue disabled in v2'):
        run_queue.main()
    assert list(tmp_path.iterdir()) == []


def test_queue_cli_is_safe_without_training_dependencies(tmp_path):
    script = Path(run_queue.__file__).resolve()
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert '--config configs/SHORT_CONFIG.json' in result.stderr
    assert '--resume only' in result.stderr
    assert not list(tmp_path.iterdir())
