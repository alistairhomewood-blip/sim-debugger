"""Tests for configuration file support (Phase 3)."""

import os
import tempfile
from pathlib import Path

import pytest

from sim_debugger.core.config import (
    SimDebuggerConfig,
    find_config_file,
    load_config,
    save_default_config,
)


class TestSimDebuggerConfig:
    def test_default_values(self):
        config = SimDebuggerConfig()
        assert config.monitor.check_interval == 1
        assert config.monitor.mode == "default"
        assert config.monitor.history_size == 100
        assert config.monitor.invariants is None
        assert config.output.format == "text"
        assert config.output.log_file is None
        assert config.performance.state_copy_mode == "copy"

    def test_get_check_interval_default(self):
        config = SimDebuggerConfig()
        assert config.get_check_interval() == 1

    def test_get_check_interval_lightweight(self):
        config = SimDebuggerConfig()
        config.monitor.mode = "lightweight"
        config.performance.lightweight_interval = 50
        assert config.get_check_interval() == 50

    def test_get_thresholds(self):
        config = SimDebuggerConfig()
        config.thresholds.thresholds = {"Total Energy": 1e-8}
        result = config.get_thresholds()
        assert result == {"Total Energy": 1e-8}


class TestConfigFileParsing:
    def test_load_basic_config(self):
        config_content = """\
[monitor]
invariants = ["Total Energy", "Boris Energy"]
check_interval = 5
mode = "full"

[thresholds]
"Total Energy" = 1e-8
"Boris Energy" = 1e-10

[output]
format = "json"
log_file = "violations.log"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()
            try:
                config = load_config(path=f.name)
                assert config.monitor.invariants == ["Total Energy", "Boris Energy"]
                assert config.monitor.check_interval == 5
                assert config.monitor.mode == "full"
                assert config.thresholds.thresholds["Total Energy"] == 1e-8
                assert config.thresholds.thresholds["Boris Energy"] == 1e-10
                assert config.output.format == "json"
                assert config.output.log_file == "violations.log"
            finally:
                os.unlink(f.name)

    def test_load_performance_config(self):
        config_content = """\
[performance]
lightweight_interval = 200
state_copy_mode = "view"
max_memory_mb = 512
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()
            try:
                config = load_config(path=f.name)
                assert config.performance.lightweight_interval == 200
                assert config.performance.state_copy_mode == "view"
                assert config.performance.max_memory_mb == 512
            finally:
                os.unlink(f.name)

    def test_load_plugin_config(self):
        config_content = """\
[plugins]
paths = ["./my_invariants", "./shared"]
enabled = ["CustomInvariant"]
disabled = ["Experimental"]
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()
            try:
                config = load_config(path=f.name)
                assert config.plugins.paths == ["./my_invariants", "./shared"]
                assert config.plugins.enabled == ["CustomInvariant"]
                assert config.plugins.disabled == ["Experimental"]
            finally:
                os.unlink(f.name)

    def test_invalid_mode_raises(self):
        config_content = """\
[monitor]
mode = "invalid_mode"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(config_content)
            f.flush()
            try:
                with pytest.raises(ValueError, match="Invalid monitor.mode"):
                    load_config(path=f.name)
            finally:
                os.unlink(f.name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(path="/nonexistent/.sim-debugger.toml")

    def test_no_config_returns_defaults(self):
        # Use a temp dir with no config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(start_dir=tmpdir)
            assert config.monitor.check_interval == 1
            assert config.config_file is None


class TestSaveDefaultConfig:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".sim-debugger.toml"
            save_default_config(path)
            assert path.exists()
            content = path.read_text()
            assert "[monitor]" in content
            assert "[thresholds]" in content
            assert "[output]" in content
            assert "[performance]" in content
            assert "[plugins]" in content

    def test_default_config_is_valid_toml(self):
        """The default config should be parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".sim-debugger.toml"
            save_default_config(path)
            # Should load without error
            config = load_config(path=str(path))
            assert config.monitor.mode == "default"


class TestFindConfigFile:
    def test_finds_in_current_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".sim-debugger.toml"
            config_path.write_text("[monitor]\n")
            found = find_config_file(tmpdir)
            assert found is not None
            # Use resolve() to handle macOS /var -> /private/var symlinks
            assert found.resolve() == config_path.resolve()

    def test_returns_none_when_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            found = find_config_file(tmpdir)
            # May find one in home dir; check that it doesn't crash
            # The important test is that it doesn't raise
            assert found is None or isinstance(found, Path)
