"""Tests for CLI project commands (create, index)."""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from serena.cli import ProjectCommands, TopLevelCommands, find_project_root
from serena.config.serena_config import ProjectConfig
from solidlsp.language_servers.clangd_language_server import ClangdLanguageServer
from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.settings import SolidLSPSettings

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        # if windows, wait a bit to avoid PermissionError on cleanup
        if os.name == "nt":
            time.sleep(0.2)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_project_dir_with_python_file():
    """Create a temporary directory with a Python file for testing."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a simple Python file so language detection works
        py_file = os.path.join(tmpdir, "test.py")
        with open(py_file, "w") as f:
            f.write("def hello():\n    pass\n")
        yield tmpdir
    finally:
        # if windows, wait a bit to avoid PermissionError on cleanup
        if os.name == "nt":
            time.sleep(0.2)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing Click commands."""
    return CliRunner()


class TestProjectCreate:
    """Tests for 'project create' command."""

    def test_create_basic_with_language(self, cli_runner, temp_project_dir):
        """Test basic project creation with explicit language."""
        result = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--language", "python"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output
        assert "python" in result.output.lower()

        # Verify project.yml was created
        yml_path = os.path.join(temp_project_dir, ".serena", "project.yml")
        assert os.path.exists(yml_path), f"project.yml not found at {yml_path}"

    def test_create_auto_detect_language(self, cli_runner, temp_project_dir_with_python_file):
        """Test project creation with auto-detected language."""
        result = cli_runner.invoke(ProjectCommands.create, [temp_project_dir_with_python_file])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output
        assert "python" in result.output.lower()

        # Verify project.yml was created
        yml_path = os.path.join(temp_project_dir_with_python_file, ".serena", "project.yml")
        assert os.path.exists(yml_path)

    def test_create_with_name(self, cli_runner, temp_project_dir):
        """Test project creation with custom name and explicit language."""
        result = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--name", "my-custom-project", "--language", "python"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output

        # Verify project.yml was created
        yml_path = os.path.join(temp_project_dir, ".serena", "project.yml")
        assert os.path.exists(yml_path)

    def test_create_with_language(self, cli_runner, temp_project_dir):
        """Test project creation with specified language."""
        result = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--language", "python"])
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output
        assert "python" in result.output.lower()

    def test_create_with_multiple_languages(self, cli_runner, temp_project_dir):
        """Test project creation with multiple languages."""
        result = cli_runner.invoke(
            ProjectCommands.create,
            [temp_project_dir, "--language", "python", "--language", "typescript"],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output

    def test_create_with_invalid_language(self, cli_runner, temp_project_dir):
        """Test project creation with invalid language raises error."""
        result = cli_runner.invoke(
            ProjectCommands.create,
            [temp_project_dir, "--language", "invalid-lang"],
        )
        assert result.exit_code != 0, "Should fail with invalid language"
        assert "Unknown language" in result.output or "invalid-lang" in result.output

    def test_create_already_exists(self, cli_runner, temp_project_dir):
        """Test that creating a project twice fails gracefully."""
        # Create once with explicit language
        result1 = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--language", "python"])
        assert result1.exit_code == 0

        # Try to create again - should fail gracefully
        result2 = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--language", "python"])
        assert result2.exit_code != 0, "Should fail when project.yml already exists"
        assert "already exists" in result2.output.lower()
        assert "Error:" in result2.output  # Should be user-friendly error

    def test_create_with_index_flag(self, cli_runner, temp_project_dir_with_python_file):
        """Test project creation with --index flag performs indexing."""
        result = cli_runner.invoke(
            ProjectCommands.create,
            [temp_project_dir_with_python_file, "--language", "python", "--index", "--log-level", "ERROR", "--timeout", "5"],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated project" in result.output
        assert "Indexing project" in result.output

        # Verify project.yml was created
        yml_path = os.path.join(temp_project_dir_with_python_file, ".serena", "project.yml")
        assert os.path.exists(yml_path)

        # Verify cache directory was created (proof of indexing)
        cache_dir = os.path.join(temp_project_dir_with_python_file, ".serena", "cache")
        assert os.path.exists(cache_dir), "Cache directory should exist after indexing"

    def test_create_without_index_flag(self, cli_runner, temp_project_dir):
        """Test that project creation without --index does NOT perform indexing."""
        result = cli_runner.invoke(ProjectCommands.create, [temp_project_dir, "--language", "python"])
        assert result.exit_code == 0
        assert "Generated project" in result.output
        assert "Indexing" not in result.output

        # Verify cache directory was NOT created
        cache_dir = os.path.join(temp_project_dir, ".serena", "cache")
        assert not os.path.exists(cache_dir), "Cache directory should not exist without --index"


class TestProjectIndex:
    """Tests for 'project index' command."""

    def test_clangd_index_max_workers_reads_custom_ls_settings(self):
        clangd_ls = object.__new__(ClangdLanguageServer)
        clangd_ls._custom_settings = SolidLSPSettings.CustomLSSettings({"index_parallelism": 7})

        assert ProjectCommands._clangd_index_max_workers(clangd_ls) == 7

    def test_index_auto_creates_project_with_files(self, cli_runner, temp_project_dir_with_python_file):
        """Test that index command auto-creates project.yml if it doesn't exist (with source files)."""
        result = cli_runner.invoke(ProjectCommands.index, [temp_project_dir_with_python_file, "--log-level", "ERROR", "--timeout", "5"])
        # Should succeed and perform indexing
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Auto-creating" in result.output or "Indexing" in result.output

        # Verify project.yml was auto-created
        yml_path = os.path.join(temp_project_dir_with_python_file, ".serena", "project.yml")
        assert os.path.exists(yml_path), "project.yml should be auto-created"

    def test_index_with_explicit_language(self, cli_runner, temp_project_dir):
        """Test index with explicit --language for empty directory."""
        result = cli_runner.invoke(
            ProjectCommands.index,
            [temp_project_dir, "--language", "python", "--log-level", "ERROR", "--timeout", "5"],
        )
        # Should succeed even without source files if language is explicit
        assert result.exit_code == 0, f"Command failed: {result.output}"

        yml_path = os.path.join(temp_project_dir, ".serena", "project.yml")
        assert os.path.exists(yml_path)

    def test_index_with_language_auto_creates(self, cli_runner, temp_project_dir):
        """Test index with --language option for auto-creation."""
        result = cli_runner.invoke(
            ProjectCommands.index,
            [temp_project_dir, "--language", "python", "--log-level", "ERROR"],
        )
        assert result.exit_code == 0 or "Indexing" in result.output

        yml_path = os.path.join(temp_project_dir, ".serena", "project.yml")
        assert os.path.exists(yml_path)

    def test_index_is_equivalent_to_create_with_index(self, cli_runner, temp_project_dir_with_python_file):
        """Test that 'index' behaves like 'create --index' for new projects."""
        # Use manual temp directory creation with Windows-safe cleanup
        # to avoid PermissionError on Windows CI when language servers hold file locks
        dir1 = tempfile.mkdtemp()
        dir2 = tempfile.mkdtemp()

        try:
            # Setup both directories with same file
            for d in [dir1, dir2]:
                with open(os.path.join(d, "test.py"), "w") as f:
                    f.write("def hello():\n    pass\n")

            # Run 'create --index' on dir1
            result1 = cli_runner.invoke(
                ProjectCommands.create, [dir1, "--language", "python", "--index", "--log-level", "ERROR", "--timeout", "5"]
            )

            # Run 'index' on dir2
            result2 = cli_runner.invoke(ProjectCommands.index, [dir2, "--language", "python", "--log-level", "ERROR", "--timeout", "5"])

            # Both should succeed
            assert result1.exit_code == 0, f"create --index failed: {result1.output}"
            assert result2.exit_code == 0, f"index failed: {result2.output}"

            # Both should create project.yml
            assert os.path.exists(os.path.join(dir1, ".serena", "project.yml"))
            assert os.path.exists(os.path.join(dir2, ".serena", "project.yml"))

            # Both should create cache (proof of indexing)
            assert os.path.exists(os.path.join(dir1, ".serena", "cache"))
            assert os.path.exists(os.path.join(dir2, ".serena", "cache"))
        finally:
            # Windows-safe cleanup: wait for file handles to be released
            if os.name == "nt":
                time.sleep(0.2)
            # Use ignore_errors to handle lingering file locks on Windows
            shutil.rmtree(dir1, ignore_errors=True)
            shutil.rmtree(dir2, ignore_errors=True)

    def test_index_project_uses_parallel_helper_only_for_clangd(self, monkeypatch):
        clangd_ls = object.__new__(ClangdLanguageServer)
        clangd_ls.language = Language.CPP
        clangd_ls._custom_settings = SolidLSPSettings.CustomLSSettings({"index_parallelism": 2})
        clangd_ls.save_cache = lambda: None

        fake_proj = SimpleNamespace(
            project_root="/tmp/project",
            gather_source_files=lambda: ["a.cpp", "b.cpp"],
            create_language_server_manager=lambda: SimpleNamespace(
                get_language_server=lambda _path: clangd_ls,
                save_all_caches=lambda: None,
                stop_all=lambda: None,
            ),
        )
        fake_registered_project = SimpleNamespace(get_project_instance=lambda serena_config: fake_proj)
        fake_config = SimpleNamespace()

        calls: list[str] = []

        def fake_parallel(ls, files, max_workers, request_log_file_path=None, cache_save_interval_seconds=30):
            calls.append(f"parallel:{max_workers}")
            return SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[])

        def fake_serial(ls, files):
            calls.append("serial")
            return SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[])

        monkeypatch.setattr("serena.cli.SerenaConfig.from_config_file", lambda: fake_config)
        monkeypatch.setattr("serena.cli.click.echo", lambda _msg: None)
        monkeypatch.setattr("serena.cli.ProjectCommands._index_language_parallel", fake_parallel)
        monkeypatch.setattr("serena.cli.ProjectCommands._index_language_serial", fake_serial)

        ProjectCommands._index_project(fake_registered_project, "ERROR", timeout=5)

        assert calls == ["parallel:2"]

    def test_index_project_keeps_non_clangd_serial(self, monkeypatch):
        python_ls = SimpleNamespace(language=Language.PYTHON, _custom_settings={"index_parallelism": 8}, save_cache=lambda: None)

        fake_proj = SimpleNamespace(
            project_root="/tmp/project",
            gather_source_files=lambda: ["a.py", "b.py"],
            create_language_server_manager=lambda: SimpleNamespace(
                get_language_server=lambda _path: python_ls,
                save_all_caches=lambda: None,
                stop_all=lambda: None,
            ),
        )
        fake_registered_project = SimpleNamespace(get_project_instance=lambda serena_config: fake_proj)
        fake_config = SimpleNamespace()

        calls: list[str] = []

        def fake_parallel(ls, files, max_workers, request_log_file_path=None, cache_save_interval_seconds=30):
            calls.append(f"parallel:{max_workers}")
            return SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[])

        def fake_serial(ls, files):
            calls.append("serial")
            return SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[])

        monkeypatch.setattr("serena.cli.SerenaConfig.from_config_file", lambda: fake_config)
        monkeypatch.setattr("serena.cli.click.echo", lambda _msg: None)
        monkeypatch.setattr("serena.cli.ProjectCommands._index_language_parallel", fake_parallel)
        monkeypatch.setattr("serena.cli.ProjectCommands._index_language_serial", fake_serial)

        ProjectCommands._index_project(fake_registered_project, "ERROR", timeout=5)

        assert calls == ["serial"]

    def test_index_project_clears_requested_files_log_at_start(self, monkeypatch, temp_project_dir):
        request_log = os.path.join(temp_project_dir, ".serena", "logs", "indexing_requested_files.txt")
        os.makedirs(os.path.dirname(request_log), exist_ok=True)
        with open(request_log, "w", encoding="utf-8") as f:
            f.write("stale log line\n")

        python_ls = SimpleNamespace(language=Language.PYTHON, _custom_settings={"index_parallelism": 1}, save_cache=lambda: None)
        fake_proj = SimpleNamespace(
            project_root=temp_project_dir,
            gather_source_files=lambda: ["a.py"],
            create_language_server_manager=lambda: SimpleNamespace(
                get_language_server=lambda _path: python_ls,
                save_all_caches=lambda: None,
                stop_all=lambda: None,
            ),
        )
        fake_registered_project = SimpleNamespace(get_project_instance=lambda serena_config: fake_proj)
        fake_config = SimpleNamespace()

        monkeypatch.setattr("serena.cli.SerenaConfig.from_config_file", lambda: fake_config)
        monkeypatch.setattr("serena.cli.click.echo", lambda _msg: None)
        monkeypatch.setattr(
            "serena.cli.ProjectCommands._index_language_serial",
            lambda ls, files: SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[]),
        )

        ProjectCommands._index_project(fake_registered_project, "INFO", timeout=5)

        with open(request_log, encoding="utf-8") as f:
            assert f.read() == ""

    def test_index_project_does_not_enable_requested_files_log_for_warning(self, monkeypatch, temp_project_dir):
        request_log = os.path.join(temp_project_dir, ".serena", "logs", "indexing_requested_files.txt")
        os.makedirs(os.path.dirname(request_log), exist_ok=True)
        with open(request_log, "w", encoding="utf-8") as f:
            f.write("stale log line\n")

        python_ls = SimpleNamespace(language=Language.PYTHON, _custom_settings={"index_parallelism": 1}, save_cache=lambda: None)
        fake_proj = SimpleNamespace(
            project_root=temp_project_dir,
            gather_source_files=lambda: ["a.py"],
            create_language_server_manager=lambda: SimpleNamespace(
                get_language_server=lambda _path: python_ls,
                save_all_caches=lambda: None,
                stop_all=lambda: None,
            ),
        )
        fake_registered_project = SimpleNamespace(get_project_instance=lambda serena_config: fake_proj)
        fake_config = SimpleNamespace()

        monkeypatch.setattr("serena.cli.SerenaConfig.from_config_file", lambda: fake_config)
        monkeypatch.setattr("serena.cli.click.echo", lambda _msg: None)
        monkeypatch.setattr(
            "serena.cli.ProjectCommands._index_language_serial",
            lambda ls, files: SimpleNamespace(indexed_count=len(files), failed_files=[], exceptions=[]),
        )

        ProjectCommands._index_project(fake_registered_project, "WARNING", timeout=5)

        with open(request_log, encoding="utf-8") as f:
            assert f.read() == "stale log line\n"

    def test_index_language_parallel_logs_requested_files_and_progress(self, monkeypatch, temp_project_dir):
        ls = SimpleNamespace(language=Language.CPP)
        requested_files: list[str] = []
        request_log = os.path.join(temp_project_dir, ".serena", "logs", "indexing_requested_files.txt")

        def request_document_symbols(file_path: str) -> None:
            requested_files.append(file_path)

        ls.request_document_symbols = request_document_symbols

        result = ProjectCommands._index_language_parallel(ls, ["a.cpp", "b.cpp"], max_workers=2, request_log_file_path=request_log)

        assert result.indexed_count == 2
        assert result.failed_files == []
        assert sorted(requested_files) == ["a.cpp", "b.cpp"]
        with open(request_log, encoding="utf-8") as f:
            logged_lines = [line.strip() for line in f.readlines()]
        assert "Requesting[cpp] a.cpp" in logged_lines
        assert "Requesting[cpp] b.cpp" in logged_lines
        indexed_lines = [line for line in logged_lines if line.startswith("Indexed[cpp] ")]
        assert len(indexed_lines) == 2
        assert any(" 1/2 " in line for line in indexed_lines)
        assert any(" 2/2 " in line for line in indexed_lines)
        assert any(line.endswith(" a.cpp") for line in indexed_lines)
        assert any(line.endswith(" b.cpp") for line in indexed_lines)

    def test_index_language_parallel_saves_cache_during_long_runs(self):
        files = ["a.cpp", "b.cpp", "c.cpp"]
        requested_files: list[str] = []
        save_points: list[int] = []
        ls = SimpleNamespace(language=Language.CPP)

        def request_document_symbols(file_path: str) -> None:
            requested_files.append(file_path)

        def save_cache() -> None:
            save_points.append(len(requested_files))

        ls.request_document_symbols = request_document_symbols
        ls.save_cache = save_cache

        result = ProjectCommands._index_language_parallel(
            ls,
            files,
            max_workers=1,
            request_log_file_path=None,
            cache_save_interval_seconds=0,
        )

        assert result.indexed_count == len(files)
        assert save_points
        assert min(save_points) < len(files)


class DummySolidLanguageServer(SolidLanguageServer):
    def _start_server(self) -> None:
        raise NotImplementedError


class TestSolidLanguageServerCacheSaveBarrier:
    def test_save_cache_waits_for_active_index_requests(self):
        ls = object.__new__(DummySolidLanguageServer)
        ls._cache_save_lock = threading.RLock()
        ls._index_activity_lock = threading.RLock()
        ls._index_activity_cond = threading.Condition(ls._index_activity_lock)
        ls._active_index_requests = 0

        calls: list[str] = []
        ls._save_raw_document_symbols_cache = lambda: calls.append("raw")
        ls._save_document_symbols_cache = lambda: calls.append("document")

        request_started = threading.Event()
        release_request = threading.Event()
        save_completed = threading.Event()

        def hold_request() -> None:
            with ls._track_index_request():
                request_started.set()
                release_request.wait(timeout=5)

        def save_cache() -> None:
            ls.save_cache()
            save_completed.set()

        request_thread = threading.Thread(target=hold_request)
        request_thread.start()
        assert request_started.wait(timeout=5)

        save_thread = threading.Thread(target=save_cache)
        save_thread.start()

        time.sleep(0.1)
        assert not save_completed.is_set()

        release_request.set()
        request_thread.join(timeout=5)
        save_thread.join(timeout=5)

        assert save_completed.is_set()
        assert calls == ["raw", "document"]


class TestProjectCreateHelper:
    """Tests for _create_project helper method."""

    def test_create_project_helper_returns_config(self, temp_project_dir):
        """Test that _create_project returns a ProjectConfig with explicit language."""
        config = ProjectCommands._create_project(temp_project_dir, "test-project", ("python",)).project_config
        assert isinstance(config, ProjectConfig)
        assert config.project_name == "test-project"

    def test_create_project_helper_with_auto_detect(self, temp_project_dir_with_python_file):
        """Test _create_project with auto-detected language."""
        config = ProjectCommands._create_project(temp_project_dir_with_python_file, "my-project", ()).project_config
        assert isinstance(config, ProjectConfig)
        assert config.project_name == "my-project"
        assert len(config.languages) >= 1

    def test_create_project_helper_with_languages(self, temp_project_dir):
        """Test _create_project with language specification."""
        config = ProjectCommands._create_project(temp_project_dir, None, ("python", "typescript")).project_config
        assert isinstance(config, ProjectConfig)
        assert len(config.languages) >= 1

    def test_create_project_helper_file_exists_error(self, temp_project_dir):
        """Test _create_project raises error if project.yml exists."""
        # Create project first with explicit language
        ProjectCommands._create_project(temp_project_dir, None, ("python",))

        # Try to create again - should raise FileExistsError
        with pytest.raises(FileExistsError):
            ProjectCommands._create_project(temp_project_dir, None, ("python",))


class TestFindProjectRoot:
    """Tests for find_project_root helper with virtual chroot boundary."""

    def test_finds_serena_from_subdirectory(self, temp_project_dir):
        """Test that .serena/project.yml is found when searching from a subdirectory."""
        serena_dir = os.path.join(temp_project_dir, ".serena")
        os.makedirs(serena_dir)
        Path(os.path.join(serena_dir, "project.yml")).touch()
        subdir = os.path.join(temp_project_dir, "src", "nested")
        os.makedirs(subdir)

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            result = find_project_root(root=temp_project_dir)
            assert result is not None
            assert os.path.samefile(result, temp_project_dir)
        finally:
            os.chdir(original_cwd)

    def test_serena_preferred_over_git(self, temp_project_dir):
        """Test that .serena/project.yml takes priority over .git at the same level."""
        serena_dir = os.path.join(temp_project_dir, ".serena")
        os.makedirs(serena_dir)
        Path(os.path.join(serena_dir, "project.yml")).touch()
        os.makedirs(os.path.join(temp_project_dir, ".git"))

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_project_dir)
            result = find_project_root(root=temp_project_dir)
            assert result is not None
            assert os.path.isdir(os.path.join(result, ".serena"))
            assert os.path.samefile(result, temp_project_dir)
        finally:
            os.chdir(original_cwd)

    def test_git_used_as_fallback(self, temp_project_dir):
        """Test that .git is found when no .serena exists."""
        os.makedirs(os.path.join(temp_project_dir, ".git"))
        subdir = os.path.join(temp_project_dir, "src")
        os.makedirs(subdir)

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            result = find_project_root(root=temp_project_dir)
            assert result is not None
            assert os.path.samefile(result, temp_project_dir)
        finally:
            os.chdir(original_cwd)

    def test_falls_back_to_none_when_no_markers(self, temp_project_dir):
        """Test falls back to None when no markers exist within boundary."""
        subdir = os.path.join(temp_project_dir, "src")
        os.makedirs(subdir)

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            result = find_project_root(root=temp_project_dir)
            assert result is None
        finally:
            os.chdir(original_cwd)


class TestProjectFromCwdMutualExclusivity:
    """Tests for --project-from-cwd mutual exclusivity."""

    def test_project_from_cwd_with_project_flag_fails(self, cli_runner):
        """Test that --project-from-cwd with --project raises error."""
        result = cli_runner.invoke(
            TopLevelCommands.start_mcp_server,
            ["--project-from-cwd", "--project", "/some/path"],
        )
        assert result.exit_code != 0
        assert "cannot be used with" in result.output


if __name__ == "__main__":
    # For manual testing, you can run this file directly:
    # uv run pytest test/serena/test_cli_project_commands.py -v
    pytest.main([__file__, "-v"])
