from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar, cast
from unittest.mock import Mock, patch

from serena.dashboard import SerenaDashboardAPI, SerenaDashboardViewer
from solidlsp.ls_config import Language

if TYPE_CHECKING:
    from serena.agent import SerenaAgent
    from serena.util.logging import MemoryLogHandler

T = TypeVar("T")


class _DummyMemoryLogHandler:
    def get_log_messages(self, from_idx: int = 0) -> SimpleNamespace:  # pragma: no cover - simple stub
        del from_idx
        return SimpleNamespace(messages=[], max_idx=-1)

    def clear_log_messages(self) -> None:  # pragma: no cover - simple stub
        pass


class _DummyAgent:
    def __init__(self, project: SimpleNamespace | None) -> None:
        self._project = project

    def execute_task(self, func: Callable[[], T], *, logged: bool | None = None, name: str | None = None) -> T:
        del logged, name
        return func()

    def get_active_project(self) -> SimpleNamespace | None:
        return self._project


def _make_dashboard(project_languages: list[Language] | None) -> SerenaDashboardAPI:
    project = None
    if project_languages is not None:
        project = SimpleNamespace(project_config=SimpleNamespace(languages=project_languages))
    agent = _DummyAgent(project)
    return SerenaDashboardAPI(
        memory_log_handler=cast("MemoryLogHandler", _DummyMemoryLogHandler()),
        tool_names=[],
        agent=cast("SerenaAgent", agent),
        tool_usage_stats=None,
    )


def test_available_languages_include_experimental_when_no_active_project():
    dashboard = _make_dashboard(project_languages=None)
    response = dashboard._get_available_languages()
    expected = sorted(lang.value for lang in Language.iter_all(include_experimental=True))
    assert response.languages == expected


def test_available_languages_exclude_project_languages():
    dashboard = _make_dashboard(project_languages=[Language.PYTHON, Language.MARKDOWN])
    response = dashboard._get_available_languages()
    available = set(response.languages)
    assert Language.PYTHON.value not in available
    assert Language.MARKDOWN.value not in available
    # ensure experimental languages remain available for selection
    assert Language.ANSIBLE.value in available


def test_dashboard_viewer_quit_requests_agent_shutdown():
    viewer = SerenaDashboardViewer.__new__(SerenaDashboardViewer)
    viewer._url = "http://127.0.0.1:24282/dashboard/index.html"

    with patch("serena.dashboard.urllib.request.urlopen") as urlopen, patch("serena.dashboard.urllib.request.Request") as request:
        request.return_value = Mock()

        viewer._request_agent_shutdown()

    request.assert_called_once_with("http://127.0.0.1:24282/shutdown", method="PUT")
    urlopen.assert_called_once_with(request.return_value, timeout=2)
