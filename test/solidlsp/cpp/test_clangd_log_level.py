import logging

import pytest

from solidlsp.language_servers.clangd_language_server import ClangdLanguageServer


@pytest.mark.cpp
class TestClangdLogLevel:
    def test_compiler_command_with_error_flags_is_info(self) -> None:
        line = (
            '"C:\\Program Files\\LLVM\\bin\\clang-cl.exe" --driver-mode=cl '
            "/I . -Wno-error=deprecated-declarations -Wno-implicit-exception-spec-mismatch "
            '-- "D:\\UE\\Project\\VehicleTest_5_7\\Source\\Game.cpp"'
        )

        assert ClangdLanguageServer._determine_log_level(line) == logging.INFO

    def test_clangd_prefixed_error_remains_error(self) -> None:
        line = "E[22:50:50.292] failed to parse compile command"

        assert ClangdLanguageServer._determine_log_level(line) == logging.ERROR

    def test_clangd_prefixed_warning_is_warning(self) -> None:
        line = "W[22:50:50.292] config warning"

        assert ClangdLanguageServer._determine_log_level(line) == logging.WARNING
