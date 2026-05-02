import concurrent.futures
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sensai.util import logging
from tqdm import tqdm

from serena.project import Project
from solidlsp.language_servers.clangd_language_server import ClangdLanguageServer
from solidlsp.ls_config import Language

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]
SerialIndexer = Callable[[Any, list[str]], "IndexingResult"]
ParallelIndexer = Callable[[Any, list[str], int, str | None, float], "IndexingResult"]
MaxWorkersProvider = Callable[[Any], int]


@dataclass
class IndexingResult:
    indexed_count: int = 0
    failed_files: list[str] = field(default_factory=list)
    exceptions: list[Exception] = field(default_factory=list)


@dataclass
class ProjectIndexingResult(IndexingResult):
    language_file_counts: dict[Language, int] = field(default_factory=dict)


_parallel_index_output_lock = threading.Lock()


def _emit_progress(message: str, progress_callback: ProgressCallback | None = None) -> None:
    log.info(message)
    if progress_callback is not None:
        progress_callback(message)


def _append_parallel_index_request_log(log_file_path: str, log_line: str) -> None:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with _parallel_index_output_lock:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{log_line}\n")


def _merge_indexing_result(target: IndexingResult, source: IndexingResult) -> None:
    target.indexed_count += source.indexed_count
    target.failed_files.extend(source.failed_files)
    target.exceptions.extend(source.exceptions)


def _clangd_index_max_workers(ls: Any) -> int:
    custom_settings = getattr(ls, "_custom_settings", None)
    if custom_settings is not None and hasattr(custom_settings, "get"):
        value = custom_settings.get("index_parallelism")
        if isinstance(value, int):
            return max(1, value)
    return 1


def _should_report_progress(completed_count: int, total_count: int) -> bool:
    if completed_count == 1 or completed_count == total_count:
        return True
    progress_interval = max(1, total_count // 100)
    return completed_count % progress_interval == 0


def _index_language_serial(
    ls: Any,
    files: list[str],
    *,
    use_tqdm: bool,
    progress_callback: ProgressCallback | None,
) -> IndexingResult:
    result = IndexingResult()
    total_count = len(files)
    _emit_progress(f"Pre-indexing[{ls.language.value}] started for {total_count} file(s).", progress_callback)
    file_iterator = tqdm(files, desc=f"Indexing[{ls.language.value}]") if use_tqdm else files

    for completed_count, file_path in enumerate(file_iterator, start=1):
        try:
            ls.request_document_symbols(file_path)
            result.indexed_count += 1
            if _should_report_progress(completed_count, total_count):
                _emit_progress(f"Pre-indexing[{ls.language.value}] {completed_count}/{total_count}: {file_path}", progress_callback)
        except Exception as e:
            log.error("Failed to pre-index %s, continuing.", file_path, exc_info=e)
            result.failed_files.append(file_path)
            result.exceptions.append(e)
    return result


def _index_language_parallel(
    ls: Any,
    files: list[str],
    max_workers: int,
    *,
    request_log_file_path: str | None,
    use_tqdm: bool,
    progress_callback: ProgressCallback | None,
    cache_save_interval_seconds: float = 30,
) -> IndexingResult:
    result = IndexingResult()
    total_count = len(files)
    _emit_progress(
        f"Pre-indexing[{ls.language.value}:parallel] started for {total_count} file(s) with {max_workers} worker(s).",
        progress_callback,
    )

    def work(file_path: str) -> tuple[str, Exception | None]:
        try:
            if request_log_file_path is not None:
                _append_parallel_index_request_log(request_log_file_path, f"Requesting[{ls.language.value}] {file_path}")
            ls.request_document_symbols(file_path)
            return file_path, None
        except Exception as e:
            return file_path, e

    def save_cache(completed_count: int) -> None:
        ls.save_cache()
        _emit_progress(
            f"Pre-indexing[{ls.language.value}:parallel] cache saved after {completed_count}/{total_count} file(s).",
            progress_callback,
        )

    def process_completed_future(future: concurrent.futures.Future[tuple[str, Exception | None]], completed_count: int) -> int:
        file_path, err = future.result()
        completed_count += 1
        if err is None:
            result.indexed_count += 1
            if request_log_file_path is not None:
                _append_parallel_index_request_log(
                    request_log_file_path,
                    f"Indexed[{ls.language.value}] {completed_count}/{total_count} {file_path}",
                )
            if _should_report_progress(completed_count, total_count):
                _emit_progress(
                    f"Pre-indexing[{ls.language.value}:parallel] {completed_count}/{total_count}: {file_path}",
                    progress_callback,
                )
        else:
            log.error("Failed to pre-index %s, continuing.", file_path, exc_info=err)
            result.failed_files.append(file_path)
            result.exceptions.append(err)
            if request_log_file_path is not None:
                _append_parallel_index_request_log(
                    request_log_file_path,
                    f"Failed[{ls.language.value}] {completed_count}/{total_count} {file_path}",
                )

        return completed_count

    last_save_time = time.monotonic()
    file_iter = iter(files)
    pending_futures: set[concurrent.futures.Future[tuple[str, Exception | None]]] = set()
    completed_count = 0
    save_due = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"index-{ls.language.value}") as executor:
        progress_bar = tqdm(total=total_count, desc=f"Indexing[{ls.language.value}:parallel]") if use_tqdm else None

        def submit_until_capacity() -> None:
            while len(pending_futures) < max_workers:
                try:
                    file_path = next(file_iter)
                except StopIteration:
                    return
                pending_futures.add(executor.submit(work, file_path))

        try:
            submit_until_capacity()
            while pending_futures:
                done_futures, pending_futures = concurrent.futures.wait(
                    pending_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_futures:
                    completed_count = process_completed_future(future, completed_count)
                    if progress_bar is not None:
                        progress_bar.update(1)

                now = time.monotonic()
                if cache_save_interval_seconds >= 0 and now - last_save_time >= cache_save_interval_seconds:
                    save_due = True

                if save_due and not pending_futures:
                    save_cache(completed_count)
                    last_save_time = time.monotonic()
                    save_due = False

                if not save_due:
                    submit_until_capacity()
        finally:
            if progress_bar is not None:
                progress_bar.close()

    return result


def write_indexing_failures(result: IndexingResult, log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        for file, exception in zip(result.failed_files, result.exceptions, strict=True):
            f.write(f"{file}\n")
            f.write(f"{exception}\n")


def index_project(
    project: Project,
    *,
    use_tqdm: bool = False,
    progress_callback: ProgressCallback | None = None,
    failed_log_file: str | None = None,
    parallel_request_log_file: str | None = None,
    stop_language_servers: bool = False,
    cache_save_interval_seconds: float = 30,
    serial_indexer: SerialIndexer | None = None,
    parallel_indexer: ParallelIndexer | None = None,
    max_workers_provider: MaxWorkersProvider | None = None,
) -> ProjectIndexingResult:
    project_name = getattr(project, "project_name", "<unknown>")
    _emit_progress(f"Pre-indexing project '{project_name}' at {project.project_root}.", progress_callback)
    ls_mgr = getattr(project, "language_server_manager", None) or project.create_language_server_manager()
    try:
        if parallel_request_log_file is not None:
            os.makedirs(os.path.dirname(parallel_request_log_file), exist_ok=True)
            with open(parallel_request_log_file, "w", encoding="utf-8"):
                pass

        files = project.gather_source_files()
        _emit_progress(f"Pre-indexing discovered {len(files)} source file(s).", progress_callback)

        language_file_counts: defaultdict[Language, int] = defaultdict(int)
        files_by_ls_key: dict[int, tuple[Any, list[str]]] = {}
        for file_path in files:
            ls = ls_mgr.get_language_server(file_path)
            ls_key = id(ls)
            if ls_key not in files_by_ls_key:
                files_by_ls_key[ls_key] = (ls, [])
            files_by_ls_key[ls_key][1].append(file_path)

        aggregated_result = ProjectIndexingResult()
        last_save_time = time.monotonic()

        for ls, ls_files in files_by_ls_key.values():
            max_workers = (max_workers_provider or _clangd_index_max_workers)(ls)
            if isinstance(ls, ClangdLanguageServer) and max_workers > 1:
                result = (
                    parallel_indexer(ls, ls_files, max_workers, parallel_request_log_file, cache_save_interval_seconds)
                    if parallel_indexer is not None
                    else _index_language_parallel(
                        ls,
                        ls_files,
                        max_workers,
                        request_log_file_path=parallel_request_log_file,
                        use_tqdm=use_tqdm,
                        progress_callback=progress_callback,
                        cache_save_interval_seconds=cache_save_interval_seconds,
                    )
                )
                language_file_counts[ls.language] += result.indexed_count
                _merge_indexing_result(aggregated_result, result)
                ls.save_cache()
                _emit_progress(f"Pre-indexing[{ls.language.value}] cache saved.", progress_callback)
                continue

            result = (
                serial_indexer(ls, ls_files)
                if serial_indexer is not None
                else _index_language_serial(ls, ls_files, use_tqdm=use_tqdm, progress_callback=progress_callback)
            )
            language_file_counts[ls.language] += result.indexed_count
            _merge_indexing_result(aggregated_result, result)
            now = time.monotonic()
            if now - last_save_time >= cache_save_interval_seconds:
                ls_mgr.save_all_caches()
                last_save_time = now
                _emit_progress("Pre-indexing caches saved.", progress_callback)

        ls_mgr.save_all_caches()
        aggregated_result.language_file_counts = dict(language_file_counts)
        language_counts = {language.value: count for language, count in aggregated_result.language_file_counts.items()}
        _emit_progress(f"Pre-indexing completed. Indexed files per language: {language_counts}", progress_callback)

        if aggregated_result.failed_files and failed_log_file is not None:
            write_indexing_failures(aggregated_result, failed_log_file)
            _emit_progress(
                f"Pre-indexing failed for {len(aggregated_result.failed_files)} file(s); see {failed_log_file}", progress_callback
            )

        return aggregated_result
    finally:
        if stop_language_servers:
            shutdown = getattr(project, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                ls_mgr.stop_all()
