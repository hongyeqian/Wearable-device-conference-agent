"""
MeetingWatcher - File system watcher for automatic meeting hot reload.

Uses watchdog to monitor the data directory for new `con*` folders.
When a new folder is detected and all 4 required files are present,
triggers an incremental sync callback to index the meeting without
restarting the system.

Required files per con* folder:
  - metaData*.json  (meeting metadata)
  - data*.md        (transcript)
  - meetLevel*.md   (meeting-level summary)
  - summary*.md     (summary embeddings)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Set, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirCreatedEvent

logger = logging.getLogger(__name__)

# Glob patterns for the 4 required files in each con* directory
_REQUIRED_FILE_PATTERNS = [
    "metaData*.json",
    "data*.md",
    "meetLevel*.md",
    "summary*.md",
]

# Debounce: wait this many seconds after detecting a new directory
# before checking file completeness (allows time for all files to be copied)
_DEBOUNCE_SECONDS = 10

# Retry: if files are incomplete after debounce, retry every N seconds
_RETRY_INTERVAL_SECONDS = 5

# Maximum number of retries before giving up on a directory
_MAX_RETRIES = 6


def check_completeness(con_dir: Path) -> bool:
    """
    Check whether a con* directory contains all 4 required files.

    Args:
        con_dir: Path to the con* directory.

    Returns:
        True if all 4 required file patterns are matched, False otherwise.
    """
    for pattern in _REQUIRED_FILE_PATTERNS:
        matches = list(con_dir.glob(pattern))
        if not matches:
            return False
    return True


class _NewConDirHandler(FileSystemEventHandler):
    """
    Watchdog event handler that detects new con* directories
    created inside the data directory.
    """

    def __init__(self, watcher: "MeetingWatcher"):
        super().__init__()
        self._watcher = watcher

    def on_created(self, event):
        """Handle directory creation events."""
        if not isinstance(event, DirCreatedEvent):
            return

        dir_path = Path(event.src_path)

        # Only process directories matching the con* naming pattern
        if not dir_path.name.startswith("con"):
            return

        # Skip directories that have already been processed
        if dir_path.name in self._watcher.processed_dir_names:
            return

        logger.info(f"[MeetingWatcher] Detected new directory: {dir_path.name}")
        self._watcher._enqueue_pending(dir_path)


class MeetingWatcher:
    """
    Watches data directory for new con* meeting folders and triggers
    incremental indexing when all required files are present.

    Usage:
        watcher = MeetingWatcher(
            data_dir=Path("datademo"),
            on_new_meetings_callback=agent._incremental_sync,
            processed_meeting_dirs={"con1", "con4", ...}
        )
        watcher.start()
        # ... system runs ...
        watcher.stop()
    """

    def __init__(
        self,
        data_dir: Path,
        on_new_meetings_callback: Callable[[List[Path]], None],
        processed_meeting_dirs: Optional[Set[str]] = None,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ):
        """
        Args:
            data_dir: Path to the data directory (e.g. datademo/).
            on_new_meetings_callback: Callable that accepts a list of ready
                con* directory paths. Called from the watcher thread.
            processed_meeting_dirs: Set of con* directory names already
                processed at startup (e.g. {"con1", "con4", ...}).
                New directories not in this set will be monitored.
            debounce_seconds: Seconds to wait after detection before
                checking file completeness.
        """
        self.data_dir = Path(data_dir)
        self._callback = on_new_meetings_callback
        self.processed_dir_names: Set[str] = set(processed_meeting_dirs or set())
        self._debounce_seconds = debounce_seconds

        # Pending directories awaiting completeness check.
        # Protected by _pending_lock.
        self._pending: Set[Path] = set()
        self._pending_lock = threading.Lock()

        # Watchdog observer
        self._observer: Optional[Observer] = None

        # Background checker thread
        self._checker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the watchdog observer and the background checker thread."""
        if self._observer is not None:
            logger.warning("[MeetingWatcher] Already started, ignoring duplicate start()")
            return

        logger.info(f"[MeetingWatcher] Starting file system watcher on: {self.data_dir}")
        logger.info(f"[MeetingWatcher] Already processed directories: {len(self.processed_dir_names)}")

        # Start watchdog observer
        handler = _NewConDirHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.data_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()

        # Start background checker thread for debounce + completeness
        self._stop_event.clear()
        self._checker_thread = threading.Thread(
            target=self._checker_loop,
            name="MeetingWatcher-Checker",
            daemon=True,
        )
        self._checker_thread.start()

        logger.info("[MeetingWatcher] ✅ Watcher started successfully")

    def stop(self) -> None:
        """Stop the watchdog observer and background checker thread."""
        logger.info("[MeetingWatcher] Stopping watcher...")
        self._stop_event.set()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        if self._checker_thread is not None:
            self._checker_thread.join(timeout=5)
            self._checker_thread = None

        logger.info("[MeetingWatcher] ✅ Watcher stopped")

    def _enqueue_pending(self, dir_path: Path) -> None:
        """Add a newly detected directory to the pending set."""
        with self._pending_lock:
            self._pending.add(dir_path)

    def _checker_loop(self) -> None:
        """
        Background loop that periodically checks pending directories
        for file completeness. Implements debounce + retry logic.

        Flow:
        1. Sleep for debounce interval
        2. Take a snapshot of pending directories
        3. For each pending dir, check completeness
        4. Ready dirs -> call callback -> move to processed
        5. Not-ready dirs -> increment retry count, keep in pending
           or discard if max retries exceeded
        """
        # Track retry counts per directory
        retry_counts: dict[str, int] = {}

        while not self._stop_event.is_set():
            # Sleep in small increments to allow responsive shutdown
            for _ in range(int(self._debounce_seconds * 2)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

            # Take snapshot of pending dirs
            with self._pending_lock:
                if not self._pending:
                    continue
                snapshot = set(self._pending)

            ready_dirs: List[Path] = []
            still_pending: Set[Path] = set()

            for dir_path in snapshot:
                dir_name = dir_path.name

                if check_completeness(dir_path):
                    logger.info(
                        f"[MeetingWatcher] ✅ All 4 files ready for {dir_name}"
                    )
                    ready_dirs.append(dir_path)
                    retry_counts.pop(dir_name, None)
                else:
                    count = retry_counts.get(dir_name, 0) + 1
                    retry_counts[dir_name] = count

                    if count > _MAX_RETRIES:
                        logger.warning(
                            f"[MeetingWatcher] ⚠️ {dir_name} still incomplete after "
                            f"{_MAX_RETRIES} retries, giving up. "
                            f"Missing files will not be indexed until next restart."
                        )
                        retry_counts.pop(dir_name, None)
                    else:
                        logger.info(
                            f"[MeetingWatcher] ⏳ {dir_name} incomplete, "
                            f"retry {count}/{_MAX_RETRIES}"
                        )
                        still_pending.add(dir_path)

            # Update pending set: remove processed, keep retrying
            with self._pending_lock:
                self._pending -= snapshot
                self._pending |= still_pending

            # Trigger callback for ready directories
            if ready_dirs:
                # Sort for consistent ordering
                ready_dirs.sort(key=lambda p: p.name)
                logger.info(
                    f"[MeetingWatcher] Triggering incremental sync for "
                    f"{len(ready_dirs)} new meeting(s): "
                    f"{[d.name for d in ready_dirs]}"
                )
                try:
                    self._callback(ready_dirs)
                    # Mark as processed after successful callback
                    for d in ready_dirs:
                        self.processed_dir_names.add(d.name)
                except Exception as e:
                    logger.error(
                        f"[MeetingWatcher] ❌ Incremental sync callback failed: {e}",
                        exc_info=True,
                    )
