"""
MeetingWatcher - File system watcher for automatic meeting hot reload.

Uses watchdog to monitor the data directory for new `con*` folders nested
within user directories (e.g., datademo/PersonName/con*).
When a new folder is detected and all 4 required files are present,
triggers an incremental sync callback. Also handles deletions and renames.

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
from watchdog.events import FileSystemEventHandler, DirCreatedEvent, DirDeletedEvent, DirMovedEvent

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
    if not con_dir.exists():
        return False
    for pattern in _REQUIRED_FILE_PATTERNS:
        matches = list(con_dir.glob(pattern))
        if not matches:
            return False
    return True


class _MeetingDirHandler(FileSystemEventHandler):
    """
    Watchdog event handler that detects con* directory changes
    (created, deleted, moved) inside the data directory recursively.
    """

    def __init__(self, watcher: "MeetingWatcher"):
        super().__init__()
        self._watcher = watcher

    def _get_rel_path(self, absolute_path: str) -> Optional[str]:
        """Convert absolute path to relative path from data_dir if it's a con* dir."""
        try:
            path = Path(absolute_path)
            # We only care about directories named con*
            if not path.name.startswith("con"):
                return None
            
            # Get relative path from data_dir (e.g., "Person/con1")
            rel_path = path.relative_to(self._watcher.data_dir)
            return str(rel_path).replace("\\", "/")
        except ValueError:
            return None

    def on_created(self, event):
        """Handle directory creation events."""
        if not event.is_directory:
            return
        
        rel_path = self._get_rel_path(event.src_path)
        if not rel_path:
            return

        # Skip directories that have already been processed
        if rel_path in self._watcher.processed_rel_paths:
            return

        logger.info(f"[MeetingWatcher] Detected new meeting directory: {rel_path}")
        self._watcher._enqueue_pending(Path(event.src_path))

    def on_deleted(self, event):
        """Handle directory deletion events."""
        if not event.is_directory:
            return
            
        rel_path = self._get_rel_path(event.src_path)
        if not rel_path:
            # Check if it was a person directory (direct child of data_dir)
            path = Path(event.src_path)
            if path.parent == self._watcher.data_dir:
                logger.info(f"[MeetingWatcher] Person directory deleted: {path.name}")
                # Trigger a full reload for this specific user or notify
                self._watcher._handle_deletion(rel_path=None, person_name=path.name)
            return

        logger.info(f"[MeetingWatcher] Meeting directory deleted: {rel_path}")
        self._watcher._handle_deletion(rel_path=rel_path)

    def on_moved(self, event):
        """Handle directory rename/move events."""
        if not event.is_directory:
            return

        old_rel = self._get_rel_path(event.src_path)
        new_rel = self._get_rel_path(event.dest_path)

        if old_rel:
            logger.info(f"[MeetingWatcher] Meeting directory moved/renamed from: {old_rel}")
            self._watcher._handle_deletion(rel_path=old_rel)
        
        if new_rel:
            logger.info(f"[MeetingWatcher] Meeting directory moved/renamed to: {new_rel}")
            self._watcher._enqueue_pending(Path(event.dest_path))


class MeetingWatcher:
    """
    Watches data directory recursively for con* meeting folders and triggers
    incremental indexing or deletion sync.

    Usage:
        watcher = MeetingWatcher(
            data_dir=Path("datademo"),
            on_new_meetings_callback=agent._incremental_sync,
            on_deleted_meetings_callback=agent._on_meeting_deleted,
            processed_rel_paths={"Person1/con1", "Person2/con4", ...}
        )
        watcher.start()
    """

    def __init__(
        self,
        data_dir: Path,
        on_new_meetings_callback: Callable[[List[Path]], None],
        on_deleted_meetings_callback: Callable[[Optional[str], Optional[str]], None],
        processed_rel_paths: Optional[Set[str]] = None,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ):
        """
        Args:
            data_dir: Path to the data directory (e.g. datademo/).
            on_new_meetings_callback: Callback for new/ready con* directory paths.
            on_deleted_meetings_callback: Callback for deleted paths. 
                                        Args: (rel_path, person_name)
            processed_rel_paths: Set of relative paths already processed (e.g. {"Elon/con1"}).
            debounce_seconds: Seconds to wait after detection before checking completeness.
        """
        self.data_dir = Path(data_dir).absolute()
        self._callback_add = on_new_meetings_callback
        self._callback_del = on_deleted_meetings_callback
        self.processed_rel_paths: Set[str] = set(processed_rel_paths or set())
        self._debounce_seconds = debounce_seconds

        # Pending directories awaiting completeness check
        self._pending: Set[Path] = set()
        self._pending_lock = threading.Lock()

        # Watchdog observer
        self._observer: Optional[Observer] = None

        # Background checker thread
        self._checker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the recursive watchdog observer and background checker."""
        if self._observer is not None:
            return

        logger.info(f"[MeetingWatcher] Starting recursive watcher on: {self.data_dir}")
        
        # Start watchdog observer with recursive=True
        handler = _MeetingDirHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.data_dir), recursive=True)
        self._observer.daemon = True
        self._observer.start()

        # Start background checker thread
        self._stop_event.clear()
        self._checker_thread = threading.Thread(
            target=self._checker_loop,
            name="MeetingWatcher-Checker",
            daemon=True,
        )
        self._checker_thread.start()
        logger.info("[MeetingWatcher] ✅ Recursive watcher started")

    def stop(self) -> None:
        """Stop the observer and checker."""
        self._stop_event.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._checker_thread:
            self._checker_thread.join(timeout=5)
            self._checker_thread = None

    def _enqueue_pending(self, dir_path: Path) -> None:
        with self._pending_lock:
            self._pending.add(dir_path.absolute())

    def _handle_deletion(self, rel_path: Optional[str] = None, person_name: Optional[str] = None) -> None:
        """Handle deletion by updating local state and triggering callback."""
        if rel_path in self.processed_rel_paths:
            self.processed_rel_paths.discard(rel_path)
        
        # Trigger the deletion callback to update metadata/vector store
        try:
            self._callback_del(rel_path, person_name)
        except Exception as e:
            logger.error(f"[MeetingWatcher] Deletion callback failed: {e}")

    def _checker_loop(self) -> None:
        """Background loop for debounce + completeness check."""
        retry_counts: dict[str, int] = {}

        while not self._stop_event.is_set():
            time.sleep(1.0) # Check every second
            
            with self._pending_lock:
                if not self._pending:
                    continue
                snapshot = set(self._pending)

            ready_dirs: List[Path] = []
            still_pending: Set[Path] = set()

            for dir_path in snapshot:
                try:
                    rel_path = str(dir_path.relative_to(self.data_dir)).replace("\\", "/")
                except ValueError:
                    continue

                if check_completeness(dir_path):
                    logger.info(f"[MeetingWatcher] ✅ Ready: {rel_path}")
                    ready_dirs.append(dir_path)
                    retry_counts.pop(rel_path, None)
                else:
                    count = retry_counts.get(rel_path, 0) + 1
                    retry_counts[rel_path] = count
                    if count > _MAX_RETRIES:
                        logger.warning(f"[MeetingWatcher] ⚠️ Giving up on {rel_path} after {_MAX_RETRIES} retries")
                        retry_counts.pop(rel_path, None)
                    else:
                        still_pending.add(dir_path)

            with self._pending_lock:
                self._pending -= snapshot
                self._pending |= still_pending

            if ready_dirs:
                try:
                    self._callback_add(ready_dirs)
                    for d in ready_dirs:
                        rel = str(d.relative_to(self.data_dir)).replace("\\", "/")
                        self.processed_rel_paths.add(rel)
                except Exception as e:
                    logger.error(f"[MeetingWatcher] Add callback failed: {e}")

