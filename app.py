# app.py
"""
Application Entry Point.

Handles multiprocessing safety guards and environment configuration
BEFORE importing the heavy GUI and ML modules.
"""

import sys
import os
import signal
import multiprocessing
import logging


def main():
    try:
        _run()
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Attempt to show a user-friendly dialog if Qt is available
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            _app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Fatal Error",
                f"The application failed to start:\n\n{exc}")
        except Exception:
            print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run():
    # ── CRITICAL: Multiprocessing Guards ──────────────────────────────────
    # Required on Windows. Without this, the child process re-runs the
    # main script, which would otherwise import PySide6 and create a
    # second blank window.
    multiprocessing.freeze_support()

    # If this is a child process spawned by multiprocessing, bail out
    # immediately. The multiprocessing machinery will call the target
    # function (render_project_worker) directly.
    if multiprocessing.parent_process() is not None:
        return

    # ── Early Configuration & Logging ─────────────────────────────────────
    # Must happen BEFORE importing torch, transformers, or PySide6.
    import utils
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    from config import AppConfig
    config = AppConfig()

    # Resolve log file path from config
    _log_file = config.get("log_file_path", "")
    if _log_file and _log_file.lower() != "off":
        if not os.path.isabs(_log_file):
            _log_file = os.path.join(_log_dir, _log_file)
        config.settings["log_file_path"] = _log_file
    else:
        # Default: auto-enable file logging at logs/app.log
        config.settings["log_file_path"] = os.path.join(_log_dir, "app.log")

    utils.setup_logging(log_config=config.settings)
    logger = logging.getLogger(__name__)

    # ── Configure Global Cache Directory ───────────────────────────────────
    _cache_dir = config.get("hf_cache_dir", "")
    if not _cache_dir:
        _cache_dir = os.path.join(AppConfig.APP_ROOT, "models_cache")
    if not os.path.isabs(_cache_dir):
        _cache_dir = os.path.abspath(_cache_dir)

    os.environ["HF_HOME"]                = _cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"]  = _cache_dir
    os.environ["TRANSFORMERS_CACHE"]     = _cache_dir

    # Respect the user-configured datasets directory
    _datasets_dir = config.get("hf_datasets_dir", "")
    if not _datasets_dir:
        _datasets_dir = os.path.join(_cache_dir, "datasets")
    if not os.path.isabs(_datasets_dir):
        _datasets_dir = os.path.abspath(_datasets_dir)
    os.environ["HF_DATASETS_CACHE"]      = _datasets_dir

    # Respect the user-configured symlink setting (broken if unconditionally set to 0)
    _use_symlinks = config.get("hf_use_symlinks", False)
    if not _use_symlinks:
        os.environ["HF_HUB_SYMLINKS"]        = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"]       = "0"

    logger.info(f"[App] Global cache configured to: {_cache_dir}")

    # ── Import GUI & Run Application ──────────────────────────────────────
    from PySide6.QtWidgets import QApplication
    from gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Backup cleanup if the event loop exits without closeEvent firing
    app.aboutToQuit.connect(window._on_about_to_quit)

    # ── Graceful shutdown on Ctrl+C / SIGTERM ──────────────────────────
    # Without this, Ctrl+C kills the process instantly, orphaning worker
    # subprocesses (no poison pills, no GPU cleanup, no manifest saves).
    # We hook the signal and ask Qt to close the window, which triggers
    # closeEvent → _on_about_to_quit → full worker pool teardown.
    _shutdown_requested = False

    def _sig_handler(sig, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Second Ctrl+C — force exit immediately
            logger.critical("[App] Forced exit (second signal)")
            os._exit(1)
        _shutdown_requested = True
        logger.warning(f"[App] Received signal {sig} — shutting down gracefully…")
        # Mark force-close so closeEvent skips the confirmation dialog
        window._force_close = True
        # Schedule quit on the Qt event loop thread (signal handler context)
        # Use a single-shot timer so it fires after the signal handler returns.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, app.quit)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, OSError):
        # signal.signal can fail on some platforms (e.g. Windows + GUI)
        pass

    ret = app.exec()
    sys.exit(ret)


if __name__ == "__main__":
    main()