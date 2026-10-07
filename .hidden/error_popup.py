import ctypes
import time
import sys
import platform

def is_windows():
    """Check if running on Windows"""
    try:
        return getattr(platform, 'system', lambda: '')() == 'Windows'
    except (Exception, OSError):
        return False

def show_error_dialog(title=None, message=None):
    """Show a Windows error message dialog. Optional title and message override defaults."""
    if not is_windows():
        print("Error: This script only works on Windows")
        return False

    # Windows MessageBox constants
    MB_OKCANCEL = 0x00000001
    MB_ICONERROR = 0x00000010
    MB_SYSTEMMODAL = 0x00001000

    default_title = "Python"
    default_message = (
        "The exception unknown software exception (0xc000041d) occurred in the application at location 0x00007FFA8B2C1234.\n\n"
        "Click on OK to terminate the program\n"
        "Click on CANCEL to debug the program"
    )
    title = default_title if title is None else title
    message = default_message if message is None else message
    if message is None:
        message = default_message
    if isinstance(message, bytes):
        try:
            message = message.decode('utf-8', errors='replace')
        except Exception:
            message = "An error occurred."
    if message is None:
        message = "An error occurred."
    if isinstance(title, bytes):
        try:
            title = title.decode('utf-8', errors='replace')
        except Exception:
            title = "Python"

    # Show the error dialog using MessageBoxW for Unicode support
    try:
        if not hasattr(ctypes, 'windll') or ctypes.windll is None or not hasattr(ctypes.windll, 'user32'):
            print("Error: Windows user32 not available")
            return False
        msg = (str(message) if message else "An error occurred.")[:2000]
        if not (msg and msg.strip()):
            msg = "An error occurred."
        tit = (str(title) if title else "Python")[:200]
        if not (tit and tit.strip()):
            tit = "Python"
        ret = ctypes.windll.user32.MessageBoxW(
            None,
            msg,
            tit,
            MB_OKCANCEL | MB_ICONERROR | MB_SYSTEMMODAL
        )
        # 0 = error, 1 = IDOK, 2 = IDCANCEL
        if ret == 0:
            return False
        try:
            return int(ret) == 1
        except (TypeError, ValueError):
            return False
    except (AttributeError, OSError, TypeError, ValueError, Exception) as e:
        try:
            print(f"Failed to show error dialog: {e}")
        except Exception:
            pass
        return False

def main():
    """Simulate a running program that crashes after 10 seconds"""
    if not is_windows():
        print("Error: This script only works on Windows.")
        sys.exit(1)
    try:
        print("Running...")
    except (OSError, ValueError):
        pass
    print("Press Ctrl+C to cancel\n")
    try:
        for i in range(10, 0, -1):
            try:
                sys.stdout.write(f"\rError in {i} seconds...  ")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
            try:
                time.sleep(1)
            except (OSError, ValueError, Exception):
                pass
        try:
            print("\r" + " " * 30 + "\r", end='')
            print("Showing error dialog...\n")
        except (OSError, ValueError):
            pass
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    if not show_error_dialog():
        try:
            sys.exit(1)
        except Exception:
            pass
        return
    try:
        sys.exit(0)
    except Exception:
        try:
            sys.exit(0)
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            print(f"Fatal: {e}")
        except Exception:
            pass
        try:
            sys.exit(1)
        except Exception:
            pass
