"""
Start the DriveSentinel technician web app.

    .venv/Scripts/python.exe -m webapp                 laptop, http://127.0.0.1:8000
    .venv/Scripts/python.exe -m webapp --host 0.0.0.0  reachable from a phone on
                                                       the same network, or on the
                                                       PYNQ board
Options: --port, --pace (seconds per replay step in the background fleet replay).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(prog="python -m webapp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pace", type=float, default=0.25,
                    help="seconds per replay step for the background fleet replay")
    a = ap.parse_args()
    os.environ["DS_PORT"] = str(a.port)
    os.environ["DS_PACE"] = str(a.pace)

    import uvicorn
    from webapp.server import create_app
    print(f"  DriveSentinel technician app on http://{a.host}:{a.port}/  (ctrl-c to stop)")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
