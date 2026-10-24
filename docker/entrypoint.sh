#!/bin/sh
# Container entrypoint.
#   ui (default)   build the KB index if missing, then serve the dashboard on $PORT
#   anything else  run it as a `helpdesk` subcommand, e.g. `query "..."` or `eval --limit 5`
#   smoke          run scripts/smoke_test.py (live scenarios)
set -e

ensure_index() {
    python /app/scripts/prepare_data.py --index-only --if-missing
}

case "${1:-ui}" in
    ui)
        ensure_index
        exec helpdesk ui --host 0.0.0.0 --port "${PORT:-8080}" --headless
        ;;
    smoke)
        ensure_index
        shift
        exec python /app/scripts/smoke_test.py "$@"
        ;;
    sh|bash)
        exec "$@"
        ;;
    *)
        ensure_index
        exec helpdesk "$@"
        ;;
esac
