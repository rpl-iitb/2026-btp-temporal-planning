#!/usr/bin/env bash
# Start a command only when the machine is ready for it (for unattended experiment runs).
#
#   tools/when.sh [--mem MB] [--done FILE]... [--gone PATTERN]... -- command [args...]
#
# Checks every 60 s and execs the command once all conditions hold:
#   --mem MB        MemAvailable is at least MB
#   --done FILE     FILE exists (e.g. a stage's .done marker); FILE:FAILPATTERN also accepts a line matching
#                   FAILPATTERN in the driver log given by --log
#   --gone PATTERN  no process command line (other than this script's) contains PATTERN (grep regex)
#   --log FILE      driver log for the FAIL patterns
set -uo pipefail
mem=0; dones=(); gones=(); log=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) mem=$2; shift 2 ;;
    --done) dones+=("$2"); shift 2 ;;
    --gone) gones+=("$2"); shift 2 ;;
    --log) log=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "when.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
ready() {
  local f p
  for f in "${dones[@]}"; do
    p=${f#*:}; f=${f%%:*}
    if [ ! -f "$f" ]; then
      if [ "$p" = "$f" ] || [ -z "$log" ] || ! grep -q "$p" "$log" 2>/dev/null; then return 1; fi
    fi
  done
  local procs
  procs=$(ps -eo args)                     # a snapshot (no pipes: grep -q + pipefail would hide a match)
  procs=$(grep -v when.sh <<< "$procs")
  for p in "${gones[@]}"; do grep -q -- "$p" <<< "$procs" && return 1; done
  [ "$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)" -ge "$mem" ] || return 1
  return 0
}
until ready; do sleep 60; done
exec "$@"
