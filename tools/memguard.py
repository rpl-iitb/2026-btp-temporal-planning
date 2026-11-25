"""Memory guard for long unattended experiment runs.

    python tools/memguard.py --root <driver pid> [--min_mb 1500] [--log file]

Every 2 s: if the machine's MemAvailable drops below --min_mb, kill (SIGKILL) the whole process tree of the largest
stage of the driver --root: a stage is a child process of the driver, its tree is every descendant by parent links
(so planners started in their own session, e.g. POPF, are included). Killing whole trees makes the stage fail cleanly
(the driver logs it and moves on) instead of leaving a worker pool waiting for a killed worker. Nothing outside the
driver's tree (editor, browser, other jobs) is ever touched. Exits when the driver is gone.
"""
import argparse
import os
import signal
import time
from collections import defaultdict

PAGE = os.sysconf("SC_PAGE_SIZE")


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 1 << 30


def table():
    """-> {pid: (ppid, rss MB, cmd)}"""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            st = open("/proc/{}/stat".format(p)).read()
            ppid = int(st[st.rindex(")") + 2:].split()[1])
            rss = int(open("/proc/{}/statm".format(p)).read().split()[1]) * PAGE / (1 << 20)
            cmd = open("/proc/{}/cmdline".format(p), "rb").read().replace(b"\0", b" ").decode(errors="replace")[:160]
            out[int(p)] = (ppid, rss, cmd)
        except (OSError, ValueError, IndexError):
            continue
    return out


def tree(pid, kids):
    out, stack = [], [pid]
    while stack:
        q = stack.pop()
        out.append(q)
        stack += kids.get(q, [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=int, required=True)
    ap.add_argument("--min_mb", type=int, default=1500)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    me = os.getpid()

    def log(s):
        s = time.strftime("%Y-%m-%d %H:%M:%S ") + s
        if a.log:
            open(a.log, "a").write(s + "\n")
    log("memguard: driver {}, kill a stage tree below {} MB available".format(a.root, a.min_mb))
    while os.path.exists("/proc/{}".format(a.root)):
        m = avail_mb()
        if m < a.min_mb:
            T = table()
            kids = defaultdict(list)
            for p, (pp, _, _) in T.items():
                kids[pp].append(p)
            stages = []
            for s in kids.get(a.root, []):
                if s == me or T.get(s, (0, 0, ""))[2].startswith(("systemd-inhibit", "sleep")):
                    continue
                pids = tree(s, kids)
                stages.append((sum(T[q][1] for q in pids if q in T), s, pids))
            if stages:
                mb, s, pids = max(stages)
                for q in reversed(pids):
                    try:
                        os.kill(q, signal.SIGKILL)
                    except OSError:
                        pass
                log("memguard: {} MB available -> killed stage pid {} ({} processes, {:.0f} MB): {}".format(
                    m, s, len(pids), mb, T.get(s, (0, 0, "?"))[2]))
                time.sleep(5)
        time.sleep(2)
    log("memguard: driver {} gone, exit".format(a.root))


if __name__ == "__main__":
    main()
