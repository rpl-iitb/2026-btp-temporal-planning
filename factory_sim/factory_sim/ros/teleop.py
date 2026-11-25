"""Keyboard teleop for the FactorySim robots (run next to a Gazebo or hardware run): select a robot, switch its mode,
drive it, jog its arm, or give it factory tasks.

    factory_sim/ros/teleop.sh            (Docker run)      or      python3 -m factory_sim.ros.teleop

Modes per robot: AUTO (the policy decides), TASK (you are its policy: pick one of its legal actions; it drives, docks
and picks / places by itself), DRIVE (you drive it with the keys and jog its arm; the factory gives it no work).
"""
import json
import select
import sys
import termios
import threading
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from . import robot as arm

HELP = """keys: 1-9 select robot | z AUTO  t TASK  v DRIVE | q quit
DRIVE: w/x faster/slower  a/d turn left/right  s stop  SPACE e-stop | arm robots: j/l base  i/k shoulder  u/o elbow  n/m wrist  g gripper  h home  c carry
TASK : [ / ] choose action  ENTER do it  (the robot drives, docks and picks / places by itself)"""
MAX_V, MAX_W = 0.26, 1.5


class Teleop(Node):
    def __init__(self):
        super().__init__("factory_teleop")
        self.status, self.sel, self.cursor, self.v, self.w, self.msg = None, None, 0, 0.0, 0.0, ""
        self.cmd_pub, self.vel = self.create_publisher(String, "/factory_sim/teleop", 10), {}
        self.create_subscription(String, "/factory_sim/status", self.on_status, 10)
        self.create_timer(0.1, self.drive)
        self.create_timer(0.25, self.draw)

    def on_status(self, m):
        self.status = json.loads(m.data)
        if self.sel is None and self.status["robots"]:
            self.sel = sorted(self.status["robots"])[0]

    def robot(self):
        return (self.status or {}).get("robots", {}).get(self.sel)

    def send(self, **c):
        self.cmd_pub.publish(String(data=json.dumps(dict(c, robot=self.sel))))

    def drive(self):
        r = self.robot()
        if not r or r["mode"] != "drive" or not r["held"]:
            return
        if self.sel not in self.vel:
            self.vel[self.sel] = self.create_publisher(Twist, "/{}/cmd_vel".format(self.sel), 10)
        t = Twist()
        t.linear.x, t.angular.z = self.v, self.w
        self.vel[self.sel].publish(t)

    def label(self, a):
        jobs = self.status["jobs"]
        if a[0] == "pickup":
            p, where, at = jobs[str(a[1])]
            return "pick up job {} ({}) at {}".format(a[1], p, at)
        if a[0] == "deliver":
            return "deliver to {}".format(a[1])
        if a[0] == "move":
            return "move to {}".format(a[1])
        if a[0] == "charge":
            return "go and charge"
        return " ".join(map(str, a))

    def key(self, k):
        r = self.robot()
        if k in "123456789" and self.status:
            names = sorted(self.status["robots"])
            if int(k) <= len(names):
                self.sel, self.cursor, self.v, self.w = names[int(k) - 1], 0, 0.0, 0.0
        elif r is None:
            return
        elif k in "ztv":
            self.v = self.w = 0.0
            self.send(cmd="mode", mode={"z": "auto", "t": "task", "v": "drive"}[k])
        elif r["mode"] == "task" and r["legal"]:
            if k == "[":
                self.cursor = (self.cursor - 1) % len(r["legal"])
            elif k == "]":
                self.cursor = (self.cursor + 1) % len(r["legal"])
            elif k in "\r\n":
                a = r["legal"][min(self.cursor, len(r["legal"]) - 1)]
                self.send(cmd="act", action=a)
                self.msg, self.cursor = "sent: " + self.label(a), 0
        elif r["mode"] == "drive":
            if k == "w":
                self.v = min(MAX_V, self.v + 0.05)
            elif k == "x":
                self.v = max(-MAX_V, self.v - 0.05)
            elif k == "a":
                self.w = min(MAX_W, self.w + 0.3)
            elif k == "d":
                self.w = max(-MAX_W, self.w - 0.3)
            elif k in "s ":
                self.v = self.w = 0.0
            elif k in "jliknmuogh" + "c" and r.get("q"):
                q = list(r["q"])
                step = {"j": (0, 0.15), "l": (0, -0.15), "i": (1, 0.1), "k": (1, -0.1), "u": (2, 0.1), "o": (2, -0.1), "n": (3, 0.1), "m": (3, -0.1)}
                if k in step:
                    q[step[k][0]] += step[k][1]
                elif k == "g":
                    q[4] = q[5] = arm.CLOSED if q[4] > (arm.OPEN + arm.CLOSED) / 2 else arm.OPEN
                elif k == "h":
                    q[:4] = list(arm.HOME)
                elif k == "c":
                    q[:4] = list(arm.CARRY)
                self.send(cmd="arm", q=q)

    def draw(self):
        st = self.status
        out = ["\033[2J\033[H\033[1mFactorySim teleop\033[0m"]
        if not st:
            out.append("waiting for /factory_sim/status (is the factory node running?)")
        else:
            out.append("t = {} s   orders done {}/{}".format(st["t"], st["done"], st["orders"]))
            out.append("    #  robot    type             mode   phase          battery  carry   at")
            for i, n in enumerate(sorted(st["robots"])):
                r = st["robots"][n]
                mode = r["mode"].upper() + ("" if r["mode"] != "drive" or r["held"] else " (after task)")
                out.append("{} {}  {:<8} {:<16} {:<6} {:<14} {:>6}   {:<7} {}".format(">" if n == self.sel else " ", i + 1, n, r.get("type", "-"), mode, ("FAULT" if r["fault"] else r["phase"]), "-" if r["battery"] is None else "{:.0f}%".format(r["battery"]), "-" if r["carry"] is None else "job {}".format(r["carry"]), r["node"]))
            r = self.robot()
            if r and r["mode"] == "task":
                out.append("\n{} TASK: {}".format(self.sel, "choose an action" if r["legal"] else "busy: {}".format(r["task"])))
                for k, a in enumerate(r["legal"]):
                    out.append("  {} {}".format("->" if k == self.cursor else "  ", self.label(a)))
            elif r and r["mode"] == "drive":
                out.append("\n{} DRIVE: v = {:.2f} m/s  w = {:.2f} rad/s   {}".format(self.sel, self.v, self.w, "" if r["held"] else "(takes over when its current task ends)"))
        out += ["", self.msg, HELP]
        sys.stdout.write("\r\n".join(out) + "\r\n")
        sys.stdout.flush()


def main():
    rclpy.init()
    node = Teleop()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                node.key(k)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.v = node.w = 0.0
        node.drive()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
