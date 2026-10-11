#!/usr/bin/env python3
"""List or set a USB camera's V4L2 controls (a stand-in for v4l2-ctl, which isn't installed on the GMKtec).

    python robot/v4l2_ctl.py <device>                                   # list controls + current values
    python robot/v4l2_ctl.py <device> auto_exposure=1 exposure_time_absolute=150

Works while the camera bridge is streaming (controls aren't exclusive), so no bridge restart is needed.
Names are the kernel's control names, lowercased with spaces -> underscores.
"""
import ctypes
import fcntl
import os
import sys


class QueryCtrl(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_uint32), ("name", ctypes.c_char * 32),
                ("minimum", ctypes.c_int32), ("maximum", ctypes.c_int32), ("step", ctypes.c_int32),
                ("default_value", ctypes.c_int32), ("flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 2)]


class Control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


def _iowr(nr, size):
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_QUERYCTRL = _iowr(36, ctypes.sizeof(QueryCtrl))
VIDIOC_G_CTRL = _iowr(27, ctypes.sizeof(Control))
VIDIOC_S_CTRL = _iowr(28, ctypes.sizeof(Control))
NEXT_CTRL, DISABLED, CLASS_TYPE = 0x80000000, 0x0001, 6


def controls(fd):
    q, out = QueryCtrl(id=NEXT_CTRL), {}
    while True:
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYCTRL, q)
        except OSError:
            return out
        if not q.flags & DISABLED and q.type != CLASS_TYPE:
            name = q.name.decode().lower().replace(",", "").replace(" ", "_")
            c = Control(id=q.id)
            try:
                fcntl.ioctl(fd, VIDIOC_G_CTRL, c)
                val = c.value
            except OSError:
                val = None
            out[name] = dict(id=q.id, min=q.minimum, max=q.maximum, step=q.step, default=q.default_value, value=val)
        q = QueryCtrl(id=q.id | NEXT_CTRL)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    fd = os.open(os.path.realpath(sys.argv[1]), os.O_RDWR)
    ctrls = controls(fd)
    for kv in sys.argv[2:]:  # in order: set auto_exposure before exposure_time_absolute
        name, v = kv.split("=", 1)
        if name not in ctrls:
            sys.exit(f"unknown control {name}; have: {', '.join(ctrls)}")
        old = ctrls[name]["value"]
        fcntl.ioctl(fd, VIDIOC_S_CTRL, Control(id=ctrls[name]["id"], value=int(v)))
        ctrls = controls(fd)
        print(f"{name}: {old} -> {ctrls[name]['value']}")
    if len(sys.argv) == 2:
        for name, c in ctrls.items():
            print(f"{name:<36} value={c['value']!s:<6} default={c['default']:<6} range={c['min']}..{c['max']} step={c['step']}")
