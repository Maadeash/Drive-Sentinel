"""
Resolve a register-accessible object for a PYNQ overlay cell, tolerant of
however PYNQ happens to have named it.

WHY THIS EXISTS -- a real finding, from a real board, 2026-09-24
------------------------------------------------------------------
`board/build_overlay.tcl` adds the accelerator with

    create_bd_cell -type module -reference ds_pynq_wrap ds_0

which is a plain RTL module reference: `ds_pynq_wrap` was never packaged as an
IP-XACT core (no VLNV, no `package_ip` / `ipx::package_project` step). On the
first real board run, that produced:

    ol.ip_dict            == {'axi_dma_0': {...}, 'ds_0/s_axi': {...}}   # no 'ds_0' key
    core = ol.ds_0                  # succeeds -- returns a hierarchy proxy
    core.read(...)                  # AttributeError: Could not find IP or
                                     #   hierarchy 'read' in overlay

i.e. PYNQ folded the module's one AXI4-Lite interface into the address map
keyed by `<cell>/<bus interface>` ('ds_0/s_axi') instead of `<cell>` alone,
and `ol.ds_0` is a bare hierarchy container -- the actual register file is one
level down, at `ol.ds_0.s_axi`. `axi_dma_0` did not show this, because it is a
properly packaged Xilinx IP with its own driver.

Confirmed on the board the same day: the notebook's next run printed
`hierarchies: ['ds_0']` and resolved the core to PYNQ's own `DefaultIP` at
0x43c00000 -- i.e. path 2 below, `ol.ds_0.s_axi` -- and went on to read the
build ID and run 120 windows through it (board/board_run.json). The function
stays defensive rather than hard-coded to `.s_axi`, so it keeps working if a
future rebuild packages the core properly, without a bitstream rebuild to
work around a Python-side naming difference.

WHAT IT TRIES, IN ORDER
------------------------
  1. `ol.<cell>` itself, if it already has `.read()`/`.write()` -- the case
     for a properly packaged IP (this is what `axi_dma_0`-style access needs,
     and would also be right if a future rebuild packages `ds_0` properly).
  2. `ol.<cell>.<iface>` for every `ip_dict` key `<cell>/<iface>` -- the case
     observed on the board above.
  3. A raw `pynq.MMIO` built directly from that `ip_dict` entry's
     `phys_addr` / `addr_range` -- independent of whatever attribute path
     PYNQ chose, and so the most robust of the three, at the cost of bypassing
     PYNQ's own IP wrapper.

Raises `RuntimeError`, naming everything it tried, if none of the three work.
"""

from typing import Any


def _has_registers(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "read") and hasattr(obj, "write")


def resolve_mmio(ol, cell: str):
    """A register-accessible object (`.read(offset)` / `.write(offset, value)`)
    for the overlay cell `cell`, e.g. `resolve_mmio(ol, "ds_0")`."""
    tried = []
    root = getattr(ol, cell, None)
    if _has_registers(root):
        return root
    tried.append(f"ol.{cell} (present but has no .read()/.write())"
                 if root is not None else f"ol.{cell} (not found)")

    keys = sorted(k for k in ol.ip_dict if k == cell or k.startswith(cell + "/"))
    for key in keys:
        if key == cell:
            continue
        iface = key[len(cell) + 1:]
        obj, ok = root, root is not None
        for part in iface.split("/"):
            obj = getattr(obj, part, None) if ok else None
            ok = obj is not None
        if _has_registers(obj):
            return obj
        tried.append(f"ol.{cell}.{iface.replace('/', '.')}"
                     + ("" if ok else " (not found)"))

    for key in keys:
        info = ol.ip_dict.get(key, {})
        if "phys_addr" in info and "addr_range" in info:
            from pynq import MMIO      # the real driver, or the mock's stand-in
            return MMIO(info["phys_addr"], info["addr_range"])
        tried.append(f"MMIO(ip_dict[{key!r}]) (missing phys_addr/addr_range)")

    raise RuntimeError(
        f"could not find a register interface for {cell!r}; tried: {tried}; "
        f"ol.ip_dict keys: {sorted(ol.ip_dict)}")
