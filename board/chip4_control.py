"""Guarded four-NPU natural IRQ handoff and unchanged Corr broadcast ABI.

No host PC/jump register is assumed. Counter vectors are per physical selector,
so an older NPU0 counter can differ from the other three and wrap independently.
"""
from __future__ import annotations

import math
import time


NPU_SELECTORS = (0, 1, 2, 3)
CORR_NPU_MASTER_OFFSETS = (0, 0x10000, 0x20000, 0x30000)
NPU_ALWAYS_ACTIVE_TILE_INFO = (2, 1, 3, 1)


def _counters(values):
    if not isinstance(values, (tuple, list)) or len(values) != 4 or any(
            type(v) is not int or not 0 <= v < 256 for v in values):
        raise ValueError('Supply exactly four uint8 IRQ counters, in physical selector order')
    return tuple(values)


def counters_after(values, count):
    values = _counters(values)
    if type(count) is not int or not 0 <= count < 256:
        raise ValueError('Resume count must be a uint8 integer')
    return [(value+count) & 255 for value in values]


def _bases(board):
    base, stride = int(getattr(board, 'base', 0x2000)), int(getattr(board, 'stride', 0x1000))
    if base < 0 or base % 0x1000 or stride != 0x1000:
        raise ValueError('Expected four consecutive 4 KiB NPU register windows')
    return tuple(base+sel*stride for sel in NPU_SELECTORS)


def resume_all(board, expected_counters, *, name='four-chip natural resume'):
    counters = _counters(expected_counters)
    bases = _bases(board)
    status = [int(board.read_reg(base+0xc)) for base in bases]
    # Inspect every chip before the first mutation. An IRQ on just NPU0 is not
    # enough to prove a fresh four-chip barrier.
    if any(not 0 <= value <= 0xffffffff or (value & 3) != 2 or
           ((value >> 24) & 255) != count for value, count in zip(status, counters)):
        raise RuntimeError(name+': all four fresh IRQ/counter states are required before resume: '+str(status))
    started = time.perf_counter()
    for base in bases:
        for value in (0, 1, 0):
            board.write_reg(base+8, value)
    return dict(name=name, seconds=time.perf_counter()-started,
        irq_counters_before=list(counters),irq_counters_after=counters_after(counters, 1),
        instruction_words_written=0,start_pulses=0,cpu_jump_used=False)


def _wait_all(board, expected_counters, *, finish, corr_finish, timeout, poll_interval, name):
    counters = _counters(expected_counters)
    bases = _bases(board)
    timeout = float(board.timeout if timeout is None else timeout)
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval < 0:
        raise ValueError('Positive finite timeout and nonnegative finite polling interval required')
    wanted, started = (1 if finish else 2), time.perf_counter()
    polls, observations, previous = 0, [], None
    while True:
        status = [int(board.read_reg(base+0xc)) for base in bases]
        corr = int(board.read_reg(4)) if corr_finish else None
        polls += 1
        if any(not 0 <= value <= 0xffffffff for value in status):
            raise IOError(name+': invalid NPU CSR0 value')
        if any((value & 3) == 3 for value in status):
            raise RuntimeError(name+': unexpected simultaneous FINISH and IRQ')
        state = (tuple(status),corr)
        if state != previous:
            if len(observations) < 100:
                observations.append(dict(seconds=time.perf_counter()-started,npu_csr0=list(status),corr_status=corr))
            previous = state
        done = all((value & 3) == wanted and ((value >> 24) & 255) == count
                   for value, count in zip(status, counters))
        if done and (not corr_finish or corr & 2):
            return dict(name=name,seconds=time.perf_counter()-started,npu_csr0=status,
                irq_counters=list(counters),corr_status=corr,polls=polls,transitions=observations,
                instruction_words_written=0,file_writes=0,golden_comparisons=0,
                timer_scope='Host-observed completion of every physical NPU, including polling')
        if not finish and any(value & 1 for value in status):
            raise RuntimeError(name+': unexpected FINISH before requested resident IRQ')
        # A just-issued RESUME can still expose the previous sticky IRQ. Wait
        # for fresh counters; do not diagnose posted-write latency as an error.
        if time.perf_counter()-started > timeout:
            raise TimeoutError(name+': timeout; CSR0='+str([hex(value) for value in status])+
                               ' Corr='+str(corr)+' expected='+str(list(counters)))
        if poll_interval:
            time.sleep(poll_interval)


def wait_all_irq(board, expected_counters, *, corr_finish=False, timeout=None,
                 poll_interval=.00001, name='four-chip resident host IRQ'):
    return _wait_all(board, expected_counters, finish=False,corr_finish=corr_finish,
                     timeout=timeout,poll_interval=poll_interval,name=name)


def wait_all_finish(board, expected_counters, *, corr_finish=True, timeout=None,
                    poll_interval=.00001, name='four-chip Corr/NPU FINISH'):
    return _wait_all(board, expected_counters, finish=True,corr_finish=corr_finish,
                     timeout=timeout,poll_interval=poll_interval,name=name)


class Fixed4ChipCorrConfig:
    def __init__(self, base):
        self._base = base
        values = base.registers()
        if (base.width,base.height,base.tile_width,base.tile_height,
            base.scale_iters,base.delta_iters,base.chips) != (160,40,40,20,8,24,4):
            raise ValueError('Four-chip no-skip ABI requires 160x40, 8 physical tiles and fixed 8 SU + 24 DU')
        if values.get(51) != 8 or tuple(values.get(42+i) for i in NPU_SELECTORS) != CORR_NPU_MASTER_OFFSETS:
            raise ValueError('Corr physical NPU S02 offsets must be 0,0x10000,0x20000,0x30000')

    def __getattr__(self, name): return getattr(self._base,name)

    @property
    def min_scale_iters(self): return 63

    @property
    def min_delta_iters(self): return 63

    def registers(self):
        values = dict(self._base.registers())
        values.update({37:(8<<6)|63,38:(24<<6)|63,51:0})
        return values

    def policy_report(self):
        return dict(mode='four_chip_fixed_8su_24du_all_tiles',chips=4,physical_tiles=8,
            early_stop_enabled=False,corr_old_mask_bypassed=True,
            npu_tile_info_required=list(NPU_ALWAYS_ACTIVE_TILE_INFO),
            register_overrides={'37':575,'38':1599,'51':0},
            npu_s02_offsets=list(CORR_NPU_MASTER_OFFSETS),
            broadcast_physical_selector_order=[1,2,3,0],
            irq_source_selector=0,thresholds_unchanged=True,original_rtl_modified=False)


def fixed_corr_config4(base): return Fixed4ChipCorrConfig(base)
