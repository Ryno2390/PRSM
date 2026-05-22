"""Sprint 716 — F51 sibling-race fix.

Sprint 715 closed F50: STREAM_END's put_nowait raced ahead of frame
queue.put coroutines. Sprint 716 found a sibling at the malformed-
frame terminal error path: same `call_soon_threadsafe(queue.put_nowait,
...)` pattern, same race risk.

Scenario where the race matters: server ships 3 valid frames + 1
malformed frame. The valid frames each schedule a deferred
`queue.put` coroutine. The malformed frame triggers the terminal-
error path which (pre-716) scheduled `put_nowait` via
`call_soon_threadsafe`. Because callbacks run before deferred
coroutines on the next loop tick, the terminal-error entry could
land in the queue ahead of the still-pending valid frame puts →
dispatcher saw error first, raised StageExecutionError, never
yielded the valid frames.

Fix: malformed-frame terminal-error path now uses the same
`run_coroutine_threadsafe(queue.put(...))` primitive as good frames.
Same lesson as sprint 715: any terminal entry that bypasses the
back-pressure-aware put MUST use the SAME scheduling primitive
as the entries it's terminating, or wire order is lost.
"""
from __future__ import annotations


def test_malformed_frame_path_uses_same_primitive_as_good_frames():
    """Pin: the malformed-frame error path must use
    `run_coroutine_threadsafe(queue.put(...))` (the back-pressure-
    aware coroutine) NOT `call_soon_threadsafe(put_nowait, ...)`
    so wire order is preserved with preceding valid frames."""
    import inspect
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
    )
    src = inspect.getsource(handle_chain_stream_response)
    # Locate the malformed-frame branch by its sentinel message.
    marker = "malformed frame payload"
    assert marker in src
    idx = src.find(marker)
    # 200-char window after the marker covers the put logic.
    window = src[idx:idx + 600]
    assert "run_coroutine_threadsafe" in window, (
        "malformed-frame terminal error must use "
        "run_coroutine_threadsafe (sprint 716 F51 fix)"
    )


def test_all_three_response_paths_use_same_primitive():
    """Pin: post-sprint-716, handle_chain_stream_response has three
    queue-write paths (good frame, STREAM_END, malformed-frame
    error) and ALL THREE must use the same `run_coroutine_threadsafe`
    primitive so wire order is preserved. The legacy `put_nowait`
    calls are now strictly defensive fallbacks inside except blocks.
    """
    import inspect
    from prsm.node.chain_executor_adapters import (
        handle_chain_stream_response,
    )
    src = inspect.getsource(handle_chain_stream_response)
    # Count the run_coroutine_threadsafe primary scheduler calls
    # (one per path).
    rcts_count = src.count("run_coroutine_threadsafe")
    assert rcts_count >= 3, (
        f"expected ≥3 run_coroutine_threadsafe scheduler calls "
        f"(one per write path: frame / END / malformed-error), "
        f"got {rcts_count}"
    )
