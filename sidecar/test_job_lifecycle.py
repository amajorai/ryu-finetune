"""Regression checks for fine-tune job state transitions and GPU serialization."""

from __future__ import annotations

import threading

from ryu_unsloth import trainer
from ryu_unsloth.jobs import Job


def terminal_states_are_final() -> None:
    job = Job(id="done", config={})
    job.mark_running()
    job.mark_done(output_dir="/tmp/out")
    job.mark_failed("late error")
    job.mark_cancelled()
    job.mark_running()
    assert job.state == "succeeded", job.state
    assert job.error is None
    assert [e["state"] for e in job.events if e["type"] == "state"] == ["running", "succeeded"]


def cancelled_while_queued_never_trains() -> None:
    calls: list[str] = []
    original = trainer._run
    trainer._run = lambda job: calls.append(job.id)  # type: ignore[assignment]
    try:
        job = Job(id="queued", config={})
        job.request_cancel()
        trainer.run_job(job)
    finally:
        trainer._run = original  # type: ignore[assignment]
    assert calls == []
    assert job.state == "cancelled", job.state


def jobs_run_one_at_a_time() -> None:
    active = 0
    peak = 0
    guard = threading.Lock()
    release = threading.Event()

    def fake_run(job: Job) -> None:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        release.wait(timeout=2)
        with guard:
            active -= 1
        job.mark_done(output_dir="/tmp/out")

    original = trainer._run
    trainer._run = fake_run  # type: ignore[assignment]
    try:
        jobs = [Job(id=f"j{i}", config={}) for i in range(3)]
        threads = [threading.Thread(target=trainer.run_job, args=(j,)) for j in jobs]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        trainer._run = original  # type: ignore[assignment]
    assert peak == 1, peak
    assert all(j.state == "succeeded" for j in jobs)


def test_terminal_states_are_final() -> None:
    terminal_states_are_final()


def test_cancelled_while_queued_never_trains() -> None:
    cancelled_while_queued_never_trains()


def test_jobs_run_one_at_a_time() -> None:
    jobs_run_one_at_a_time()


def main() -> None:
    terminal_states_are_final()
    cancelled_while_queued_never_trains()
    jobs_run_one_at_a_time()


if __name__ == "__main__":
    main()
