//! Automatic parallel-download slot selection.
//!
//! Parallelism helps a download only while the *link* is idle waiting on one
//! stream. On a fat, high-latency pipe a single TCP connection leaves most of the
//! bandwidth-delay product unused, so N streams finish N× faster; once the link
//! (or the server, or the disk) is the bottleneck, adding streams just splits the
//! same bytes into more pieces and makes every individual file slower — and each
//! extra connection still costs a handshake and a server slot. So there is no
//! single right number: it depends on the connection in front of the user.
//!
//! This is why the tuner does not simply read a bandwidth number off a table and
//! trust it. The table only *seeds* the search — from there it hill-climbs against
//! measured aggregate throughput, which is the only signal that actually knows
//! whether the last slot it added bought anything:
//!
//! 1. **Seed** from the first real throughput sample ([`slots_for_throughput`]),
//!    so a fast connection does not have to climb from 1 one step at a time.
//! 2. **Only judge while slot-limited** — if fewer downloads are running than we
//!    already allow, or nothing is queued behind them, the slot count is provably
//!    not the constraint and a sample says nothing about it. Adjusting on those
//!    samples is how a naive controller talks itself down to 1 slot during a lull.
//! 3. **Climb on evidence** — a sample meaningfully *better* than the best seen
//!    (>10%) means the last increase paid off, so try one more. A sample
//!    meaningfully *worse* (>10%) means we pushed past the bottleneck (or the
//!    server started throttling), so give a slot back. Flat means we found the
//!    knee: hold.
//!
//! Bounds are [`MIN_SLOTS`]..=[`MAX_SLOTS`]. The ceiling of 8 is the same
//! neighbourhood every mainstream downloader settles on (browsers allow 6
//! connections per host, aria2 defaults to 5 concurrent downloads, IDM to 8
//! segments) — past that the marginal file gets slower and shared hosts start
//! refusing connections.
//!
//! Everything here is pure and synchronous so the policy is unit-testable without
//! a network; [`crate::DownloadCenter`] owns the sampling loop that feeds it.

/// Never drop below a single active transfer — 0 would deadlock the queue.
pub const MIN_SLOTS: usize = 1;
/// Ceiling on parallel transfers. See the module docs for why 8.
pub const MAX_SLOTS: usize = 8;

/// Improvement/regression band. A sample within ±10% of the best is "flat" —
/// noise, not signal, so it neither buys another slot nor costs one.
const BAND: f64 = 0.10;

/// One observation of the download engine, taken on a fixed interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThroughputSample {
    /// Sum of the per-task sampled speeds across actively streaming downloads.
    pub aggregate_bps: u64,
    /// How many downloads are streaming right now.
    pub active: usize,
    /// How many are waiting for a slot.
    pub queued: usize,
}

/// Seed slot count for a measured aggregate throughput, in bytes/sec.
///
/// Deliberately coarse: this only has to land in the right neighbourhood, because
/// [`AutoTuner::observe`] refines it against what actually happens next. The
/// thresholds are round numbers in the consumer-connection range (≈12 Mbps,
/// 50 Mbps, 125 Mbps, 320 Mbps).
pub fn slots_for_throughput(bps: u64) -> usize {
    const MB: u64 = 1024 * 1024;
    match bps {
        b if b < MB + MB / 2 => 2,
        b if b < 6 * MB => 3,
        b if b < 15 * MB => 4,
        b if b < 40 * MB => 6,
        _ => MAX_SLOTS,
    }
}

/// Hill-climbing controller over the slot count. Cheap to hold; one per center.
#[derive(Debug, Clone)]
pub struct AutoTuner {
    slots: usize,
    best_bps: u64,
    seeded: bool,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new(crate::DEFAULT_SLOTS)
    }
}

impl AutoTuner {
    pub fn new(initial_slots: usize) -> Self {
        Self {
            slots: initial_slots.clamp(MIN_SLOTS, MAX_SLOTS),
            best_bps: 0,
            seeded: false,
        }
    }

    /// The slot count the tuner currently recommends.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Best aggregate throughput observed so far (bytes/sec); `0` before the first
    /// meaningful sample. This is the evidence the UI shows next to "Auto".
    pub fn measured_bps(&self) -> u64 {
        self.best_bps
    }

    /// Fold one sample in and return the (possibly unchanged) slot recommendation.
    pub fn observe(&mut self, sample: ThroughputSample) -> usize {
        // Nothing is moving — there is no evidence in this sample either way.
        if sample.active == 0 || sample.aggregate_bps == 0 {
            return self.slots;
        }

        // First real sample: jump straight to the right neighbourhood.
        if !self.seeded {
            self.seeded = true;
            self.best_bps = sample.aggregate_bps;
            self.slots = slots_for_throughput(sample.aggregate_bps).clamp(MIN_SLOTS, MAX_SLOTS);
            return self.slots;
        }

        // Not slot-limited: fewer transfers running than allowed, or nothing
        // waiting behind them. The slot count is not what is holding throughput
        // back, so it must not be adjusted on this sample — but a new high-water
        // mark is still real evidence about the link and is worth keeping.
        if sample.active < self.slots || sample.queued == 0 {
            self.best_bps = self.best_bps.max(sample.aggregate_bps);
            return self.slots;
        }

        let best = self.best_bps as f64;
        let now = sample.aggregate_bps as f64;
        if now >= best * (1.0 + BAND) {
            // The last slot paid off — buy one more.
            self.best_bps = sample.aggregate_bps;
            self.slots = (self.slots + 1).min(MAX_SLOTS);
        } else if now <= best * (1.0 - BAND) {
            // Past the knee (or being throttled) — give one back.
            self.slots = self.slots.saturating_sub(1).max(MIN_SLOTS);
        }
        // Flat: this is the knee. Hold.
        self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;

    fn slot_limited(bps: u64, slots: usize) -> ThroughputSample {
        ThroughputSample {
            aggregate_bps: bps,
            active: slots,
            queued: 2,
        }
    }

    #[test]
    fn seed_scales_with_measured_bandwidth() {
        assert_eq!(slots_for_throughput(0), 2);
        assert_eq!(slots_for_throughput(MB), 2);
        assert_eq!(slots_for_throughput(3 * MB), 3);
        assert_eq!(slots_for_throughput(10 * MB), 4);
        assert_eq!(slots_for_throughput(20 * MB), 6);
        assert_eq!(slots_for_throughput(500 * MB), MAX_SLOTS);
    }

    #[test]
    fn first_sample_seeds_instead_of_stepping() {
        let mut t = AutoTuner::new(3);
        // A 20 MB/s link should not have to climb 3→4→5→6 one interval at a time.
        assert_eq!(t.observe(slot_limited(20 * MB, 3)), 6);
        assert_eq!(t.measured_bps(), 20 * MB);
    }

    #[test]
    fn idle_samples_never_move_the_slot_count() {
        let mut t = AutoTuner::new(4);
        let before = t.slots();
        t.observe(ThroughputSample {
            aggregate_bps: 0,
            active: 0,
            queued: 0,
        });
        assert_eq!(t.slots(), before);
    }

    /// The failure mode this guard exists for: a single small download trickling
    /// in with an empty queue is not evidence that parallelism is too high, and a
    /// controller that treats it as such walks itself down to 1 slot.
    #[test]
    fn unsaturated_samples_do_not_shrink_the_slot_count() {
        let mut t = AutoTuner::new(3);
        t.observe(slot_limited(10 * MB, 3)); // seed → 4
        let seeded = t.slots();
        for _ in 0..10 {
            t.observe(ThroughputSample {
                aggregate_bps: 100_000,
                active: 1,
                queued: 0,
            });
        }
        assert_eq!(t.slots(), seeded);
    }

    #[test]
    fn climbs_while_throughput_keeps_improving() {
        let mut t = AutoTuner::new(2);
        t.observe(slot_limited(MB, 2)); // seed → 2
        let mut bps = 2 * MB;
        for _ in 0..10 {
            let slots = t.slots();
            t.observe(slot_limited(bps, slots));
            bps *= 2;
        }
        assert_eq!(t.slots(), MAX_SLOTS, "must not climb past the ceiling");
    }

    #[test]
    fn backs_off_when_throughput_regresses() {
        let mut t = AutoTuner::new(4);
        t.observe(slot_limited(10 * MB, 4)); // seed → 4
        let peak = t.slots();
        // Throughput collapses (server throttling / link saturated).
        t.observe(slot_limited(MB, peak));
        assert_eq!(t.slots(), peak - 1);
    }

    #[test]
    fn never_backs_off_below_one_slot() {
        let mut t = AutoTuner::new(2);
        t.observe(slot_limited(MB, 2));
        for _ in 0..20 {
            let slots = t.slots();
            t.observe(slot_limited(1, slots));
        }
        assert_eq!(t.slots(), MIN_SLOTS);
    }

    #[test]
    fn holds_at_the_knee_when_throughput_is_flat() {
        let mut t = AutoTuner::new(3);
        t.observe(slot_limited(5 * MB, 3)); // seed → 3
        let knee = t.slots();
        for _ in 0..10 {
            // Within the ±10% band in both directions.
            t.observe(slot_limited(5 * MB + MB / 40, knee));
        }
        assert_eq!(t.slots(), knee);
    }
}
