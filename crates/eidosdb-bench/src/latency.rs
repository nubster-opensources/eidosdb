//! Latency percentile summary over a set of durations.

use std::time::Duration;

/// Percentile summary of a batch of measured durations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencySummary {
    /// Median latency.
    pub p50: Duration,
    /// 99th percentile latency.
    pub p99: Duration,
}

/// Computes p50 and p99 from `samples` using nearest-rank on a sorted copy.
///
/// Returns zeroed durations when `samples` is empty.
#[must_use]
pub fn summarize(samples: &[Duration]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary { p50: Duration::ZERO, p99: Duration::ZERO };
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    LatencySummary {
        p50: percentile(&sorted, 0.50),
        p99: percentile(&sorted, 0.99),
    }
}

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    // Nearest-rank: rank = ceil(fraction * n), clamped to [1, n], 1-indexed.
    let n = sorted.len();
    let rank = (fraction * n as f64).ceil() as usize;
    let index = rank.clamp(1, n) - 1;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::{summarize, LatencySummary};
    use std::time::Duration;

    #[test]
    fn empty_is_zeroed() {
        assert_eq!(
            summarize(&[]),
            LatencySummary { p50: Duration::ZERO, p99: Duration::ZERO }
        );
    }

    #[test]
    fn computes_percentiles_on_a_ramp() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let summary = summarize(&samples);
        assert_eq!(summary.p50, Duration::from_millis(50));
        assert_eq!(summary.p99, Duration::from_millis(99));
    }
}
