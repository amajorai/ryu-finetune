//! Unified model insight for the agent-picker hover card.
//!
//! Cascade (first hit wins for base facts; AA always enriches when keyed):
//! 1. **models.dev** — cost, context, modalities, capabilities (built-in / ACP)
//! 2. **OpenRouter** — same shape when models.dev misses, or provider is OpenRouter
//! 3. **Artificial Analysis** — intelligence / speed / blended price when an AA
//!    key is configured (local models especially; also fills cloud gaps)
//!
//! Scores are 1–5 bars for the UI. Missing inputs leave the matching score
//! `None` so the hover card can hide that meter rather than invent numbers.

use serde::Serialize;

use crate::aa::{self, AaStats};
use crate::models_dev::{self, ModelMeta};
use crate::openrouter_meta;

/// Normalized hover-card payload returned by `GET /api/models/insight`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInsight {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Which registry supplied the base facts.
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_input_per_1m: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_output_per_1m: Option<f64>,
    pub modalities_input: Vec<String>,
    pub modalities_output: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knowledge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<bool>,
    /// Artificial Analysis match name when benchmarks were attached.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aa_matched_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intelligence_index: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_s: Option<f64>,
    /// 1–5 bars (higher = better). Cost is inverted (cheaper → higher).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_speed: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_cost: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_intelligence: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_context: Option<u8>,
    /// True when an AA API key is configured (even if this model didn't match).
    pub aa_key_present: bool,
}

/// Resolve a hover-card insight for `model` (optionally scoped by `provider`).
/// Returns `None` only when every source misses — the UI then skips the card.
pub async fn insight_for(model: &str, provider: Option<&str>) -> Option<ModelInsight> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut meta = models_dev::meta_for(trimmed, provider).await;
    let mut source = meta.as_ref().map(|m| m.source.clone()).unwrap_or_default();

    if meta.is_none() {
        if let Some(m) = openrouter_meta::meta_for(trimmed).await {
            source = m.source.clone();
            meta = Some(m);
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok();
    let aa_key_present = aa::has_api_key();
    let aa_stats = if aa_key_present {
        if let Some(ref c) = client {
            let name = meta.as_ref().map(|m| m.name.as_str()).unwrap_or(trimmed);
            // For local stems the "repo" is the stem itself; for cloud ids use
            // the bare model segment so AA's name matcher has something to chew.
            let repo = bare_model_id(trimmed);
            aa::stats_for(c, name, repo).await
        } else {
            None
        }
    } else {
        None
    };

    // Local-only path: no registry hit, but AA matched — still show a card.
    if meta.is_none() {
        let Some(stats) = aa_stats.as_ref() else {
            return None;
        };
        return Some(insight_from_aa_only(trimmed, stats, aa_key_present));
    }

    let meta = meta?;
    Some(build_insight(meta, source, aa_stats, aa_key_present))
}

fn bare_model_id(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn insight_from_aa_only(model: &str, stats: &AaStats, aa_key_present: bool) -> ModelInsight {
    let cost = stats.price_usd_per_1m;
    ModelInsight {
        id: model.to_owned(),
        name: if stats.matched_name.is_empty() {
            model.to_owned()
        } else {
            stats.matched_name.clone()
        },
        description: None,
        source: "artificial-analysis".to_owned(),
        context_tokens: None,
        max_output_tokens: None,
        cost_input_per_1m: cost,
        cost_output_per_1m: cost,
        modalities_input: vec!["text".to_owned()],
        modalities_output: vec!["text".to_owned()],
        knowledge: None,
        reasoning: None,
        tool_call: None,
        aa_matched_name: Some(stats.matched_name.clone()),
        intelligence_index: stats.intelligence_index,
        output_tokens_per_second: stats.output_tokens_per_second,
        time_to_first_token_s: stats.time_to_first_token_s,
        score_speed: stats.output_tokens_per_second.and_then(score_speed),
        score_cost: cost.and_then(score_blended_cost),
        score_intelligence: stats.intelligence_index.and_then(score_intelligence),
        score_context: None,
        aa_key_present,
    }
}

fn build_insight(
    meta: ModelMeta,
    source: String,
    aa: Option<AaStats>,
    aa_key_present: bool,
) -> ModelInsight {
    let mut cost_input = meta.cost_input_per_1m;
    let mut cost_output = meta.cost_output_per_1m;
    // AA blended price fills a total absence of pricing (typical for local).
    if cost_input.is_none() && cost_output.is_none() {
        if let Some(p) = aa.as_ref().and_then(|s| s.price_usd_per_1m) {
            cost_input = Some(p);
            cost_output = Some(p);
        }
    }

    let intelligence = aa.as_ref().and_then(|s| s.intelligence_index);
    let tps = aa.as_ref().and_then(|s| s.output_tokens_per_second);
    let ttft = aa.as_ref().and_then(|s| s.time_to_first_token_s);
    let heuristic = heuristic_intelligence(&meta);
    let context_score = meta.context.and_then(score_context);

    let blended = match (cost_input, cost_output) {
        (Some(i), Some(o)) => Some((3.0 * i + o) / 4.0),
        (Some(i), None) => Some(i),
        (None, Some(o)) => Some(o),
        _ => None,
    };

    ModelInsight {
        id: meta.id,
        name: meta.name,
        description: meta.description,
        source,
        context_tokens: meta.context,
        max_output_tokens: meta.max_output,
        cost_input_per_1m: cost_input,
        cost_output_per_1m: cost_output,
        modalities_input: meta.modalities_input,
        modalities_output: meta.modalities_output,
        knowledge: meta.knowledge,
        reasoning: meta.reasoning,
        tool_call: meta.tool_call,
        aa_matched_name: aa.as_ref().map(|s| s.matched_name.clone()),
        intelligence_index: intelligence,
        output_tokens_per_second: tps,
        time_to_first_token_s: ttft,
        score_speed: tps.and_then(score_speed),
        score_cost: blended.and_then(score_blended_cost),
        score_intelligence: intelligence.and_then(score_intelligence).or(heuristic),
        score_context: context_score,
        aa_key_present,
    }
}

/// Context bars: log-ish steps against common windows.
/// 8k→1, 32k→2, 128k→3, 200k→4, 1M+→5.
fn score_context(tokens: u32) -> Option<u8> {
    let score = if tokens >= 1_000_000 {
        5
    } else if tokens >= 200_000 {
        4
    } else if tokens >= 100_000 {
        3
    } else if tokens >= 32_000 {
        2
    } else if tokens >= 4_000 {
        1
    } else {
        return None;
    };
    Some(score)
}

/// Cost bars: cheaper blended $/1M → higher score.
/// ≤0.5→5, ≤2→4, ≤8→3, ≤30→2, else 1.
fn score_blended_cost(usd_per_1m: f64) -> Option<u8> {
    if !usd_per_1m.is_finite() || usd_per_1m < 0.0 {
        return None;
    }
    let score = if usd_per_1m <= 0.5 {
        5
    } else if usd_per_1m <= 2.0 {
        4
    } else if usd_per_1m <= 8.0 {
        3
    } else if usd_per_1m <= 30.0 {
        2
    } else {
        1
    };
    Some(score)
}

/// AA Intelligence Index (~0–100) → 1–5 bars.
fn score_intelligence(index: f64) -> Option<u8> {
    if !index.is_finite() || index < 0.0 {
        return None;
    }
    let score = if index >= 60.0 {
        5
    } else if index >= 45.0 {
        4
    } else if index >= 30.0 {
        3
    } else if index >= 15.0 {
        2
    } else {
        1
    };
    Some(score)
}

/// Output tok/s → 1–5. Cloud AA numbers often sit 20–200; local can be higher.
fn score_speed(tps: f64) -> Option<u8> {
    if !tps.is_finite() || tps <= 0.0 {
        return None;
    }
    let score = if tps >= 150.0 {
        5
    } else if tps >= 80.0 {
        4
    } else if tps >= 40.0 {
        3
    } else if tps >= 15.0 {
        2
    } else {
        1
    };
    Some(score)
}

/// Weak stand-in when AA isn't available: reasoning models get a bump.
fn heuristic_intelligence(meta: &ModelMeta) -> Option<u8> {
    match meta.reasoning {
        Some(true) => Some(4),
        Some(false) => Some(3),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_score_buckets() {
        assert_eq!(score_context(8_000), Some(1));
        assert_eq!(score_context(32_000), Some(2));
        assert_eq!(score_context(128_000), Some(3));
        assert_eq!(score_context(200_000), Some(4));
        assert_eq!(score_context(1_000_000), Some(5));
    }

    #[test]
    fn cost_score_inverts_price() {
        assert_eq!(score_blended_cost(0.2), Some(5));
        assert_eq!(score_blended_cost(1.0), Some(4));
        assert_eq!(score_blended_cost(5.0), Some(3));
        assert_eq!(score_blended_cost(20.0), Some(2));
        assert_eq!(score_blended_cost(100.0), Some(1));
    }

    #[test]
    fn intelligence_and_speed_buckets() {
        assert_eq!(score_intelligence(70.0), Some(5));
        assert_eq!(score_intelligence(20.0), Some(2));
        assert_eq!(score_speed(200.0), Some(5));
        assert_eq!(score_speed(10.0), Some(1));
    }
}
