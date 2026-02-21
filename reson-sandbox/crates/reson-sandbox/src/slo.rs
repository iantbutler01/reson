#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SloBudget {
    pub metric: &'static str,
    pub p95_ms: Option<u64>,
    pub p99_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SloSample {
    pub metric: String,
    pub p95_ms: Option<u64>,
    pub p99_ms: Option<u64>,
}

pub const DEFAULT_SLO_BUDGETS: &[SloBudget] = &[
    SloBudget {
        metric: "session.attach",
        p95_ms: Some(2_000),
        p99_ms: None,
    },
    SloBudget {
        metric: "control.command.dispatch",
        p95_ms: Some(500),
        p99_ms: Some(1_500),
    },
    SloBudget {
        metric: "exec.stream.establish.warm_vm",
        p95_ms: Some(1_500),
        p99_ms: Some(4_000),
    },
    SloBudget {
        metric: "session.create.warm_pool",
        p95_ms: Some(8_000),
        p99_ms: Some(20_000),
    },
    SloBudget {
        metric: "session.create.cold_cache_hit",
        p95_ms: Some(30_000),
        p99_ms: None,
    },
];

pub fn evaluate_slo(samples: &[SloSample], budgets: &[SloBudget]) -> Vec<String> {
    let mut violations = Vec::new();
    for sample in samples {
        if let Some(budget) = budgets.iter().find(|budget| budget.metric == sample.metric) {
            if let (Some(observed), Some(limit)) = (sample.p95_ms, budget.p95_ms)
                && observed > limit
            {
                violations.push(format!(
                    "{} p95 exceeded: observed={}ms budget={}ms",
                    sample.metric, observed, limit
                ));
            }
            if let (Some(observed), Some(limit)) = (sample.p99_ms, budget.p99_ms)
                && observed > limit
            {
                violations.push(format!(
                    "{} p99 exceeded: observed={}ms budget={}ms",
                    sample.metric, observed, limit
                ));
            }
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SLO_BUDGETS, SloSample, evaluate_slo};

    #[test]
    fn slo_budget_passes_when_samples_are_within_limits() {
        let samples = vec![
            SloSample {
                metric: "session.attach".to_string(),
                p95_ms: Some(1_500),
                p99_ms: None,
            },
            SloSample {
                metric: "control.command.dispatch".to_string(),
                p95_ms: Some(400),
                p99_ms: Some(1_300),
            },
        ];

        let violations = evaluate_slo(&samples, DEFAULT_SLO_BUDGETS);
        assert!(
            violations.is_empty(),
            "expected no violations: {violations:?}"
        );
    }

    #[test]
    fn slo_budget_fails_when_sample_exceeds_budget() {
        let samples = vec![SloSample {
            metric: "exec.stream.establish.warm_vm".to_string(),
            p95_ms: Some(1_700),
            p99_ms: Some(4_800),
        }];

        let violations = evaluate_slo(&samples, DEFAULT_SLO_BUDGETS);
        assert_eq!(violations.len(), 2);
        assert!(violations[0].contains("p95 exceeded"));
        assert!(violations[1].contains("p99 exceeded"));
    }
}
