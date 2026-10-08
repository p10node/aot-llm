//! Token sampling: greedy, temperature, top-k, top-p and repetition penalty.

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `0` selects the arg-max token.
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    /// `1.0` disables top-p.
    pub top_p: f32,
    /// `1.0` disables the penalty.
    pub repeat_penalty: f32,
    /// Window of recent tokens the penalty applies to.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: 40, top_p: 0.95, repeat_penalty: 1.0, repeat_last_n: 64, seed: 42 }
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: u64,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = splitmix(cfg.seed);
        Sampler { cfg, rng, cand: Vec::new() }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token from `logits` (modified in place). `recent` holds
    /// previously generated tokens for the repetition penalty.
    pub fn sample(&mut self, logits: &mut [f32], recent: &[u32]) -> u32 {
        if self.cfg.repeat_penalty != 1.0 && self.cfg.repeat_last_n > 0 {
            let start = recent.len().saturating_sub(self.cfg.repeat_last_n);
            for &t in &recent[start..] {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 { *l / self.cfg.repeat_penalty } else { *l * self.cfg.repeat_penalty };
            }
        }
        if self.cfg.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.cfg.temperature;
        self.cand.clear();
        self.cand.extend(logits.iter().enumerate().map(|(i, &l)| (l * inv_t, i as u32)));
        let k = if self.cfg.top_k > 0 { self.cfg.top_k.min(self.cand.len()) } else { self.cand.len() };
        if k < self.cand.len() {
            // Partition so the k largest come first, then sort just those.
            self.cand.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            self.cand.truncate(k);
        }
        self.cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let max = self.cand[0].0;
        let mut sum = 0f32;
        for c in self.cand.iter_mut() {
            c.0 = (c.0 - max).exp();
            sum += c.0;
        }
        let mut n = self.cand.len();
        if self.cfg.top_p < 1.0 {
            let mut acc = 0f32;
            for (i, c) in self.cand.iter().enumerate() {
                acc += c.0 / sum;
                if acc >= self.cfg.top_p {
                    n = i + 1;
                    break;
                }
            }
            sum = self.cand[..n].iter().map(|c| c.0).sum();
        }
        let r = self.next_f32() * sum;
        let mut acc = 0f32;
        for c in &self.cand[..n] {
            acc += c.0;
            if acc >= r {
                return c.1;
            }
        }
        self.cand[n - 1].1
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

fn splitmix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplerConfig::default());
        let mut l = vec![0.1, 5.0, 2.0];
        assert_eq!(s.sample(&mut l, &[]), 1);
    }

    #[test]
    fn temperature_sampling_respects_top_k() {
        let cfg = SamplerConfig { temperature: 1.0, top_k: 2, top_p: 1.0, seed: 7, ..Default::default() };
        let mut s = Sampler::new(cfg);
        for _ in 0..200 {
            let mut l = vec![0.0, 10.0, 9.0, -5.0];
            let t = s.sample(&mut l, &[]);
            assert!(t == 1 || t == 2);
        }
    }

    #[test]
    fn repeat_penalty_lowers_recent() {
        let cfg = SamplerConfig { repeat_penalty: 100.0, ..Default::default() };
        let mut s = Sampler::new(cfg);
        let mut l = vec![1.0, 1.1];
        assert_eq!(s.sample(&mut l, &[1]), 0);
    }
}
