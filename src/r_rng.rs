//! R's default random number generator, bit for bit: Mersenne-Twister seeded
//! by `set.seed`, `unif_rand`, and `sample()` without replacement under the
//! "Rejection" sample kind (R ≥ 3.6). mgcv's `bam` draws its basis-setup rows
//! with it, so reproducing bam above `chunk.size` rows needs the same draws.

const N: usize = 624;
const M: usize = 397;

pub struct RRng {
    mt: [u32; N],
    mti: usize,
}

impl RRng {
    /// `set.seed(seed)` with `kind = "Mersenne-Twister"`.
    pub fn set_seed(seed: u32) -> Self {
        let mut s = seed;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        // R fills dummy[0..625]; dummy[0] (the position) is then reset to N.
        s = s.wrapping_mul(69069).wrapping_add(1);
        let mut mt = [0u32; N];
        for v in mt.iter_mut() {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *v = s;
        }
        Self { mt, mti: N }
    }

    fn genrand(&mut self) -> f64 {
        const MAG01: [u32; 2] = [0, 0x9908_b0df];
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        if self.mti >= N {
            for kk in 0..N - M {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            for kk in N - M..N - 1 {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER) | (self.mt[0] & LOWER);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y as f64 * 2.328_306_436_538_696_3e-10
    }

    /// `unif_rand()`: uniform on (0, 1).
    pub fn unif_rand(&mut self) -> f64 {
        const I2_32M1: f64 = 2.328_306_437_080_797e-10;
        let v = self.genrand();
        if v <= 0.0 {
            0.5 * I2_32M1
        } else if 1.0 - v <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            v
        }
    }

    fn rbits(&mut self, bits: u32) -> f64 {
        let mut v: i64 = 0;
        let mut n = 0;
        while n <= bits {
            let v1 = (self.unif_rand() * 65536.0).floor() as i64;
            v = 65536 * v + v1;
            n += 16;
        }
        (v & ((1i64 << bits) - 1)) as f64
    }

    /// `R_unif_index(dn)`: a uniform integer in `[0, dn)` by rejection.
    pub fn unif_index(&mut self, dn: f64) -> f64 {
        if dn <= 0.0 {
            return 0.0;
        }
        let bits = dn.log2().ceil() as u32;
        loop {
            let dv = self.rbits(bits);
            if dn > dv {
                return dv;
            }
        }
    }

    /// `sample.int(n, size)` without replacement, as 0-based indices.
    pub fn sample(&mut self, n: usize, size: usize) -> Vec<usize> {
        let mut x: Vec<usize> = (0..n).collect();
        let mut left = n;
        (0..size)
            .map(|_| {
                let j = self.unif_index(left as f64) as usize;
                let out = x[j];
                left -= 1;
                x[j] = x[left];
                out
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::RRng;

    #[derive(serde::Deserialize)]
    struct Draw {
        seed: u32,
        n: usize,
        size: usize,
        /// `runif` draws times 2³², exact integers so the JSON round trip is exact.
        unif_2_32: Vec<f64>,
        sample: Vec<usize>,
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        draws: Vec<Draw>,
    }

    #[test]
    fn matches_r_runif_and_sample() {
        let fx: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/r_rng_and_mini_mf.json")).unwrap();
        for d in fx.draws {
            let mut rng = RRng::set_seed(d.seed);
            let u: Vec<f64> = (0..d.unif_2_32.len())
                .map(|_| (rng.unif_rand() * 4294967296.0).round())
                .collect();
            assert_eq!(u, d.unif_2_32, "runif, seed {}", d.seed);
            let s: Vec<usize> = RRng::set_seed(d.seed)
                .sample(d.n, d.size)
                .into_iter()
                .map(|i| i + 1)
                .collect();
            assert_eq!(s, d.sample, "sample({}, {}), seed {}", d.n, d.size, d.seed);
        }
    }
}
