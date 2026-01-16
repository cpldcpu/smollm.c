/// SmolLM2 Rust Inference Engine
/// Rust implementation of SmolLM2-135M with Q8 quantization

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

const QUANT_Q8: u32 = 0;

#[derive(Debug, Clone)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    head_dim: usize,
}

#[derive(Clone)]
struct Q8Tensor {
    scale: f32,
    data: Vec<i8>,
    rows: usize,
    cols: usize,
}

#[derive(Clone)]
struct FP32Tensor {
    data: Vec<f32>,
}

#[derive(Clone)]
struct LayerWeights {
    input_layernorm: FP32Tensor,
    q_proj: Q8Tensor,
    k_proj: Q8Tensor,
    v_proj: Q8Tensor,
    o_proj: Q8Tensor,
    post_attention_layernorm: FP32Tensor,
    gate_proj: Q8Tensor,
    up_proj: Q8Tensor,
    down_proj: Q8Tensor,
}

struct ModelWeights {
    embed_tokens: Q8Tensor,
    layers: Vec<LayerWeights>,
    final_norm: FP32Tensor,
}

struct KVCache {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    cache_len: usize,
}

struct Tokenizer {
    vocab: Vec<String>,
    merges: Vec<String>,
}

struct SmolLM2 {
    config: Config,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    kv_caches: Vec<KVCache>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    // Scratch buffers
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
}

// Helper functions for reading binary data
fn read_u32(reader: &mut BufReader<File>) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(reader: &mut BufReader<File>) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_bytes(reader: &mut BufReader<File>, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_q8(reader: &mut BufReader<File>, rows: usize, cols: usize) -> std::io::Result<Q8Tensor> {
    let scale = read_f32(reader)?;
    let data_bytes = read_bytes(reader, rows * cols)?;
    let data: Vec<i8> = data_bytes.into_iter().map(|b| b as i8).collect();
    Ok(Q8Tensor { scale, data, rows, cols })
}

fn read_fp32(reader: &mut BufReader<File>, n: usize) -> std::io::Result<FP32Tensor> {
    let mut data = vec![0f32; n];
    for i in 0..n {
        data[i] = read_f32(reader)?;
    }
    Ok(FP32Tensor { data })
}

// Math functions
fn rmsnorm(o: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let ss = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        o[i] = x[i] * ss * w[i];
    }
}

fn matmul_q8(o: &mut [f32], w: &Q8Tensor, x: &[f32]) {
    let scale = w.scale;
    for i in 0..w.rows {
        let mut sum = 0.0f32;
        let row_start = i * w.cols;
        for j in 0..w.cols {
            sum += w.data[row_start + j] as f32 * scale * x[j];
        }
        o[i] = sum;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn precompute_rope(head_dim: usize, max_seq_len: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; max_seq_len * head_dim];
    let mut sin = vec![0.0f32; max_seq_len * head_dim];

    for p in 0..max_seq_len {
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let c = (p as f32 * freq).cos();
            let s = (p as f32 * freq).sin();
            cos[p * head_dim + i] = c;
            cos[p * head_dim + head_dim / 2 + i] = c;
            sin[p * head_dim + i] = s;
            sin[p * head_dim + head_dim / 2 + i] = s;
        }
    }
    (cos, sin)
}

fn apply_rope(v: &mut [f32], hd: usize, cos: &[f32], sin: &[f32]) {
    let h = hd / 2;
    for i in 0..h {
        let v0 = v[i];
        let v1 = v[i + h];
        v[i] = v0 * cos[i] - v1 * sin[i];
        v[i + h] = v1 * cos[i + h] + v0 * sin[i + h];
    }
}

impl SmolLM2 {
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read magic
        let magic = read_bytes(&mut reader, 4)?;
        if magic != b"SMOL" {
            return Err("Bad magic".into());
        }

        // Read version
        let ver = read_u32(&mut reader)?;
        if ver != 1 && ver != 2 {
            return Err("Bad version".into());
        }

        // Handle v2 format
        if ver == 2 {
            let qt = read_u32(&mut reader)?;
            let _ = read_u32(&mut reader)?; // group_size
            if qt != QUANT_Q8 {
                return Err("Q8 only supported".into());
            }
        }

        // Read config
        let hidden_size = read_u32(&mut reader)? as usize;
        let intermediate_size = read_u32(&mut reader)? as usize;
        let num_layers = read_u32(&mut reader)? as usize;
        let num_heads = read_u32(&mut reader)? as usize;
        let num_kv_heads = read_u32(&mut reader)? as usize;
        let vocab_size = read_u32(&mut reader)? as usize;
        let max_seq_len = read_u32(&mut reader)? as usize;
        let rope_theta = read_f32(&mut reader)?;
        let rms_norm_eps = read_f32(&mut reader)?;
        let head_dim = hidden_size / num_heads;

        let config = Config {
            hidden_size, intermediate_size, num_layers, num_heads,
            num_kv_heads, vocab_size, max_seq_len, rope_theta, rms_norm_eps, head_dim
        };

        println!("Config: hidden={} layers={} heads={} kv={} vocab={}",
                 hidden_size, num_layers, num_heads, num_kv_heads, vocab_size);

        // Read tokenizer
        let nv = read_u32(&mut reader)? as usize;
        let nm = read_u32(&mut reader)? as usize;

        let mut vocab = Vec::with_capacity(nv);
        for _ in 0..nv {
            let len = read_u32(&mut reader)? as usize;
            let bytes = read_bytes(&mut reader, len)?;
            vocab.push(String::from_utf8_lossy(&bytes).to_string());
        }

        let mut merges = Vec::with_capacity(nm);
        for _ in 0..nm {
            let len = read_u32(&mut reader)? as usize;
            let bytes = read_bytes(&mut reader, len)?;
            merges.push(String::from_utf8_lossy(&bytes).to_string());
        }

        println!("Tokenizer: {} vocab, {} merges", nv, nm);
        let tokenizer = Tokenizer { vocab, merges };

        // Read weights
        let embed_tokens = read_q8(&mut reader, vocab_size, hidden_size)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let input_layernorm = read_fp32(&mut reader, hidden_size)?;
            let q_proj = read_q8(&mut reader, num_heads * head_dim, hidden_size)?;
            let k_proj = read_q8(&mut reader, num_kv_heads * head_dim, hidden_size)?;
            let v_proj = read_q8(&mut reader, num_kv_heads * head_dim, hidden_size)?;
            let o_proj = read_q8(&mut reader, hidden_size, num_heads * head_dim)?;
            let post_attention_layernorm = read_fp32(&mut reader, hidden_size)?;
            let gate_proj = read_q8(&mut reader, intermediate_size, hidden_size)?;
            let up_proj = read_q8(&mut reader, intermediate_size, hidden_size)?;
            let down_proj = read_q8(&mut reader, hidden_size, intermediate_size)?;

            layers.push(LayerWeights {
                input_layernorm, q_proj, k_proj, v_proj, o_proj,
                post_attention_layernorm, gate_proj, up_proj, down_proj,
            });
        }

        let final_norm = read_fp32(&mut reader, hidden_size)?;
        let weights = ModelWeights { embed_tokens, layers, final_norm };

        // Precompute RoPE
        let (rope_cos, rope_sin) = precompute_rope(head_dim, max_seq_len, rope_theta);

        // Initialize KV caches
        let mut kv_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            kv_caches.push(KVCache {
                k_cache: vec![0.0f32; num_kv_heads * max_seq_len * head_dim],
                v_cache: vec![0.0f32; num_kv_heads * max_seq_len * head_dim],
                cache_len: 0,
            });
        }

        // Initialize scratch buffers
        let x = vec![0.0f32; hidden_size];
        let xb = vec![0.0f32; hidden_size];
        let xb2 = vec![0.0f32; hidden_size];
        let q = vec![0.0f32; num_heads * head_dim];
        let k = vec![0.0f32; num_kv_heads * head_dim];
        let v = vec![0.0f32; num_kv_heads * head_dim];
        let att = vec![0.0f32; num_heads * max_seq_len];
        let logits = vec![0.0f32; vocab_size];
        let hb = vec![0.0f32; intermediate_size];
        let hb2 = vec![0.0f32; intermediate_size];

        println!("Model loaded");

        Ok(SmolLM2 {
            config, weights, tokenizer, kv_caches, rope_cos, rope_sin,
            x, xb, xb2, q, k, v, att, logits, hb, hb2,
        })
    }

    fn reset_cache(&mut self) {
        for kv in &mut self.kv_caches {
            kv.cache_len = 0;
        }
    }

    fn forward(&mut self, tok: usize, pos: usize) -> &[f32] {
        let c = &self.config;
        let hs = c.hidden_size;
        let hd = c.head_dim;
        let nh = c.num_heads;
        let nkv = c.num_kv_heads;
        let ng = nh / nkv;

        // Embedding
        let scale = self.weights.embed_tokens.scale;
        let emb_start = tok * hs;
        for i in 0..hs {
            self.x[i] = self.weights.embed_tokens.data[emb_start + i] as f32 * scale;
        }

        let rope_start = pos * hd;

        for l in 0..c.num_layers {
            // Input layernorm
            rmsnorm(&mut self.xb, &self.x, &self.weights.layers[l].input_layernorm.data, c.rms_norm_eps);

            // QKV projections
            matmul_q8(&mut self.q, &self.weights.layers[l].q_proj, &self.xb);
            matmul_q8(&mut self.k, &self.weights.layers[l].k_proj, &self.xb);
            matmul_q8(&mut self.v, &self.weights.layers[l].v_proj, &self.xb);

            // Apply RoPE
            for h in 0..nh {
                let start = h * hd;
                apply_rope(
                    &mut self.q[start..start + hd],
                    hd,
                    &self.rope_cos[rope_start..rope_start + hd],
                    &self.rope_sin[rope_start..rope_start + hd],
                );
            }
            for h in 0..nkv {
                let start = h * hd;
                apply_rope(
                    &mut self.k[start..start + hd],
                    hd,
                    &self.rope_cos[rope_start..rope_start + hd],
                    &self.rope_sin[rope_start..rope_start + hd],
                );
            }

            // Update KV cache
            let cache_offset = self.kv_caches[l].cache_len * hd;
            for h in 0..nkv {
                let src_start = h * hd;
                let dst_start = h * c.max_seq_len * hd + cache_offset;
                for d in 0..hd {
                    self.kv_caches[l].k_cache[dst_start + d] = self.k[src_start + d];
                    self.kv_caches[l].v_cache[dst_start + d] = self.v[src_start + d];
                }
            }
            let slen = self.kv_caches[l].cache_len + 1;

            // Attention
            self.xb2.fill(0.0);
            for h in 0..nh {
                let kvh = h / ng;
                let qh_start = h * hd;
                let kc_start = kvh * c.max_seq_len * hd;
                let vc_start = kvh * c.max_seq_len * hd;
                let att_start = h * c.max_seq_len;
                let scale = 1.0 / (hd as f32).sqrt();

                // Compute attention scores
                for t in 0..slen {
                    let kt_start = kc_start + t * hd;
                    let mut score = 0.0f32;
                    for d in 0..hd {
                        score += self.q[qh_start + d] * self.kv_caches[l].k_cache[kt_start + d];
                    }
                    self.att[att_start + t] = score * scale;
                }

                // Softmax
                softmax(&mut self.att[att_start..att_start + slen]);

                // Weighted sum of values
                let oh_start = h * hd;
                for t in 0..slen {
                    let a = self.att[att_start + t];
                    let vt_start = vc_start + t * hd;
                    for d in 0..hd {
                        self.xb2[oh_start + d] += a * self.kv_caches[l].v_cache[vt_start + d];
                    }
                }
            }

            // Output projection and residual
            matmul_q8(&mut self.xb, &self.weights.layers[l].o_proj, &self.xb2);
            for i in 0..hs {
                self.x[i] += self.xb[i];
            }

            // Post-attention layernorm
            rmsnorm(&mut self.xb, &self.x, &self.weights.layers[l].post_attention_layernorm.data, c.rms_norm_eps);

            // MLP
            matmul_q8(&mut self.hb, &self.weights.layers[l].gate_proj, &self.xb);
            matmul_q8(&mut self.hb2, &self.weights.layers[l].up_proj, &self.xb);
            for i in 0..c.intermediate_size {
                self.hb[i] = silu(self.hb[i]) * self.hb2[i];
            }
            matmul_q8(&mut self.xb, &self.weights.layers[l].down_proj, &self.hb);
            for i in 0..hs {
                self.x[i] += self.xb[i];
            }

            self.kv_caches[l].cache_len = slen;
        }

        // Final norm
        rmsnorm(&mut self.xb, &self.x, &self.weights.final_norm.data, c.rms_norm_eps);
        std::mem::swap(&mut self.x, &mut self.xb);

        // LM head (tied weights)
        let scale = self.weights.embed_tokens.scale;
        for i in 0..c.vocab_size {
            let mut sum = 0.0f32;
            let row_start = i * hs;
            for j in 0..hs {
                sum += self.x[j] * self.weights.embed_tokens.data[row_start + j] as f32 * scale;
            }
            self.logits[i] = sum;
        }

        &self.logits
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        // Convert to GPT2 byte encoding
        let mut encoded = String::new();
        let mut start = true;
        for c in text.chars() {
            if c == ' ' {
                if !start {
                    encoded.push_str("\u{0120}"); // Ġ
                }
            } else if c == '\n' {
                encoded.push_str("\u{010A}"); // Ċ
                start = true;
            } else {
                encoded.push(c);
                start = false;
            }
        }

        // Initial tokenization - greedy longest match
        let mut tokens: Vec<usize> = Vec::new();
        let chars: Vec<char> = encoded.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;
            for len in 1..=32.min(chars.len() - i) {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(id) = self.tokenizer.vocab.iter().position(|v| v == &substr) {
                    best_len = len;
                    best_id = Some(id);
                }
            }
            if let Some(id) = best_id {
                tokens.push(id);
                i += best_len;
            } else {
                // Single char fallback
                let c: String = chars[i..i + 1].iter().collect();
                if let Some(id) = self.tokenizer.vocab.iter().position(|v| v == &c) {
                    tokens.push(id);
                }
                i += 1;
            }
        }

        // Apply BPE merges
        let mut changed = true;
        let mut iter = 0;
        while changed && tokens.len() > 1 && iter < 1000 {
            changed = false;
            iter += 1;
            for merge in &self.tokenizer.merges {
                if let Some(sp) = merge.find(' ') {
                    let t1 = &merge[..sp];
                    let t2 = &merge[sp + 1..];
                    for j in 0..tokens.len() - 1 {
                        if self.tokenizer.vocab[tokens[j]] == t1 &&
                           self.tokenizer.vocab[tokens[j + 1]] == t2 {
                            let merged = format!("{}{}", t1, t2);
                            if let Some(mid) = self.tokenizer.vocab.iter().position(|v| v == &merged) {
                                tokens[j] = mid;
                                tokens.remove(j + 1);
                                changed = true;
                                break;
                            }
                        }
                    }
                    if changed {
                        break;
                    }
                }
            }
        }

        tokens
    }

    fn decode(&self, tok: usize) -> String {
        if tok >= self.tokenizer.vocab.len() {
            return String::new();
        }
        let mut result = String::new();
        let vocab_entry = &self.tokenizer.vocab[tok];
        let chars: Vec<char> = vocab_entry.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '\u{0120}' { // Ġ
                result.push(' ');
            } else if chars[i] == '\u{010A}' { // Ċ
                result.push('\n');
            } else {
                result.push(chars[i]);
            }
            i += 1;
        }
        result
    }

    fn sample(&self, logits: &[f32], temp: f32) -> usize {
        if temp <= 0.0 {
            // Greedy
            logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            // Temperature sampling
            let mut probs: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
            softmax(&mut probs);
            let r: f32 = rand::random();
            let mut cs = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cs += p;
                if cs >= r {
                    return i;
                }
            }
            probs.len() - 1
        }
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize, temp: f32) {
        let tokens = self.tokenize(prompt);
        if tokens.is_empty() {
            println!("Tokenize failed");
            return;
        }

        print!("Tokens: ");
        for t in &tokens {
            print!("{} ", t);
        }
        println!("\n");
        print!("{}", prompt);

        self.reset_cache();

        // Process prompt
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let l = self.forward(tok, i);
            logits = l.to_vec();
        }

        // Generate
        for i in 0..max_tokens {
            let next = self.sample(&logits, temp);
            if next == 2 { // EOS token
                break;
            }
            print!("{}", self.decode(next));
            use std::io::Write;
            std::io::stdout().flush().unwrap();
            let l = self.forward(next, tokens.len() + i);
            logits = l.to_vec();
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model_path = "../models/smollm2-135m-q8.bin".to_string();
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 50;
    let mut temp = 0.0f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 1;
            }
            "-p" if i + 1 < args.len() => {
                prompt = args[i + 1].clone();
                i += 1;
            }
            "-n" if i + 1 < args.len() => {
                max_tokens = args[i + 1].parse().unwrap_or(50);
                i += 1;
            }
            "-t" if i + 1 < args.len() => {
                temp = args[i + 1].parse().unwrap_or(0.0);
                i += 1;
            }
            "-h" => {
                println!("Usage: smolr [-m model] [-p prompt] [-n tokens] [-t temp]");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading {}...", model_path);
    match SmolLM2::load(&model_path) {
        Ok(mut model) => {
            model.generate(&prompt, max_tokens, temp);
        }
        Err(e) => {
            eprintln!("Error loading model: {}", e);
        }
    }
}
