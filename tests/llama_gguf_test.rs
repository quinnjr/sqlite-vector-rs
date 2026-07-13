//! Integration test: load a small GGUF model via llama-gguf, generate real
//! embeddings, store them in the vector table, KNN search, and verify
//! that the model can consume retrieved context for RAG-style generation.
//!
//! The model (~145 MB) is downloaded automatically on first run and cached
//! locally.  Execute with:
//!
//!     cargo test --test llama_gguf_test

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::open_with_extension;
use rusqlite::params;
use sqlite_vector_rs::types::slice_to_blob;

use llama_gguf::backend::cpu::CpuBackend;
use llama_gguf::model::embeddings::{EmbeddingConfig, EmbeddingExtractor};
use llama_gguf::model::load_llama_model;
use llama_gguf::sampling::{Sampler, SamplerConfig};
use llama_gguf::tokenizer::Tokenizer;
use llama_gguf::{Backend, GgufFile, InferenceContext, Model};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// HuggingFace direct download URL for SmolLM-135M at Q8_0 quantisation.
///
/// Q8_0 is used because llama-gguf 0.13.0 has a block-size mismatch bug for
/// IQ4_NL tensors (which Q2_K files use), causing silent tensor load failures.
/// Q8_0 uses the basic 32-element block format that works correctly.
const MODEL_URL: &str =
    "https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q8_0.gguf";

/// Local cache path (inside the build directory so it persists across runs).
fn model_cache_path() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&dir).ok();
    dir.join("SmolLM-135M.Q8_0.gguf")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Download the model if it isn't already cached.  Uses a simple HTTP GET
/// with `std::process::Command` calling curl, avoiding heavy HTTP client
/// dependencies.
///
/// A `Once` guard ensures only one thread performs the download when tests
/// run in parallel.
fn ensure_model() -> PathBuf {
    use std::sync::Once;
    static DOWNLOAD: Once = Once::new();

    let path = model_cache_path();

    DOWNLOAD.call_once(|| {
        if path.exists() && std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 1_000_000 {
            return;
        }

        eprintln!("Downloading SmolLM-135M Q8_0 (~145 MB) …");
        let tmp_path = path.with_extension("gguf.tmp");
        let status = std::process::Command::new("curl")
            .args(["-L", "-o"])
            .arg(&tmp_path)
            .arg(MODEL_URL)
            .arg("--fail")
            .arg("--silent")
            .arg("--show-error")
            .status()
            .expect("curl must be installed to download the test model");
        assert!(status.success(), "model download failed");

        // Atomic rename so other threads never see a partial file.
        std::fs::rename(&tmp_path, &path).expect("failed to rename downloaded model");
    });

    assert!(path.exists(), "model file missing after download");
    path
}

/// Load the model, tokenizer, and create an inference context + embedding
/// extractor.  Returns everything the tests need.
struct ModelBundle {
    model: llama_gguf::LlamaModel,
    tokenizer: Tokenizer,
    ctx: InferenceContext,
    extractor: EmbeddingExtractor,
}

fn load_model() -> ModelBundle {
    let path = ensure_model();
    let gguf = GgufFile::open(&path).expect("failed to open GGUF file");
    let model = load_llama_model(&path).expect("failed to load LLaMA model");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let ctx = InferenceContext::new(model.config(), backend);
    let extractor = EmbeddingExtractor::new(EmbeddingConfig::default(), model.config());
    ModelBundle {
        model,
        tokenizer,
        ctx,
        extractor,
    }
}

/// Generate an embedding for a short text.
fn embed(bundle: &mut ModelBundle, text: &str) -> Vec<f32> {
    bundle.ctx.reset();
    bundle
        .extractor
        .embed_text(&bundle.model, &bundle.tokenizer, &mut bundle.ctx, text)
        .unwrap_or_else(|e| {
            panic!(
                "embed_text failed for {:?}: {e}",
                &text[..text.len().min(40)]
            )
        })
}

/// Simple Shakespeare text chunks for testing (avoids PDF dependency in this
/// test file — keeps them small and deterministic).
const PASSAGES: &[&str] = &[
    "To be or not to be that is the question whether tis nobler in the mind to suffer",
    "Now is the winter of our discontent made glorious summer by this sun of York",
    "All the world is a stage and all the men and women merely players",
    "Romeo Romeo wherefore art thou Romeo deny thy father and refuse thy name",
    "Double double toil and trouble fire burn and cauldron bubble",
    "Out out brief candle life is but a walking shadow a poor player",
    "The quality of mercy is not strained it droppeth as the gentle rain from heaven",
    "Friends Romans countrymen lend me your ears I come to bury Caesar not to praise him",
    "If music be the food of love play on give me excess of it",
    "We are such stuff as dreams are made on and our little life is rounded with a sleep",
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn embedding_dimensions_match_model() {
    let mut bundle = load_model();
    let dim = bundle.extractor.embedding_dim();
    assert!(dim > 0, "embedding dim must be positive");

    let emb = embed(&mut bundle, "hello world");
    assert_eq!(
        emb.len(),
        dim,
        "embedding length {} != declared dim {dim}",
        emb.len()
    );
}

#[test]
fn embeddings_are_normalised() {
    let mut bundle = load_model();
    let emb = embed(&mut bundle, "to be or not to be");
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "embedding should be L2-normalised, got norm={norm}"
    );
}

#[test]
fn store_real_embeddings_and_knn_search() {
    let mut bundle = load_model();
    let dim = bundle.extractor.embedding_dim();
    let conn = open_with_extension();

    // Create vector table with the model's native embedding dimension.
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE docs USING vector(dim={dim}, type=float4, metric=cosine)"
    ))
    .unwrap();

    // Embed each passage and insert.
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for passage in PASSAGES {
        let emb = embed(&mut bundle, passage);
        let blob = slice_to_blob(&emb);
        conn.execute("INSERT INTO docs(vector) VALUES(?)", [blob.as_slice()])
            .unwrap();
        blobs.push(blob);
    }

    // Verify row count.
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, PASSAGES.len() as i64);

    // KNN search for a query related to Hamlet.
    let query_emb = embed(&mut bundle, "the question of existence and mortality");
    let query_blob = slice_to_blob(&query_emb);

    let mut stmt = conn
        .prepare("SELECT id, distance FROM docs WHERE knn_match(distance, ?) LIMIT 3")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 3, "expected 3 nearest neighbours");

    // Results must be ordered by ascending distance.
    for w in results.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "results not sorted: {} > {}",
            w[0].1,
            w[1].1
        );
    }

    // Cosine distance should be in [0, 2].
    for (id, dist) in &results {
        assert!(
            *dist >= 0.0 && *dist <= 2.0,
            "id {id}: cosine distance {dist} out of range"
        );
    }
}

#[test]
fn rag_retrieve_and_generate() {
    let mut bundle = load_model();
    let dim = bundle.extractor.embedding_dim();
    let conn = open_with_extension();

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE rag USING vector(dim={dim}, type=float4, metric=cosine)"
    ))
    .unwrap();

    // Insert passages.
    for passage in PASSAGES {
        let emb = embed(&mut bundle, passage);
        let blob = slice_to_blob(&emb);
        conn.execute("INSERT INTO rag(vector) VALUES(?)", [blob.as_slice()])
            .unwrap();
    }

    // Retrieve top-3 passages for a Hamlet-related query.
    let query_emb = embed(&mut bundle, "what does it mean to exist");
    let query_blob = slice_to_blob(&query_emb);

    let mut stmt = conn
        .prepare("SELECT id, distance FROM rag WHERE knn_match(distance, ?) LIMIT 3")
        .unwrap();
    let top_ids: Vec<i64> = stmt
        .query_map(params![query_blob.as_slice()], |row| row.get::<_, i64>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Build a prompt from retrieved context.
    let context: Vec<&str> = top_ids
        .iter()
        .map(|&id| PASSAGES[(id - 1) as usize]) // ids are 1-based
        .collect();
    let prompt = format!(
        "Context:\n{}\n\nQuestion: What is the meaning of existence?\nAnswer:",
        context.join("\n")
    );

    // Tokenize the prompt and run a short generation pass.
    let tokens = bundle
        .tokenizer
        .encode(&prompt, true)
        .expect("tokenization failed");
    assert!(
        !tokens.is_empty(),
        "tokenizer produced empty output for RAG prompt"
    );

    // Reset context for generation.
    bundle.ctx.reset();

    // Feed the prompt through the model (prefill).
    let vocab_size = bundle.model.config().vocab_size;
    let mut sampler = Sampler::new(SamplerConfig::default(), vocab_size);
    let mut all_tokens = tokens.clone();

    // Prefill: process all prompt tokens.
    let logits = bundle
        .model
        .forward(&tokens, &mut bundle.ctx)
        .expect("model forward pass failed on prompt");
    let logits_data = logits.as_f32().expect("logits must be f32");
    assert!(
        !logits_data.is_empty(),
        "model produced empty logits for prompt"
    );

    // Generate a few tokens autoregressively.
    let mut generated = Vec::new();
    let next_token = sampler.sample(&logits, &all_tokens);
    all_tokens.push(next_token);
    generated.push(next_token);

    for _ in 0..9 {
        let logits = bundle
            .model
            .forward(&all_tokens[all_tokens.len() - 1..], &mut bundle.ctx)
            .expect("model forward pass failed during generation");
        let next_token = sampler.sample(&logits, &all_tokens);
        all_tokens.push(next_token);
        generated.push(next_token);
    }

    // Decode generated tokens into text.
    let output_text = bundle
        .tokenizer
        .decode(&generated)
        .expect("decoding failed");

    // We don't assert semantic quality — just that the model produced
    // non-empty text and didn't crash, proving the full RAG pipeline works.
    assert!(
        !output_text.trim().is_empty(),
        "model generated empty text from RAG context"
    );

    eprintln!("RAG output (10 tokens): {output_text:?}");
}

#[test]
fn different_passages_produce_different_embeddings() {
    let mut bundle = load_model();

    let emb_a = embed(&mut bundle, PASSAGES[0]);
    let emb_b = embed(&mut bundle, PASSAGES[4]); // very different content

    // Embeddings should not be identical.
    assert_ne!(
        emb_a, emb_b,
        "different passages must produce different embeddings"
    );

    // Compute cosine similarity — should be < 1.0.
    let dot: f32 = emb_a.iter().zip(emb_b.iter()).map(|(a, b)| a * b).sum();
    assert!(
        dot < 0.999,
        "cosine similarity between very different passages should be < 1, got {dot}"
    );
}

#[test]
fn batch_embed_multiple_passages() {
    let mut bundle = load_model();
    let dim = bundle.extractor.embedding_dim();

    let texts: Vec<&str> = PASSAGES.to_vec();
    let embeddings = bundle
        .extractor
        .embed_batch(&bundle.model, &bundle.tokenizer, &mut bundle.ctx, &texts)
        .expect("embed_batch failed");

    assert_eq!(embeddings.len(), PASSAGES.len());
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            dim,
            "passage {i}: embedding dim mismatch ({} != {dim})",
            emb.len()
        );
    }
}
