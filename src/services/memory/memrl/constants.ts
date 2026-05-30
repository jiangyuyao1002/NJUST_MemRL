/**
 * MemRL Memory System — Path constants & hyperparameters.
 *
 * Dual-write strategy:
 *  - Primary:    .njust_ai/memories/  (episodic, LTM)
 *  - Secondary:  .roo/session-memories/  (acceptance criteria mirror)
 */

// ── Storage directories (relative to workspace root) ──────────────────────────
export const MEMRL_PRIMARY_DIR = ".njust_ai/memories"
export const MEMRL_ROO_DIR = ".roo/session-memories"

// File names inside the above directories
export const EPISODIC_FILE = "episodic.json"
export const LTM_FILE = "ltm_rules.json"

// ── Two-Phase Retrieval hyperparameters (paper defaults) ──────────────────────
/** Phase A: cosine similarity threshold to form candidate set */
export const SIM_THRESHOLD = 0.3
/** Phase A: max candidates to keep after threshold filter */
export const TOP_K1 = 20
/** Phase B: Q-weight in composite score: (1-λ)·sim̂ + λ·Q̂ */
export const LAMBDA = 0.3
/** Phase B: final top-K results to return */
export const TOP_K2 = 5

// ── Q-value update ─────────────────────────────────────────────────────────────
/** Monte Carlo learning rate: Q_new = Q_old + α·(r - Q_old) */
export const ALPHA = 0.1
/** Initial Q-value for new entries */
export const Q_INIT = 0.5

// ── LTM distillation ──────────────────────────────────────────────────────────
/** Trigger LTM distillation every N episodic writes */
export const LTM_DISTILL_INTERVAL = 10
/** Max recent episodes fed to LLM for distillation */
export const LTM_DISTILL_BATCH = 20
/** Max RuleCards to keep in LTM */
export const LTM_MAX_RULES = 200

// ── STM ───────────────────────────────────────────────────────────────────────
/** Max characters stored in a single ShortTermMemory */
export const STM_MAX_CHARS = 8_000
/** Max concurrent STM entries (LRU eviction above this) */
export const STM_LRU_LIMIT = 2_000
