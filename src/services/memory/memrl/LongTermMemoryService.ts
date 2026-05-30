/**
 * LongTermMemoryService
 *
 * Manages RuleCard-based long-term memory.
 * LLM distillation converts recent episodic episodes into reusable rules.
 * Dual-write: .njust_ai/memories/ + .roo/session-memories/
 */

import * as fs from "fs/promises"
import * as path from "path"
import { safeWriteJson } from "../../../utils/safeWriteJson"
import type { ApiHandler } from "../../../api"
import { MemoryEmbeddingAdapter } from "./MemoryEmbeddingAdapter"
import type { EpisodicEntry } from "./EpisodicMemoryService"
import {
	LAMBDA,
	LTM_DISTILL_BATCH,
	LTM_FILE,
	LTM_MAX_RULES,
	MEMRL_PRIMARY_DIR,
	MEMRL_ROO_DIR,
	SIM_THRESHOLD,
	TOP_K1,
	TOP_K2,
} from "./constants"

export interface RuleCard {
	id: string
	topic: string
	rule: string
	examples: string[]
	/** Confidence ∈ [0, 1] — updated via usage feedback */
	confidence: number
	useCount: number
	createdAt: number
	updatedAt: number
	/** Embedding of topic+rule for retrieval */
	embedding: number[]
}

interface LtmStore {
	rules: RuleCard[]
}

export class LongTermMemoryService {
	private store: LtmStore = { rules: [] }
	private loaded = false

	constructor(
		private readonly workspaceDir: string,
		private readonly embedder: MemoryEmbeddingAdapter,
		private readonly api: ApiHandler,
	) {}

	// ── Persistence ────────────────────────────────────────────────────────────

	private get primaryPath(): string {
		return path.join(this.workspaceDir, MEMRL_PRIMARY_DIR, LTM_FILE)
	}

	private get rooPath(): string {
		return path.join(this.workspaceDir, MEMRL_ROO_DIR, LTM_FILE)
	}

	async load(): Promise<void> {
		if (this.loaded) return
		try {
			const raw = await fs.readFile(this.primaryPath, "utf-8")
			this.store = JSON.parse(raw) as LtmStore
		} catch {
			this.store = { rules: [] }
		}
		this.loaded = true
	}

	private async persist(): Promise<void> {
		await safeWriteJson(this.primaryPath, this.store)
		await safeWriteJson(this.rooPath, this.store).catch(() => {
			/* best-effort mirror */
		})
	}

	// ── Distillation (fire-and-forget from MemoryManager) ─────────────────────

	/**
	 * LLM-based distillation of recent episodic entries into RuleCards.
	 * This is called fire-and-forget; errors are silently swallowed.
	 */
	async distill(recentEpisodes: EpisodicEntry[]): Promise<void> {
		await this.load()

		if (recentEpisodes.length === 0) return

		const episodeSummaries = recentEpisodes
			.slice(0, LTM_DISTILL_BATCH)
			.map(
				(e, i) =>
					`Episode ${i + 1}:\nIntent: ${e.intent}\nSummary: ${e.stmSummary}\nQ-value: ${e.qValue.toFixed(3)}`,
			)
			.join("\n\n")

		const prompt = `You are a memory distillation system. Analyze these agent episodes and extract generalizable rules.

${episodeSummaries}

Extract up to 5 distinct RuleCards from these episodes. Each RuleCard should capture a reusable strategy or lesson.
Respond with a JSON array only (no markdown), each element having:
- topic: string (short topic label, ≤ 10 words)
- rule: string (the actionable rule, 1-2 sentences)
- examples: string[] (1-2 short examples from the episodes)
- confidence: number (0.0-1.0, based on how consistently the rule appears)

Focus on high-Q episodes (Q ≥ 0.6) as positive examples.`

		try {
			let fullText = ""
			const stream = this.api.createMessage(
				"You are a precise JSON generator. Output only valid JSON arrays.",
				[{ role: "user", content: [{ type: "text", text: prompt }] }],
			)

			for await (const chunk of stream) {
				if (chunk.type === "text") {
					fullText += chunk.text
				}
			}

			// Parse LLM response
			const jsonMatch = fullText.match(/\[[\s\S]*\]/)
			if (!jsonMatch) return

			const extracted = JSON.parse(jsonMatch[0]) as Array<{
				topic: string
				rule: string
				examples: string[]
				confidence: number
			}>

			for (const raw of extracted) {
				if (!raw.topic || !raw.rule) continue
				await this.upsertRule(raw)
			}

			// Prune if over limit
			if (this.store.rules.length > LTM_MAX_RULES) {
				// Keep highest-confidence rules
				this.store.rules.sort((a, b) => b.confidence - a.confidence)
				this.store.rules = this.store.rules.slice(0, LTM_MAX_RULES)
			}

			await this.persist()
		} catch {
			// Silently swallow — distillation is best-effort
		}
	}

	private async upsertRule(raw: {
		topic: string
		rule: string
		examples: string[]
		confidence: number
	}): Promise<void> {
		const text = `${raw.topic}: ${raw.rule}`
		const embedding = await this.embedder.embed(text)

		// Check for near-duplicate (sim ≥ 0.85)
		for (const existing of this.store.rules) {
			const sim = MemoryEmbeddingAdapter.cosineSim(embedding, existing.embedding)
			if (sim >= 0.85) {
				// Merge: take higher confidence, union examples
				existing.confidence = Math.max(existing.confidence, raw.confidence)
				const newExamples = raw.examples.filter((ex) => !existing.examples.includes(ex))
				existing.examples = [...existing.examples, ...newExamples].slice(0, 4)
				existing.updatedAt = Date.now()
				return
			}
		}

		// New rule
		const now = Date.now()
		this.store.rules.push({
			id: `ltm_${now}_${Math.random().toString(36).slice(2, 8)}`,
			topic: raw.topic,
			rule: raw.rule,
			examples: (raw.examples ?? []).slice(0, 4),
			confidence: Math.max(0, Math.min(1, raw.confidence)),
			useCount: 0,
			createdAt: now,
			updatedAt: now,
			embedding,
		})
	}

	// ── Retrieval ──────────────────────────────────────────────────────────────

	/**
	 * Two-phase retrieval for LTM rules (same algorithm as episodic).
	 */
	async retrieve(queryIntent: string): Promise<RuleCard[]> {
		await this.load()
		if (this.store.rules.length === 0) return []

		const queryVec = await this.embedder.embed(queryIntent)
		if (queryVec.length === 0) return []

		// Phase A
		const withSim = this.store.rules
			.map((rule) => ({
				rule,
				sim: MemoryEmbeddingAdapter.cosineSim(queryVec, rule.embedding),
			}))
			.filter((x) => x.sim >= SIM_THRESHOLD)
			.sort((a, b) => b.sim - a.sim)
			.slice(0, TOP_K1)

		if (withSim.length === 0) return []

		// Phase B: use confidence as Q-proxy
		const sims = withSim.map((x) => x.sim)
		const qVals = withSim.map((x) => x.rule.confidence)

		const simHat = MemoryEmbeddingAdapter.zScore(sims)
		const qHat = MemoryEmbeddingAdapter.zScore(qVals)

		const scored = withSim.map((x, i) => ({
			rule: x.rule,
			score: (1 - LAMBDA) * (simHat[i] ?? 0) + LAMBDA * (qHat[i] ?? 0),
		}))

		scored.sort((a, b) => b.score - a.score)

		return scored.slice(0, TOP_K2).map((x) => x.rule)
	}

	/** Increment useCount for retrieved rules. */
	async markUsed(ruleIds: string[]): Promise<void> {
		await this.load()
		let changed = false
		for (const rule of this.store.rules) {
			if (ruleIds.includes(rule.id)) {
				rule.useCount++
				rule.updatedAt = Date.now()
				changed = true
			}
		}
		if (changed) {
			await this.persist()
		}
	}

	getRules(): readonly RuleCard[] {
		return this.store.rules
	}
}
