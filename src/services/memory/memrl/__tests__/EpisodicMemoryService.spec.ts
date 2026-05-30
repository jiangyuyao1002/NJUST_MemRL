import { describe, it, expect, vi, beforeEach } from "vitest"
import * as fs from "fs/promises"
import * as path from "path"
import os from "os"
import { EpisodicMemoryService } from "../EpisodicMemoryService"
import { MemoryEmbeddingAdapter } from "../MemoryEmbeddingAdapter"
import type { IEmbedder } from "../../../code-index/interfaces/embedder"
import { ALPHA, Q_INIT, SIM_THRESHOLD } from "../constants"

// Mock safeWriteJson to avoid real file I/O in most tests
vi.mock("../../../../utils/safeWriteJson", () => ({
	safeWriteJson: vi.fn().mockResolvedValue(undefined),
}))

function makeVec(seed: number): number[] {
	// Deterministic "embedding": unit vector in direction [seed, 1-seed]
	const x = seed
	const y = Math.sqrt(1 - x * x)
	return [x, y]
}

function makeEmbedder(vecFn: (text: string) => number[]): MemoryEmbeddingAdapter {
	const embedder: IEmbedder = {
		createEmbeddings: vi.fn().mockImplementation(async (texts: string[]) => ({
			embeddings: texts.map(vecFn),
		})),
		validateConfiguration: vi.fn().mockResolvedValue({ valid: true }),
		get embedderInfo() {
			return { name: "openai" as const }
		},
	}
	return new MemoryEmbeddingAdapter(embedder)
}

describe("EpisodicMemoryService", () => {
	let tmpDir: string
	let service: EpisodicMemoryService
	let distillCalled: boolean

	beforeEach(async () => {
		tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "memrl-test-"))
		distillCalled = false
		// Use a fixed vec for all texts so retrieval is deterministic
		const adapter = makeEmbedder(() => makeVec(0.6))
		service = new EpisodicMemoryService(tmpDir, adapter, () => {
			distillCalled = true
		})
	})

	it("starts with empty store", async () => {
		const results = await service.retrieve("anything")
		expect(results).toHaveLength(0)
	})

	it("write stores an entry and increments totalWrites", async () => {
		await service.write("fix bug", "user: fix\nassistant: fixed", 1.0)
		expect(service.totalWrites).toBe(1)
	})

	it("Q-value update follows Monte Carlo rule", async () => {
		await service.write("fix bug", "summary", 1.0)
		const expectedQ = Q_INIT + ALPHA * (1.0 - Q_INIT)
		// retrieve to get the entry back
		const entries = await service.retrieve("fix bug")
		expect(entries[0].qValue).toBeCloseTo(expectedQ, 5)
	})

	it("retrieve returns relevant entry above threshold", async () => {
		await service.write("fix bug in auth", "summary", 0.8)
		const results = await service.retrieve("fix auth bug")
		// Cosine sim of identical vecs = 1.0 > SIM_THRESHOLD
		expect(results.length).toBeGreaterThan(0)
	})

	it("retrieve returns empty when sim < threshold", async () => {
		// Use orthogonal vectors
		const adapterHigh = makeEmbedder((text) => (text === "query" ? [1, 0] : [0, 1]))
		const svc = new EpisodicMemoryService(tmpDir + "2", adapterHigh)
		await svc.write("intent", "summary", 0.5)
		const results = await svc.retrieve("query")
		expect(results).toHaveLength(0)
	})

	it("triggers distillation callback every LTM_DISTILL_INTERVAL writes", async () => {
		// LTM_DISTILL_INTERVAL = 10
		for (let i = 0; i < 9; i++) {
			await service.write(`intent ${i}`, "s", 0.5)
		}
		expect(distillCalled).toBe(false)
		await service.write("intent 10", "s", 0.5)
		expect(distillCalled).toBe(true)
	})

	it("getRecent returns most recent n entries", async () => {
		for (let i = 0; i < 5; i++) {
			await service.write(`intent ${i}`, "s", 0.5)
		}
		const recent = service.getRecent(3)
		expect(recent).toHaveLength(3)
	})
})
