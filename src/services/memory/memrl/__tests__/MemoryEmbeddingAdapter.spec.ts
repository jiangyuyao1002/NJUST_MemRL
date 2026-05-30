import { describe, it, expect, vi } from "vitest"
import { MemoryEmbeddingAdapter } from "../MemoryEmbeddingAdapter"
import type { IEmbedder } from "../../../code-index/interfaces/embedder"

function makeEmbedder(embeddings: number[][]): IEmbedder {
	return {
		createEmbeddings: vi.fn().mockResolvedValue({ embeddings }),
		validateConfiguration: vi.fn().mockResolvedValue({ valid: true }),
		get embedderInfo() {
			return { name: "openai" as const }
		},
	}
}

describe("MemoryEmbeddingAdapter", () => {
	it("embeds a single text", async () => {
		const vec = [0.1, 0.2, 0.3]
		const adapter = new MemoryEmbeddingAdapter(makeEmbedder([vec]))
		const result = await adapter.embed("hello")
		expect(result).toEqual(vec)
	})

	it("returns [] on embedder error", async () => {
		const brokenEmbedder: IEmbedder = {
			createEmbeddings: vi.fn().mockRejectedValue(new Error("fail")),
			validateConfiguration: vi.fn().mockResolvedValue({ valid: false }),
			get embedderInfo() {
				return { name: "openai" as const }
			},
		}
		const adapter = new MemoryEmbeddingAdapter(brokenEmbedder)
		const result = await adapter.embed("test")
		expect(result).toEqual([])
	})

	describe("cosineSim", () => {
		it("returns 1 for identical vectors", () => {
			const v = [1, 0, 0]
			expect(MemoryEmbeddingAdapter.cosineSim(v, v)).toBeCloseTo(1)
		})

		it("returns 0 for orthogonal vectors", () => {
			expect(MemoryEmbeddingAdapter.cosineSim([1, 0], [0, 1])).toBeCloseTo(0)
		})

		it("returns 0 for empty vectors", () => {
			expect(MemoryEmbeddingAdapter.cosineSim([], [])).toBe(0)
		})

		it("returns -1 for opposite vectors", () => {
			expect(MemoryEmbeddingAdapter.cosineSim([1, 0], [-1, 0])).toBeCloseTo(-1)
		})
	})

	describe("zScore", () => {
		it("returns [] for empty array", () => {
			expect(MemoryEmbeddingAdapter.zScore([])).toEqual([])
		})

		it("returns zeros when all values are equal", () => {
			expect(MemoryEmbeddingAdapter.zScore([5, 5, 5])).toEqual([0, 0, 0])
		})

		it("normalises to mean 0 and std ~1", () => {
			const vals = [1, 2, 3, 4, 5]
			const z = MemoryEmbeddingAdapter.zScore(vals)
			const mean = z.reduce((a, b) => a + b, 0) / z.length
			expect(mean).toBeCloseTo(0, 5)
		})
	})
})
