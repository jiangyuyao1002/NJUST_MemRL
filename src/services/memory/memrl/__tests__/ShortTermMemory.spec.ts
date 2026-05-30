import { describe, it, expect } from "vitest"
import { ShortTermMemory } from "../ShortTermMemory"

describe("ShortTermMemory", () => {
	it("stores entries and returns them", () => {
		const stm = new ShortTermMemory(10_000)
		stm.push("user", "Hello")
		stm.push("assistant", "World")
		const entries = stm.getEntries()
		expect(entries).toHaveLength(2)
		expect(entries[0].role).toBe("user")
		expect(entries[1].role).toBe("assistant")
	})

	it("trims oldest entries when over maxChars", () => {
		const stm = new ShortTermMemory(10)
		stm.push("user", "12345") // 5 chars
		stm.push("user", "67890") // 5 chars — total = 10
		stm.push("user", "XXXXX") // 5 chars — total exceeds 10, trim head
		const entries = stm.getEntries()
		// First entry should have been evicted
		expect(entries[0].content).not.toBe("12345")
	})

	it("summarize returns formatted text", () => {
		const stm = new ShortTermMemory()
		stm.push("user", "do the thing")
		stm.push("assistant", "done")
		const summary = stm.summarize()
		expect(summary).toContain("user: do the thing")
		expect(summary).toContain("assistant: done")
	})

	it("clear resets all state", () => {
		const stm = new ShortTermMemory()
		stm.push("user", "test")
		stm.clear()
		expect(stm.getEntries()).toHaveLength(0)
		expect(stm.charCount).toBe(0)
	})
})
