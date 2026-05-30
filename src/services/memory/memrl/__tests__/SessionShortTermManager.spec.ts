import { describe, it, expect } from "vitest"
import { SessionShortTermManager } from "../SessionShortTermManager"

describe("SessionShortTermManager", () => {
	it("creates a new STM for an unknown taskId", () => {
		const mgr = new SessionShortTermManager()
		const stm = mgr.get("task-1")
		expect(stm).toBeDefined()
		expect(mgr.size).toBe(1)
	})

	it("returns the same STM for the same taskId", () => {
		const mgr = new SessionShortTermManager()
		const a = mgr.get("task-1")
		const b = mgr.get("task-1")
		expect(a).toBe(b)
	})

	it("evicts LRU entry when at capacity", () => {
		const mgr = new SessionShortTermManager(3)
		mgr.get("a")
		mgr.get("b")
		mgr.get("c")
		// Access "a" so it becomes MRU
		mgr.get("a")
		// Adding "d" should evict "b" (LRU)
		mgr.get("d")
		expect(mgr.size).toBe(3)
		// "b" was LRU; a fresh STM is created for it if re-accessed
		const bStm = mgr.get("b")
		expect(bStm.getEntries()).toHaveLength(0)
	})

	it("delete removes the entry", () => {
		const mgr = new SessionShortTermManager()
		mgr.get("task-1")
		mgr.delete("task-1")
		expect(mgr.size).toBe(0)
	})
})
