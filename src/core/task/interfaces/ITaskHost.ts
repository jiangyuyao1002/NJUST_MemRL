import type { TaskLike, TodoItem } from "@njust-ai/types"
import { NJUST_AIEventName } from "@njust-ai/types"

import type { IMcpHubClient } from "../../../services/mcp/interfaces/IMcpHubClient"
import type { IMcpHubService } from "../../../services/mcp/interfaces/IMcpHubService"
import type { SkillsManager } from "../../../services/skills/SkillsManager"
import type { MemoryManager } from "../../../services/memory/memrl/MemoryManager"
import type { ContextProxy } from "../../config/ContextProxy"

import type { ITaskUINotifier } from "./ITaskUINotifier"
import type { ITaskDiffViewProvider } from "./ITaskDiffViewProvider"

export type { TaskHostState } from "./taskHostState"

/**
 * Extension host surface required by Task (B.1): state, MCP, webview, profile events.
 * ClineProvider implements this; Task imports only this interface (not ClineProvider).
 */
export interface ITaskHost extends IMcpHubClient, ITaskUINotifier {
	readonly contextProxy: ContextProxy

	log(message: string): void

	getMcpHub(): IMcpHubService | undefined

	getSkillsManager(): SkillsManager | undefined

	/** Returns the MemRL MemoryManager instance (lazy-initialised). */
	getMemoryManager(cwd?: string): MemoryManager | undefined

	delegateParentAndOpenChild(params: {
		parentTaskId: string
		message: string
		initialTodos: TodoItem[]
		mode: string
		isolationLevel?: string
		forkedContextSummary?: string
		cacheSafeParams?: import("../SubTaskOptions").CacheSafeParams
	}): Promise<TaskLike>

	setMode(mode: string): Promise<void>

	setProviderProfile(providerProfile: string): Promise<void>

	handleModeSwitch(mode: string): Promise<void>

	cancelTask(): Promise<void>

	getTaskStackSize(): number

	convertToWebviewUri(filePath: string): string

	createDiffViewProvider?(cwd: string, task: unknown): ITaskDiffViewProvider

	on(
		event: NJUST_AIEventName.ProviderProfileChanged,
		listener: (config: { name: string; provider?: string }) => void | Promise<void>,
	): void

	off(
		event: NJUST_AIEventName.ProviderProfileChanged,
		listener: (config: { name: string; provider?: string }) => void | Promise<void>,
	): void

	compileLocal?(cwd: string): Promise<{ success: boolean; output: string }>
}
