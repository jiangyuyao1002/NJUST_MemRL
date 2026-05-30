import os from "os"
import * as path from "path"
import fs from "fs/promises"
import EventEmitter from "events"

import axios from "axios"
import pWaitFor from "p-wait-for"
import * as vscode from "vscode"

import type { McpServer } from "@njust-ai-cj/types"
import {
	type TaskProviderLike,
	type TaskProviderEvents,
	type ProviderSettings,
	type NJUST_AI_CJSettings,
	type ProviderSettingsEntry,
	type TelemetryPropertiesProvider,
	type TelemetryProperties,
	type CodeActionId,
	type CodeActionName,
	type TerminalActionId,
	type TerminalActionPromptType,
	type HistoryItem,
	type CreateTaskOptions,
	type ExtensionMessage,
	type ExtensionState,
	type GlobalState,
	NJUST_AI_CJEventName,
	DEFAULT_MODES,
	TelemetryEventName,
} from "@njust-ai-cj/types"
import {} from "@njust-ai-cj/core/providers"
import { TelemetryService } from "@njust-ai-cj/telemetry"
import { Package } from "../../shared/package"

import { Mode } from "../../shared/modes"
import { EMBEDDING_MODEL_PROFILES } from "../../shared/embeddingModels"
import { ProfileValidator } from "../../shared/ProfileValidator"

import { Terminal } from "../../integrations/terminal/Terminal"
import { DiffViewProvider } from "../../integrations/editor/DiffViewProvider"

import { getTheme } from "../../integrations/theme/getTheme"
import WorkspaceTracker from "../../integrations/workspace/WorkspaceTracker"

import { McpServerManager } from "../../services/mcp/McpServerManager"
import type { IMcpHubService } from "../../services/mcp/interfaces/IMcpHubService"

import { CodeIndexManager } from "../../services/code-index/manager"

import type { IndexProgressUpdate } from "../../services/code-index/interfaces/manager"
import { SkillsManager } from "../../services/skills/SkillsManager"
import { MemoryManager } from "../../services/memory/memrl/MemoryManager"

import { setTtsEnabled, setTtsSpeed } from "../../utils/tts"
import { getWorkspacePath } from "../../utils/path"
import { OrganizationAllowListViolationError } from "../../utils/errors"

import { setPanel } from "../../activate/registerCommands"

import { t } from "../../i18n"

import { forceFullModelDetailsLoad, hasLoadedFullDetails } from "../../api/providers/fetchers/lmstudio-full-details"

import { ContextProxy } from "../config/ContextProxy"
import { ProviderSettingsManager } from "../config/ProviderSettingsManager"
import { CustomModesManager } from "../config/CustomModesManager"
import { Task } from "../task/Task"
import type { ITaskHost } from "../task/interfaces/ITaskHost"
import type { ITaskDiffViewProvider } from "../task/interfaces/ITaskDiffViewProvider"
import type { IMcpHubClient } from "../../services/mcp/interfaces/IMcpHubClient"
import { PlanEngine } from "../agent/PlanEngine"
import { AgentOrchestrator } from "../agent/AgentOrchestrator"
import { taskEventBus, type DisposableLike } from "../events/TaskEventBus"
import { presentAssistantMessage } from "../assistant-message/presentAssistantMessage"
import {
	registerActionTarget,
	unregisterActionTarget,
	getVisibleInstance as _getVisibleInstance,
	getInstance as _getInstance,
	handleCodeAction,
	handleTerminalAction,
} from "../../activate/providerActionDispatcher"
import { registerUriCallbackHandler } from "../../activate/handleUri"

import { WebviewMessageRouter } from "./WebviewMessageRouter"
import { PendingEditManager } from "./PendingEditManager"
import { WebviewContentProvider } from "./WebviewContentProvider"
import { SettingsManager } from "./SettingsManager"
import { TaskCoordinator } from "./TaskCoordinator"
import { WebviewRouter } from "./WebviewRouter"
import { getMergedCommandLists } from "./ClineProviderState"
import {
	buildWebviewState,
	getState,
	getTelemetryProperties,
	type ClineProviderState,
} from "./WebviewStateBuilder"
import {
	activateProviderProfileWithProvider,
	deleteProviderProfileWithProvider,
	getProviderProfileEntriesWithProvider,
	getProviderProfileEntryWithProvider,
	handleModeSwitchWithProvider,
	hasProviderProfileEntryWithProvider,
	restoreHistoryModeAndProfileWithProvider,
	upsertProviderProfileWithProvider,
} from "./ClineProviderModeSync"
import {
	delegateParentAndOpenChildWithProvider,
	reopenParentFromDelegationWithProvider,
} from "./ClineProviderDelegation"
import { TaskStackManager } from "./TaskStackManager"
import { TaskHistoryService, type TaskHistoryHost } from "./TaskHistoryService"
import type { TodoItem } from "@njust-ai-cj/types"
import { requestyDefaultModelId, openRouterDefaultModelId } from "@njust-ai-cj/core/providers"
import { TaskHistoryStore } from "../task-persistence"
import { REQUESTY_BASE_URL } from "../../shared/utils/requesty"
import { logger } from "../../shared/logger"
import { getErrorMessage } from "../../shared/error-utils"

/**
 * https://github.com/microsoft/vscode-webview-ui-toolkit-samples/blob/main/default/weather-webview/src/providers/WeatherViewProvider.ts
 * https://github.com/KumarVariable/vscode-extension-sidebar-html/blob/master/src/customSidebarViewProvider.ts
 */

export type ClineProviderEvents = {
	clineCreated: [cline: Task]
}

export class ClineProvider
	extends EventEmitter<TaskProviderEvents>
	implements vscode.WebviewViewProvider, TelemetryPropertiesProvider, TaskProviderLike, ITaskHost, IMcpHubClient
{
	// Used in package.json as the view's id. This value cannot be changed due
	// to how VSCode caches views based on their id, and updating the id would
	// break existing instances of the extension.
	public static readonly sideBarId = `${Package.name}.SidebarProvider`
	public static readonly tabPanelId = `${Package.name}.TabPanelProvider`
	private static activeInstances: Set<ClineProvider> = new Set()
	private disposables: vscode.Disposable[] = []
	private webviewDisposables: vscode.Disposable[] = []
	private readonly messageRouter: WebviewMessageRouter
	public readonly settingsManager: SettingsManager
	public readonly taskCoordinator: TaskCoordinator
	public readonly webviewRouter: WebviewRouter
	public view?: vscode.WebviewView | vscode.WebviewPanel
	public readonly stack: TaskStackManager
	private codeIndexStatusSubscription?: vscode.Disposable
	private codeIndexManager?: CodeIndexManager
	private _workspaceTracker?: WorkspaceTracker // workSpaceTracker read-only for access outside this class
	public mcpHub?: IMcpHubService // Must be public to satisfy IWebviewStateHost
	protected skillsManager?: SkillsManager
	private _memoryManager?: MemoryManager
	private taskCreationCallback: (task: Task) => void
	private readonly assistantPresentationSubscription: DisposableLike
	private currentWorkspacePath: string | undefined
	private _disposed = false

	public readonly taskHistoryStore: TaskHistoryStore
	public readonly taskHistory: TaskHistoryService
	private readonly pendingEditManager: PendingEditManager
	private readonly webviewContentProvider: WebviewContentProvider

	/**
	 * Monotonically increasing sequence number for clineMessages state pushes.
	 * Used by the frontend to reject stale state that arrives out-of-order.
	 */
	private clineMessagesSeq = 0

	public isViewLaunched = false
	public settingsImportedAt?: number
	public readonly latestAnnouncementId = "mar-2026-v3.51.0-gpt-54-slash-skills" // v3.51.0 OpenAI GPT-5.4 support and slash command skills
	public readonly providerSettingsManager: ProviderSettingsManager
	public readonly customModesManager: CustomModesManager
	public readonly planEngine: PlanEngine
	public readonly agentOrchestrator: AgentOrchestrator

	/**
	 * Optional compile function injected by extension.ts when CangjieCompileGuard
	 * is available. Used by CloudAgentOrchestrator to run local cjpm builds.
	 */
	compileLocal?: (cwd: string) => Promise<{ success: boolean; output: string }>

	get cloudAuthSkipModel(): boolean {
		return this.context.globalState.get<boolean>("roo-auth-skip-model") ?? false
	}

	get lockApiConfigAcrossModes(): boolean {
		return this.context.workspaceState.get("lockApiConfigAcrossModes", false)
	}

	get extensionVersion(): string {
		return this.context.extension?.packageJSON?.version ?? ""
	}

	get extensionPackageJSON(): { name?: string; version?: string } | undefined {
		return this.context.extension?.packageJSON
	}

	constructor(
		readonly context: vscode.ExtensionContext,
		private readonly outputChannel: vscode.OutputChannel,
		public readonly renderContext: "sidebar" | "editor" = "sidebar",
		public readonly contextProxy: ContextProxy,
	) {
		super()
		this.currentWorkspacePath = getWorkspacePath()
		this.messageRouter = new WebviewMessageRouter(this)
		this.pendingEditManager = new PendingEditManager({ log: (msg) => this.log(msg) })
		this.webviewContentProvider = new WebviewContentProvider({
			extensionUri: this.contextProxy.extensionUri,
			getValues: () => this.contextProxy.getValues(),
		})
		this.settingsManager = new SettingsManager(this.contextProxy)
		this.webviewRouter = new WebviewRouter({
			isDisposed: () => this._disposed,
			getWebview: () => this.view?.webview,
			buildState: () => this.getStateToPostToWebview(),
		})

		this.stack = new TaskStackManager({
			outputChannel: this.outputChannel,
			emit: (event, ...args) => EventEmitter.prototype.emit.call(this, event, ...args) as boolean,
			getState: () => this.getState(),
			getTaskWithId: (id) => this.getTaskWithId(id),
			updateTaskHistory: (item, options) => this.updateTaskHistory(item, options),
			createTaskWithHistoryItem: (historyItem, options) => this.createTaskWithHistoryItem(historyItem, options),
			performPreparationTasks: (task) => this.performPreparationTasks(task),
		})

		ClineProvider.activeInstances.add(this)
		registerActionTarget(this)
		registerUriCallbackHandler(this)

		void this.settingsManager.setGlobalValue("codebaseIndexModels", EMBEDDING_MODEL_PROFILES)

		// Initialize the per-task file-based history store.
		// The globalState write-through is debounced separately (not on every mutation)
		// since per-task files are authoritative and globalState is only for downgrade compat.
		this.taskHistoryStore = new TaskHistoryStore(this.contextProxy.globalStorageUri.fsPath, {
			// eslint-disable-next-line @typescript-eslint/require-await
			onWrite: async () => {
				this.taskHistory.scheduleGlobalStateWriteThrough()
			},
		})
		// Use Object.defineProperties with arrow function getters to preserve lexical `this`
		// (object-literal getters would bind `this` to the config object, not ClineProvider).
		this.taskHistory = new TaskHistoryService(
			Object.defineProperties(
				{
					context: this.context,
					contextProxy: this.contextProxy as unknown as TaskHistoryHost["contextProxy"],
					taskHistoryStore: this.taskHistoryStore,
					outputChannel: this.outputChannel,
					stack: this.stack,
					postMessageToWebview: (msg: ExtensionMessage) => this.postMessageToWebview(msg),
				} as TaskHistoryHost,
				{
					cwd: { get: () => this.cwd, enumerable: true, configurable: true },
					isViewLaunched: { get: () => this.isViewLaunched, enumerable: true, configurable: true },
				},
			),
		)
		this.taskHistory.initialize().catch((error) => {
			this.log(`Failed to initialize TaskHistoryStore: ${error}`)
		})
		this.taskCoordinator = new TaskCoordinator({
			getCurrentTask: () => this.stack.current,
			getTaskStackSize: () => this.stack.size,
			getCurrentTaskStack: () => this.stack.taskIds,
			getRecentTasks: () => this.taskHistory.getRecentTasks(),
			createTask: (text, images, parentTask, options, configuration) =>
				this.createTaskInternal(text, images, parentTask, options, configuration),
			cancelTask: () => this.cancelTaskInternal(),
			clearTask: () => this.clearTaskInternal(),
			resumeTask: (taskId) => this.resumeTaskInternal(taskId),
			createTaskWithHistoryItem: (historyItem, options) => this.createTaskWithHistoryItem(historyItem, options),
		})

		// Start configuration loading (which might trigger indexing) in the background.
		// Don't await, allowing activation to continue immediately.

		this._workspaceTracker = new WorkspaceTracker(this)
		this._workspaceTracker.init?.()

		this.providerSettingsManager = new ProviderSettingsManager(this.context)
		this.providerSettingsManager.initialize().catch((error) => {
			logger.error("ClineProvider", "Failed to initialize ProviderSettingsManager:", error)
		})

		this.customModesManager = new CustomModesManager(this.context, async () => {
			await this.postStateToWebviewWithoutClineMessages()
		})

		// Initialize MCP Hub through the singleton manager
		McpServerManager.getInstance(this.context, this)
			.then((hub) => {
				this.mcpHub = hub
				if (this.mcpHub) {
					this.mcpHub.registerClient()
				}
			})
			.catch((error) => {
				this.log(`Failed to initialize MCP Hub: ${error}`)
			})

		this.planEngine = new PlanEngine(this, this.outputChannel)
		this.agentOrchestrator = new AgentOrchestrator(this, this.outputChannel)
		this.assistantPresentationSubscription = taskEventBus.on(
			"task:assistant-message-requested",
			async (_event, payload) => {
				const task = (payload.data as { task?: Task } | undefined)?.task
				if (!task || task.providerRef.deref() !== this) {
					return
				}
				await presentAssistantMessage(task)
			},
		)

		// Initialize Skills Manager for skill discovery
		this.skillsManager = new SkillsManager(this)
		this.skillsManager.initialize().catch((error) => {
			this.log(`Failed to initialize Skills Manager: ${error}`)
		})

		// Forward <most> task events to the provider.
		// We do something fairly similar for the IPC-based API.
		this.taskCreationCallback = (instance: Task) => {
			this.emit(NJUST_AI_CJEventName.TaskCreated, instance)
			this.stack.bindEventForwarders(instance)
		}
	}

	/**
	 * Override EventEmitter's on method to match TaskProviderLike interface
	 */
	override on<K extends keyof TaskProviderEvents>(
		event: K,
		listener: (...args: TaskProviderEvents[K]) => void | Promise<void>,
	): this {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return super.on(event, listener as any)
	}

	/**
	 * Override EventEmitter's off method to match TaskProviderLike interface
	 */
	override off<K extends keyof TaskProviderEvents>(
		event: K,
		listener: (...args: TaskProviderEvents[K]) => void | Promise<void>,
	): this {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return super.off(event, listener as any)
	}

	public async initializeCloudProfileSyncWhenReady(): Promise<void> {}

	async performPreparationTasks(cline: Task) {
		// LMStudio: We need to force model loading in order to read its context
		// size; we do it now since we're starting a task with that model selected.
		if (cline.apiConfiguration && cline.apiConfiguration.apiProvider === "lmstudio") {
			try {
				if (!hasLoadedFullDetails(cline.apiConfiguration.lmStudioModelId!)) {
					await forceFullModelDetailsLoad(
						cline.apiConfiguration.lmStudioBaseUrl ?? "http://localhost:1234",
						cline.apiConfiguration.lmStudioModelId!,
					)
				}
			} catch (error) {
				const msg = getErrorMessage(error)
				this.log(`Failed to load full model details for LM Studio: ${msg}`)
				TelemetryService.reportError(error, TelemetryEventName.WEBVIEW_ERROR)
				vscode.window.showErrorMessage(msg)
			}
		}
	}

	createDiffViewProvider(cwd: string, task: unknown): ITaskDiffViewProvider {
		return new DiffViewProvider(cwd, task as Task)
	}

	// Pending Edit Operations Management

	/**
	 * Sets a pending edit operation with automatic timeout cleanup
	 */
	public setPendingEditOperation(
		operationId: string,
		editData: {
			messageTs: number
			editedContent: string
			images?: string[]
			messageIndex: number
			apiConversationHistoryIndex: number
		},
	): void {
		this.pendingEditManager.set(operationId, editData)
	}

	/*
	VSCode extensions use the disposable pattern to clean up resources when the sidebar/editor tab is closed by the user or system. This applies to event listening, commands, interacting with the UI, etc.
	- https://vscode-docs.readthedocs.io/en/stable/extensions/patterns-and-principles/
	- https://github.com/microsoft/vscode-extension-samples/blob/main/webview-sample/src/extension.ts
	*/
	private clearWebviewResources() {
		while (this.webviewDisposables.length) {
			const x = this.webviewDisposables.pop()
			if (x) {
				x.dispose()
			}
		}
	}

	async dispose() {
		if (this._disposed) {
			return
		}

		this._disposed = true
		this.log("Disposing ClineProvider...")

		// Clear all tasks from the stack.
		while (this.stack.size > 0) {
			await this.stack.pop()
		}

		this.log("Cleared all tasks")

		// Clear all pending edit operations to prevent memory leaks
		this.pendingEditManager.clearAll()
		this.log("Cleared pending operations")

		if (this.view && "dispose" in this.view) {
			this.view.dispose()
			this.log("Disposed webview")
		}

		this.messageRouter.dispose()
		this.assistantPresentationSubscription.dispose()
		this.clearWebviewResources()

		while (this.disposables.length) {
			const x = this.disposables.pop()

			if (x) {
				x.dispose()
			}
		}

		this._workspaceTracker?.dispose()
		this._workspaceTracker = undefined
		await this.mcpHub?.unregisterClient()
		this.mcpHub = undefined
		await this.skillsManager?.dispose()
		this.skillsManager = undefined
		this.customModesManager?.dispose()
		this.taskHistoryStore.dispose()
		this.taskHistory.flushGlobalStateWriteThrough()
		this.log("Disposed all disposables")
		ClineProvider.activeInstances.delete(this)
		unregisterActionTarget(this)

		// Clean up any event listeners attached to this provider
		this.removeAllListeners()

		McpServerManager.unregisterProvider(this)
	}

	public static getVisibleInstance(): ClineProvider | undefined {
		const instance = _getVisibleInstance()
		return instance as ClineProvider | undefined
	}

	public static async getInstance(): Promise<ClineProvider | undefined> {
		const instance = await _getInstance()
		return instance as ClineProvider | undefined
	}

	public static async isActiveTask(): Promise<boolean> {
		const visibleProvider = await ClineProvider.getInstance()

		if (!visibleProvider) {
			return false
		}

		if (visibleProvider.getCurrentTask()) {
			return true
		}

		return false
	}

	public static async handleCodeAction(
		command: CodeActionId,
		promptType: CodeActionName,
		params: Record<string, string | unknown[]>,
	): Promise<void> {
		await handleCodeAction(command, promptType, params)
	}

	public static async handleTerminalAction(
		command: TerminalActionId,
		promptType: TerminalActionPromptType,
		params: Record<string, string | unknown[]>,
	): Promise<void> {
		await handleTerminalAction(command, promptType, params)
	}

	async resolveWebviewView(webviewView: vscode.WebviewView | vscode.WebviewPanel) {
		this.view = webviewView
		const inTabMode = this.configureWebviewPanelMode(webviewView)

		await this.initializeWebviewRuntimeState()
		await this.configureWebviewContent(webviewView)
		this.messageRouter.setWebviewMessageListener(webviewView.webview)
		this.webviewDisposables.push(...this.messageRouter.getDisposables())
		this.updateCodeIndexStatusSubscription()
		this.attachWebviewLifecycleListeners(webviewView, inTabMode)

		// If the extension is starting a new session, clear previous task state.
		// But don't clear if there's already an active task (e.g., resumed via IPC/bridge).
		const currentTask = this.getCurrentTask()
		if (!currentTask || currentTask.abandoned || currentTask.abort) {
			await this.stack.pop()
		}
	}

	private configureWebviewPanelMode(webviewView: vscode.WebviewView | vscode.WebviewPanel): boolean {
		const inTabMode = "onDidChangeViewState" in webviewView
		if (inTabMode) {
			setPanel(webviewView, "tab")
		} else if ("onDidChangeVisibility" in webviewView) {
			setPanel(webviewView, "sidebar")
		}
		return inTabMode
	}

	private async initializeWebviewRuntimeState(): Promise<void> {
		const {
			terminalShellIntegrationTimeout = Terminal.defaultShellIntegrationTimeout,
			terminalShellIntegrationDisabled = false,
			terminalCommandDelay = 0,
			terminalZshClearEolMark = true,
			terminalZshOhMy = false,
			terminalZshP10k = false,
			terminalPowershellCounter = false,
			terminalZdotdir = false,
			ttsEnabled,
			ttsSpeed,
		} = await this.getState()

		Terminal.setShellIntegrationTimeout(terminalShellIntegrationTimeout)
		Terminal.setShellIntegrationDisabled(terminalShellIntegrationDisabled)
		Terminal.setCommandDelay(terminalCommandDelay)
		Terminal.setTerminalZshClearEolMark(terminalZshClearEolMark)
		Terminal.setTerminalZshOhMy(terminalZshOhMy)
		Terminal.setTerminalZshP10k(terminalZshP10k)
		Terminal.setPowershellCounter(terminalPowershellCounter)
		Terminal.setTerminalZdotdir(terminalZdotdir)
		setTtsEnabled(ttsEnabled ?? false)
		setTtsSpeed(ttsSpeed ?? 1)

		await this.contextProxy.setValue("enableWebSearch", false)
	}

	private async configureWebviewContent(webviewView: vscode.WebviewView | vscode.WebviewPanel): Promise<void> {
		const resourceRoots = [this.contextProxy.extensionUri]
		if (vscode.workspace.workspaceFolders) {
			resourceRoots.push(...vscode.workspace.workspaceFolders.map((folder) => folder.uri))
		}

		webviewView.webview.options = { enableScripts: true, localResourceRoots: resourceRoots }
		webviewView.webview.html =
			this.contextProxy.extensionMode === vscode.ExtensionMode.Development
				? await this.webviewContentProvider.getHMRHtmlContent(webviewView.webview)
				: await this.webviewContentProvider.getHtmlContent(webviewView.webview)
	}

	private attachWebviewLifecycleListeners(
		webviewView: vscode.WebviewView | vscode.WebviewPanel,
		inTabMode: boolean,
	): void {
		const activeEditorSubscription = vscode.window.onDidChangeActiveTextEditor(() => {
			this.updateCodeIndexStatusSubscription()
		})
		this.webviewDisposables.push(activeEditorSubscription)

		if ("onDidChangeViewState" in webviewView) {
			const viewStateDisposable = webviewView.onDidChangeViewState(() => {
				if (this.view?.visible) {
					void this.postMessageToWebview({ type: "action", action: "didBecomeVisible" })
				}
			})
			this.webviewDisposables.push(viewStateDisposable)
		} else if ("onDidChangeVisibility" in webviewView) {
			const visibilityDisposable = webviewView.onDidChangeVisibility(() => {
				if (this.view?.visible) {
					void this.postMessageToWebview({ type: "action", action: "didBecomeVisible" })
				}
			})
			this.webviewDisposables.push(visibilityDisposable)
		}

		webviewView.onDidDispose(
			async () => {
				if (inTabMode) {
					this.log("Disposing ClineProvider instance for tab view")
					await this.dispose()
				} else {
					this.log("Clearing webview resources for sidebar view")
					this.clearWebviewResources()
					this.codeIndexManager = undefined
				}
			},
			null,
			this.disposables,
		)

		const configDisposable = vscode.workspace.onDidChangeConfiguration(async (e) => {
			if (e?.affectsConfiguration("workbench.colorTheme")) {
				await this.postMessageToWebview({ type: "theme", text: JSON.stringify(await getTheme()) })
			}
		})
		this.webviewDisposables.push(configDisposable)
	}

	public async createTaskWithHistoryItem(
		historyItem: HistoryItem & { rootTask?: Task; parentTask?: Task },
		options?: { startTask?: boolean },
	) {
		const isCliRuntime = process.env.ROO_CLI_RUNTIME === "1"
		const skipProfileRestoreFromHistory = isCliRuntime
		const isRehydratingCurrentTask = this.getCurrentTask()?.taskId === historyItem.id

		if (!isRehydratingCurrentTask) {
			await this.stack.pop()
		}

		await this.restoreHistoryModeAndProfile(historyItem, skipProfileRestoreFromHistory)

		const task = await this.createTaskInstanceFromHistory(historyItem, options)

		if (isRehydratingCurrentTask) {
			await this.stack.rehydrate(task)
		} else {
			await this.stack.push(task)
			this.log(
				`[createTaskWithHistoryItem] ${task.parentTask ? "child" : "parent"} task ${task.taskId}.${task.instanceId} instantiated`,
			)
		}

		await this.applyPendingEditIfPresent(task)
		return task
	}

	private async restoreHistoryModeAndProfile(
		historyItem: HistoryItem & { rootTask?: Task; parentTask?: Task },
		skipProfileRestoreFromHistory: boolean,
	): Promise<void> {
		return restoreHistoryModeAndProfileWithProvider(this, historyItem, skipProfileRestoreFromHistory)
	}

	private async createTaskInstanceFromHistory(
		historyItem: HistoryItem & { rootTask?: Task; parentTask?: Task },
		options?: { startTask?: boolean },
	): Promise<Task> {
		const { apiConfiguration, enableCheckpoints, checkpointTimeout, experiments } = await this.getState()
		return new Task({
			host: this,
			apiConfiguration,
			enableCheckpoints,
			checkpointTimeout,
			consecutiveMistakeLimit: apiConfiguration.consecutiveMistakeLimit,
			historyItem,
			experiments,
			rootTask: historyItem.rootTask,
			parentTask: historyItem.parentTask,
			taskNumber: historyItem.number,
			workspacePath: historyItem.workspace,
			onCreated: this.taskCreationCallback,
			startTask: options?.startTask ?? true,
			initialStatus: historyItem.status,
		})
	}

	// eslint-disable-next-line @typescript-eslint/require-await
	private async applyPendingEditIfPresent(task: Task): Promise<void> {
		const operationId = `task-${task.taskId}`
		const pendingEdit = this.pendingEditManager.get(operationId)
		if (!pendingEdit) {
			return
		}

		this.pendingEditManager.clear(operationId)
		this.log(`[createTaskWithHistoryItem] Processing pending edit after checkpoint restoration`)
		setTimeout(async () => {
			try {
				const { messageIndex, apiConversationHistoryIndex } = (() => {
					const messageIndex = task.clineMessages.findIndex((msg) => msg.ts === pendingEdit.messageTs)
					const apiConversationHistoryIndex = task.apiConversationHistory.findIndex(
						(msg) => msg.ts === pendingEdit.messageTs,
					)
					return { messageIndex, apiConversationHistoryIndex }
				})()
				if (messageIndex !== -1) {
					await task.overwriteClineMessages(task.clineMessages.slice(0, messageIndex))
					if (apiConversationHistoryIndex !== -1) {
						await task.overwriteApiConversationHistory(
							task.apiConversationHistory.slice(0, apiConversationHistoryIndex),
						)
					}
					await task.handleWebviewAskResponse(
						"messageResponse",
						pendingEdit.editedContent,
						pendingEdit.images,
					)
				}
			} catch (error) {
				this.log(`[createTaskWithHistoryItem] Error processing pending edit: ${error}`)
			}
		}, 100)
	}

	public async postMessageToWebview(message: ExtensionMessage) {
		await this.webviewRouter.postMessage(message)
	}

	public getGlobalState<K extends keyof GlobalState>(key: K): GlobalState[K] | undefined {
		return this.contextProxy.getValue(key)
	}

	public updateGlobalState<K extends keyof GlobalState>(key: K, value: GlobalState[K]): Promise<void> {
		return this.contextProxy.setValue(key, value)
	}

	/**
	 * Handle switching to a new mode, including updating the associated API configuration
	 * @param newMode The mode to switch to
	 */
	public async handleModeSwitch(newMode: Mode) {
		return handleModeSwitchWithProvider(this, newMode)
	}

	// Provider Profile Management

	// Provider Profile Management

	/**
	 * Updates the current task's API handler.
	 * Rebuilds when:
	 * - provider or model changes, OR
	 * - explicitly forced (e.g., user-initiated profile switch/save to apply changed settings like headers/baseUrl/tier).
	 * Always synchronizes task.apiConfiguration with latest provider settings.
	 * @param providerSettings The new provider settings to apply
	 * @param options.forceRebuild Force rebuilding the API handler regardless of provider/model equality
	 */
	getProviderProfileEntries(): ProviderSettingsEntry[] {
		return getProviderProfileEntriesWithProvider(this)
	}

	getProviderProfileEntry(name: string): ProviderSettingsEntry | undefined {
		return getProviderProfileEntryWithProvider(this, name)
	}

	public hasProviderProfileEntry(name: string): boolean {
		return hasProviderProfileEntryWithProvider(this, name)
	}

	async upsertProviderProfile(
		name: string,
		providerSettings: ProviderSettings,
		activate: boolean = true,
	): Promise<string | undefined> {
		return upsertProviderProfileWithProvider(this, name, providerSettings, activate)
	}

	async deleteProviderProfile(profileToDelete: ProviderSettingsEntry): Promise<void> {
		return deleteProviderProfileWithProvider(this, profileToDelete)
	}

	async activateProviderProfile(
		args: { name: string } | { id: string },
		options?: { persistModeConfig?: boolean; persistTaskHistory?: boolean },
	): Promise<void> {
		return activateProviderProfileWithProvider(this, args, options)
	}

	async updateCustomInstructions(instructions?: string) {
		// User may be clearing the field.
		await this.settingsManager.setGlobalValue("customInstructions", instructions || undefined)
		await this.postStateToWebview()
	}

	// MCP

	async ensureMcpServersDirectoryExists(): Promise<string> {
		// Get platform-specific application data directory
		let mcpServersDir: string
		if (process.platform === "win32") {
			// Windows: %APPDATA%\NJUST_AI_CJ\MCP
			mcpServersDir = path.join(os.homedir(), "AppData", "Roaming", "NJUST_AI_CJ", "MCP")
		} else if (process.platform === "darwin") {
			// macOS: ~/Documents/Cline/MCP
			mcpServersDir = path.join(os.homedir(), "Documents", "Cline", "MCP")
		} else {
			// Linux: ~/.local/share/Cline/MCP
			mcpServersDir = path.join(os.homedir(), ".local", "share", "NJUST_AI_CJ", "MCP")
		}

		try {
			await fs.mkdir(mcpServersDir, { recursive: true })
		} catch (_error) {
			// Fallback to a relative path if directory creation fails
			return path.join(os.homedir(), ".roo-code", "mcp")
		}
		return mcpServersDir
	}

	async ensureSettingsDirectoryExists(): Promise<string> {
		const { getSettingsDirectoryPath } = await import("../../utils/storage")
		const globalStoragePath = this.contextProxy.globalStorageUri.fsPath
		return getSettingsDirectoryPath(globalStoragePath)
	}

	// OpenRouter

	async handleOpenRouterCallback(code: string) {
		const { apiConfiguration, currentApiConfigName = "default" } = await this.getState()

		let apiKey: string

		try {
			const baseUrl = apiConfiguration.openRouterBaseUrl || "https://openrouter.ai/api/v1"
			// Extract the base domain for the auth endpoint.
			const baseUrlDomain = baseUrl.match(/^(https?:\/\/[^/]+)/)?.[1] || "https://openrouter.ai"
			const response = await axios.post(`${baseUrlDomain}/api/v1/auth/keys`, { code })

			if (response.data?.key) {
				apiKey = response.data.key
			} else {
				throw new Error("Invalid response from OpenRouter API")
			}
		} catch (error) {
			this.log(
				`Error exchanging code for API key: ${JSON.stringify(error, Object.getOwnPropertyNames(error), 2)}`,
			)

			throw error
		}

		const newConfiguration: ProviderSettings = {
			...apiConfiguration,
			apiProvider: "openrouter",
			openRouterApiKey: apiKey,
			openRouterModelId: apiConfiguration?.openRouterModelId || openRouterDefaultModelId,
		}

		await this.upsertProviderProfile(currentApiConfigName, newConfiguration)
	}

	// Requesty

	async handleRequestyCallback(code: string, baseUrl: string | null) {
		const { apiConfiguration } = await this.getState()

		const newConfiguration: ProviderSettings = {
			...apiConfiguration,
			apiProvider: "requesty",
			requestyApiKey: code,
			requestyModelId: apiConfiguration?.requestyModelId || requestyDefaultModelId,
		}

		// set baseUrl as undefined if we don't provide one
		// or if it is the default requesty url
		if (!baseUrl || baseUrl === REQUESTY_BASE_URL) {
			newConfiguration.requestyBaseUrl = undefined
		} else {
			newConfiguration.requestyBaseUrl = baseUrl
		}

		const profileName = `Requesty (${new Date().toLocaleString()})`
		await this.upsertProviderProfile(profileName, newConfiguration)
	}

	// Task history

	async getTaskWithId(id: string) {
		return this.taskHistory.getTaskWithId(id)
	}

	async getTaskWithAggregatedCosts(taskId: string) {
		return this.taskHistory.getTaskWithAggregatedCosts(taskId)
	}

	async showTaskWithId(id: string): Promise<void> {
		await this.taskHistory.showTaskWithId(id, async (item) => {
			await this.createTaskWithHistoryItem(item)
		})
	}

	async exportTaskWithId(id: string) {
		await this.taskHistory.exportTaskWithId(id)
	}

	async condenseTaskContext(taskId: string) {
		await this.taskHistory.condenseTaskContext(taskId)
	}

	async deleteTaskWithId(id: string, cascadeSubtasks: boolean = true) {
		await this.taskHistory.deleteTaskWithId(id, cascadeSubtasks)
		await this.postStateToWebview()
	}

	async deleteTaskFromState(id: string) {
		await this.taskHistory.deleteTaskFromState(id)
		await this.postStateToWebview()
	}

	async refreshWorkspace() {
		this.currentWorkspacePath = getWorkspacePath()
		await this.postStateToWebview()
	}

	async postStateToWebview() {
		await this.webviewRouter.postState()
	}

	/**
	 * Like postStateToWebview but intentionally omits taskHistory.
	 */
	async postStateToWebviewWithoutTaskHistory(): Promise<void> {
		await this.webviewRouter.postStateWithoutTaskHistory()
	}

	/**
	 * Like postStateToWebview but intentionally omits both clineMessages and taskHistory.
	 */
	async postStateToWebviewWithoutClineMessages(): Promise<void> {
		await this.webviewRouter.postStateWithoutClineMessages()
	}

	async getStateToPostToWebview(): Promise<ExtensionState> {
		const providerState = await this.getState()
		const commandLists = getMergedCommandLists(providerState.allowedCommands, providerState.deniedCommands)
		return await buildWebviewState(this, providerState, commandLists)
	}

	/**
	 * Storage
	 * https://dev.to/kompotkot/how-to-use-secretstorage-in-your-vscode-extensions-2hco
	 * https://www.eliostruyf.com/devhack-code-extension-storage-options/
	 */

	async getState(): Promise<ClineProviderState> {
		return getState(this)
	}

	/**
	 * Updates a task in the task history and optionally broadcasts the updated history to the webview.
	 * Now delegates to TaskHistoryStore for per-task file persistence.
	 *
	 * @param item The history item to update or add
	 * @param options.broadcast Whether to broadcast the updated history to the webview (default: true)
	 * @returns The updated task history array
	 */
	async updateTaskHistory(item: HistoryItem, options: { broadcast?: boolean } = {}): Promise<HistoryItem[]> {
		return this.taskHistory.updateTaskHistory(item, options)
	}

	public async broadcastTaskHistoryUpdate(history?: HistoryItem[]): Promise<void> {
		return this.taskHistory.broadcastTaskHistoryUpdate(history)
	}

	public async setValue<K extends keyof NJUST_AI_CJSettings>(key: K, value: NJUST_AI_CJSettings[K]) {
		await this.settingsManager.setValue(key, value)
	}

	public getValue<K extends keyof NJUST_AI_CJSettings>(key: K) {
		return this.settingsManager.getValue(key)
	}

	public getValues() {
		return this.settingsManager.getValues()
	}

	public async setValues(values: NJUST_AI_CJSettings) {
		await this.settingsManager.setValues(values)
	}

	// dev

	async resetState() {
		const answer = await vscode.window.showInformationMessage(
			t("common:confirmation.reset_state"),
			{ modal: true },
			t("common:answers.yes"),
		)

		if (answer !== t("common:answers.yes")) {
			return
		}

		await this.contextProxy.resetAllState()
		await this.providerSettingsManager.resetAllConfigs()
		await this.customModesManager.resetCustomModes()
		await this.stack.pop()
		await this.postMessageToWebview({ type: "action", action: "resetLogin" })
		await this.postStateToWebview()
		await this.postMessageToWebview({ type: "action", action: "chatButtonClicked" })
	}

	// logging

	public log(message: string) {
		this.outputChannel.appendLine(message)
		logger.info("ClineProvider", message)
	}

	// getters

	public get workspaceTracker(): WorkspaceTracker | undefined {
		return this._workspaceTracker
	}

	get viewLaunched() {
		return this.isViewLaunched
	}

	get messages() {
		return this.getCurrentTask()?.clineMessages || []
	}

	public getMcpHub(): IMcpHubService | undefined {
		return this.mcpHub
	}

	async onMcpServersUpdated(mcpServers: McpServer[]): Promise<void> {
		await this.postMessageToWebview({ type: "mcpServers", mcpServers })
	}

	getExtensionPackageVersion(): string {
		return this.context.extension.packageJSON.version ?? "1.0.0"
	}

	public getSkillsManager(): SkillsManager | undefined {
		return this.skillsManager
	}

	/**
	 * Lazily initialise and return the MemRL MemoryManager.
	 * @param cwd - Explicit workspace path (preferred). Falls back to getWorkspacePath().
	 */
	public getMemoryManager(cwd?: string): MemoryManager | undefined {
		const resolvedCwd = cwd || getWorkspacePath()
		if (!resolvedCwd) return undefined
		// Re-create if the workspace path changed
		if (!this._memoryManager || (this._memoryManager as unknown as { workspaceDir: string }).workspaceDir !== resolvedCwd) {
			this._memoryManager = new MemoryManager(resolvedCwd)
		}
		return this._memoryManager
	}

	/**
	 * Gets the CodeIndexManager for the current active workspace
	 * @returns CodeIndexManager instance for the current workspace or the default one
	 */
	public getCurrentWorkspaceCodeIndexManager(): CodeIndexManager | undefined {
		return CodeIndexManager.getInstance(this.context)
	}

	/**
	 * Updates the code index status subscription to listen to the current workspace manager
	 */
	private updateCodeIndexStatusSubscription(): void {
		// Get the current workspace manager
		const currentManager = this.getCurrentWorkspaceCodeIndexManager()

		// If the manager hasn't changed, no need to update subscription
		if (currentManager === this.codeIndexManager) {
			return
		}

		// Dispose the old subscription if it exists
		if (this.codeIndexStatusSubscription) {
			this.codeIndexStatusSubscription.dispose()
			this.codeIndexStatusSubscription = undefined
		}

		// Update the current workspace manager reference
		this.codeIndexManager = currentManager

		// Subscribe to the new manager's progress updates if it exists
		if (currentManager) {
			this.codeIndexStatusSubscription = currentManager.onProgressUpdate((_update: IndexProgressUpdate) => {
				// Only send updates if this manager is still the current one
				if (currentManager === this.getCurrentWorkspaceCodeIndexManager()) {
					// Get the full status from the manager to ensure we have all fields correctly formatted
					const fullStatus = currentManager.getCurrentStatus()
					void this.postMessageToWebview({
						type: "indexingStatusUpdate",
						values: fullStatus,
					})
				}
			})

			if (this.view) {
				this.webviewDisposables.push(this.codeIndexStatusSubscription)
			}

			// Send initial status for the current workspace
			void this.postMessageToWebview({
				type: "indexingStatusUpdate",
				values: currentManager.getCurrentStatus(),
			})
		}
	}

	/**
	 * TaskProviderLike, TelemetryPropertiesProvider
	 */

	public getCurrentTask(): Task | undefined {
		return this.taskCoordinator?.getCurrentTask() ?? this.stack.current
	}

	getTaskStackSize(): number {
		return this.taskCoordinator?.getTaskStackSize() ?? this.stack.size
	}

	public getCurrentTaskStack(): string[] {
		return this.taskCoordinator?.getCurrentTaskStack() ?? this.stack.taskIds
	}

	public getRecentTasks(): string[] {
		return this.taskCoordinator?.getRecentTasks() ?? this.taskHistory.getRecentTasks()
	}

	// When initializing a new task, (not from history but from a tool command
	// new_task) there is no need to remove the previous task since the new
	// task is a subtask of the previous one, and when it finishes it is removed
	// from the stack and the caller is resumed in this way we can have a chain
	// of tasks, each one being a sub task of the previous one until the main
	// task is finished.
	public async createTask(
		text?: string,
		images?: string[],
		parentTask?: Task,
		options: CreateTaskOptions = {},
		configuration: NJUST_AI_CJSettings = {},
	): Promise<Task> {
		return this.taskCoordinator.createTask(text, images, parentTask, options, configuration)
	}

	private async createTaskInternal(
		text?: string,
		images?: string[],
		parentTask?: Task,
		options: CreateTaskOptions = {},
		configuration: NJUST_AI_CJSettings = {},
	): Promise<Task> {
		if (configuration) {
			await this.setValues(configuration)

			if (configuration.allowedCommands) {
				await vscode.workspace
					.getConfiguration(Package.name)
					.update("allowedCommands", configuration.allowedCommands, vscode.ConfigurationTarget.Global)
			}

			if (configuration.deniedCommands) {
				await vscode.workspace
					.getConfiguration(Package.name)
					.update("deniedCommands", configuration.deniedCommands, vscode.ConfigurationTarget.Global)
			}

			if (configuration.commandExecutionTimeout !== undefined) {
				await vscode.workspace
					.getConfiguration(Package.name)
					.update(
						"commandExecutionTimeout",
						configuration.commandExecutionTimeout,
						vscode.ConfigurationTarget.Global,
					)
			}

			if (configuration.currentApiConfigName) {
				await this.setProviderProfile(configuration.currentApiConfigName)
			}

			// Register custom modes so the CustomModesManager knows about them.
			// setValues writes to global state, but the manager overwrites that
			// when it merges .roomodes + global settings on refresh.  Persisting
			// via updateCustomMode ensures modes survive the merge cycle.
			if (configuration.customModes?.length) {
				for (const mode of configuration.customModes) {
					await this.customModesManager.updateCustomMode(mode.slug, mode)
				}
			}
		}

		const { apiConfiguration, organizationAllowList, enableCheckpoints, checkpointTimeout, experiments } =
			await this.getState()

		// Single-open-task invariant: always enforce for user-initiated top-level tasks
		if (!parentTask) {
			try {
				await this.stack.pop()
			} catch (error) {
				logger.warn("ClineProvider", "Stack pop failed", error)
				TelemetryService.reportError(error, TelemetryEventName.WEBVIEW_ERROR)
				// Non-fatal
			}
		}

		if (!ProfileValidator.isProfileAllowed(apiConfiguration, organizationAllowList)) {
			throw new OrganizationAllowListViolationError(t("common:errors.violated_organization_allowlist"))
		}

		const task = new Task({
			host: this,
			apiConfiguration,
			enableCheckpoints,
			checkpointTimeout,
			consecutiveMistakeLimit: apiConfiguration.consecutiveMistakeLimit,
			task: text,
			images,
			experiments,
			rootTask: this.stack.root,
			parentTask,
			taskNumber: this.stack.size + 1,
			onCreated: this.taskCreationCallback,
			initialTodos: options.initialTodos,
			// Ensure this task is present in stack before startTask() emits
			// its initial state update, so state.currentTaskId is available ASAP.
			startTask: false,
			...options,
		})

		await this.stack.push(task)
		task.start()

		this.log(
			`[createTask] ${task.parentTask ? "child" : "parent"} task ${task.taskId}.${task.instanceId} instantiated`,
		)

		return task
	}

	public async cancelTask(): Promise<void> {
		await (this.taskCoordinator?.cancelTask() ?? ClineProvider.prototype.cancelTaskInternal.call(this))
	}

	private async cancelTaskInternal(): Promise<void> {
		const task = this.getCurrentTask()

		if (!task) {
			return
		}

		logger.info("ClineProvider", `cancelTask: cancelling task ${task.taskId}.${task.instanceId}`)

		let historyItem: HistoryItem | undefined
		try {
			const history = await this.getTaskWithId(task.taskId)
			historyItem = history.historyItem
		} catch (error) {
			// During task startup there is a short window where currentTask exists
			// but task history has not been persisted yet. Cancelling should still
			// abort safely; we just skip post-cancel rehydration in that case.
			if (error instanceof Error && error.message === "Task not found") {
				this.log(`[cancelTask] task history missing for ${task.taskId}; skipping rehydrate`)
			} else {
				throw error
			}
		}

		// Preserve parent and root task information for history item.
		const rootTask = task.rootTask
		const parentTask = task.parentTask

		// Mark this as a user-initiated cancellation so provider-only rehydration can occur
		task.abortReason = "user_cancelled"

		// Capture the current instance to detect if rehydrate already occurred elsewhere
		const originalInstanceId = task.instanceId

		// Immediately cancel the underlying HTTP request if one is in progress
		// This ensures the stream fails quickly rather than waiting for network timeout
		task.cancelCurrentRequest()

		// Begin abort (non-blocking)
		void task.abortTask()

		// Immediately mark the original instance as abandoned to prevent any residual activity
		task.abandoned = true

		await pWaitFor(
			() =>
				this.getCurrentTask()! === undefined ||
				this.getCurrentTask()!.isStreaming === false ||
				this.getCurrentTask()!.didFinishAbortingStream ||
				// If only the first chunk is processed, then there's no
				// need to wait for graceful abort (closes edits, browser,
				// etc).
				this.getCurrentTask()!.isWaitingForFirstChunk,
			{
				timeout: 3_000,
			},
		).catch(() => {
			logger.error("ClineProvider", "cancelTask: Failed to abort task")
			TelemetryService.reportError(new Error("cancelTask: Failed to abort task"), TelemetryEventName.WEBVIEW_ERROR)
		})

		// Defensive safeguard: if current instance already changed, skip rehydrate
		const current = this.getCurrentTask()
		if (current && current.instanceId !== originalInstanceId) {
			this.log(
				`[cancelTask] Skipping rehydrate: current instance ${current.instanceId} != original ${originalInstanceId}`,
			)
			return
		}

		// Final race check before rehydrate to avoid duplicate rehydration
		{
			const currentAfterCheck = this.getCurrentTask()
			if (currentAfterCheck && currentAfterCheck.instanceId !== originalInstanceId) {
				this.log(
					`[cancelTask] Skipping rehydrate after final check: current instance ${currentAfterCheck.instanceId} != original ${originalInstanceId}`,
				)
				return
			}
		}

		if (!historyItem) {
			return
		}

		// Clears task again, so we need to abortTask manually above.
		await this.createTaskWithHistoryItem({ ...historyItem, rootTask, parentTask })
	}

	// Clear the current task without treating it as a subtask.
	// This is used when the user cancels a task that is not a subtask.
	public async clearTask(): Promise<void> {
		await (this.taskCoordinator?.clearTask() ?? ClineProvider.prototype.clearTaskInternal.call(this))
	}

	private async clearTaskInternal(): Promise<void> {
		if (this.stack.size > 0) {
			const task = this.stack.current
			logger.info("ClineProvider", `clearTask: clearing task ${task?.taskId}.${task?.instanceId}`)
			await this.stack.pop()
		}
	}

	public resumeTask(taskId: string): void {
		if (this.taskCoordinator) {
			this.taskCoordinator.resumeTask(taskId)
			return
		}
		ClineProvider.prototype.resumeTaskInternal.call(this, taskId)
	}

	private resumeTaskInternal(taskId: string): void {
		// Use the existing showTaskWithId method which handles both current and
		// historical tasks.
		this.showTaskWithId(taskId).catch((error) => {
			this.log(`Failed to resume task ${taskId}: ${error.message}`)
		})
	}

	// Modes

	public async getCustomModes() {
		return this.customModesManager.getCustomModes()
	}

	public async getModes(): Promise<{ slug: string; name: string }[]> {
		try {
			const customModes = await this.customModesManager.getCustomModes()
			return [...DEFAULT_MODES, ...customModes].map(({ slug, name }) => ({ slug, name }))
		} catch (_error) {
			return DEFAULT_MODES.map(({ slug, name }) => ({ slug, name }))
		}
	}

	public async getMode(): Promise<string> {
		const { mode } = await this.getState()
		return mode
	}

	public async setMode(mode: string): Promise<void> {
		await this.setValues({ mode })
	}

	// Provider Profiles

	public async getProviderProfiles(): Promise<{ name: string; provider?: string }[]> {
		const { listApiConfigMeta = [] } = await this.getState()
		return listApiConfigMeta.map((profile) => ({ name: profile.name, provider: profile.apiProvider }))
	}

	public async getProviderProfile(): Promise<string> {
		const { currentApiConfigName = "default" } = await this.getState()
		return currentApiConfigName
	}

	public async setProviderProfile(name: string): Promise<void> {
		await this.activateProviderProfile({ name })
	}

	// Telemetry

	public async getTelemetryProperties(): Promise<TelemetryProperties> {
		const state = await this.getState()
		return getTelemetryProperties(this, state)
	}

	public get cwd() {
		return this.currentWorkspacePath || getWorkspacePath()
	}

	/**
	 * Delegate parent task and open child task.
	 *
	 * - Enforce single-open invariant
	 * - Persist parent delegation metadata
	 * - Emit TaskDelegated (task-level; API forwards to provider/bridge)
	 * - Create child as sole active and switch mode to child's mode
	 */
	public async delegateParentAndOpenChild(params: {
		parentTaskId: string
		message: string
		initialTodos: TodoItem[]
		mode: string
		isolationLevel?: string
		forkedContextSummary?: string
	}): Promise<Task> {
		return delegateParentAndOpenChildWithProvider(this, params)
	}

	/**
	 * Reopen parent task from delegation with write-back and events.
	 */
	public async reopenParentFromDelegation(params: {
		parentTaskId: string
		childTaskId: string
		completionResultSummary: string
	}): Promise<void> {
		return reopenParentFromDelegationWithProvider(this, params)
	}

	/**
	 * Convert a file path to a webview-accessible URI
	 * This method safely converts file paths to URIs that can be loaded in the webview
	 *
	 * @param filePath - The absolute file path to convert
	 * @returns The webview URI string, or the original file URI if conversion fails
	 * @throws {Error} When webview is not available
	 * @throws {TypeError} When file path is invalid
	 */
	public convertToWebviewUri(filePath: string): string {
		try {
			const fileUri = vscode.Uri.file(filePath)

			// Check if we have a webview available
			if (this.view?.webview) {
				const webviewUri = this.view.webview.asWebviewUri(fileUri)
				return webviewUri.toString()
			}

			// Specific error for no webview available
			logger.error("ClineProvider", "No webview available for URI conversion")
			// Fallback to file URI if no webview available
			return fileUri.toString()
		} catch (error) {
			// More specific error handling
			if (error instanceof TypeError) {
				logger.error("ClineProvider", "Invalid file path provided for URI conversion:", error)
			} else {
				logger.error("ClineProvider", "Failed to convert to webview URI:", error)
			}
			TelemetryService.reportError(error, TelemetryEventName.WEBVIEW_ERROR)
			// Return file URI as fallback
			return vscode.Uri.file(filePath).toString()
		}
	}
}

