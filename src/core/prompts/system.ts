import * as vscode from "vscode"
import crypto from "crypto"

import { type ModeConfig, type PromptComponent, type CustomModePrompts, type TodoItem } from "@njust-ai-cj/types"
import { renderPrompt } from "@njust-ai-cj/prompt-engine"

import { Mode, modes, defaultModeSlug, getModeBySlug, getGroupName, getModeSelection } from "../../shared/modes"
import { DiffStrategy } from "../../shared/tools"
import { isEmpty } from "../../utils/object"

import type { IMcpHubService } from "../../services/mcp/interfaces/IMcpHubService"
import { CodeIndexManager } from "../../services/code-index/manager"
import { SkillsManager } from "../../services/skills/SkillsManager"

import { buildBudgetedSessionMemoryPrompt } from "../condense/sessionMemoryCompact"
import type { SystemPromptSettings } from "./types"
import { getMemrlMemorySection } from "./sections/memrl-memory"
import {
	getRulesSection,
	getSystemInfoSection,
	getObjectiveSection,
	getSharedToolUseSection,
	getToolUseGuidelinesSection,
	getCapabilitiesSection,
	getModesSection,
	addCustomInstructions,
	markdownFormattingSection,
	getSkillsSection,
	filterCangjieSkillRoutingRows,
	getOutputEfficiencySection,
} from "./sections"
import {
	DEFAULT_CANGJIE_CONTEXT_TOKEN_BUDGET,
	detectCangjieRelevanceForAuxiliaryModes,
	getCangjieContextSection,
} from "./sections/cangjie-context"
import { Package } from "../../shared/package"
import { getMultiFileContextSection } from "./sections/multi-file-context"
import { estimatePromptTokens, trimSectionsByBudget, derivePromptTokenBudget } from "./tokenBudget"
import type { SectionBudget } from "./tokenBudget"

export const SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "\n\n====\n\nSYSTEM_PROMPT_DYNAMIC_BOUNDARY\n\n====\n\n"

/**
 * Scale Cangjie dynamic context token budget based on the active model's context window.
 *
 * Called by TaskRequestBuilder and passed as `settings.cangjieContextTokenBudget`.
 * Models with larger context windows get more room for Cangjie-specific context
 * (diagnostics, corpus snippets, editing context, etc.); smaller windows are
 * automatically compressed to protect essential diagnostic sections.
 *
 * Mapping:
 * - >= 200k tokens → 6000
 * - >= 100k tokens → 4500
 * - >= 64k  tokens → 3800
 * - >= 32k  tokens → 3000
 * - >= 16k  tokens → 2400
 * - < 16k  tokens → max(800, min(DEFAULT, floor(window * 0.08)))
 * - UnsafeAny / < 4096 → DEFAULT_CANGJIE_CONTEXT_TOKEN_BUDGET (4800)
 *
 * The result of this function is *not* the final budget — it feeds into
 * `resolveCangjieContextTokenBudget`, which allows the user's VS Code
 * `cangjieContextTokenBudget` setting to override it.
 */
export function deriveCangjieContextTokenBudgetFromContextWindow(contextWindow: number | undefined): number {
	const fallback = DEFAULT_CANGJIE_CONTEXT_TOKEN_BUDGET
	if (contextWindow === undefined || contextWindow <= 0 || contextWindow < 4096) {
		return fallback
	}
	if (contextWindow >= 200_000) return 6000
	if (contextWindow >= 100_000) return 4500
	if (contextWindow >= 64_000) return 3800
	if (contextWindow >= 32_000) return 3000
	if (contextWindow >= 16_000) return 2400
	return Math.max(800, Math.min(fallback, Math.floor(contextWindow * 0.08)))
}

/**
 * Resolve the effective Cangjie context token budget with the following priority:
 *
 * 1. VS Code setting `njust-ai.cangjieContextTokenBudget` — explicit user override
 * 2. `settings.cangjieContextTokenBudget` — model-scaled value from
 *    `deriveCangjieContextTokenBudgetFromContextWindow`
 * 3. `DEFAULT_CANGJIE_CONTEXT_TOKEN_BUDGET` (4800) — hardcoded fallback
 *
 * This means small-context models will automatically receive a compressed
 * Cangjie block (e.g. 2400 tokens at 16k context), but a user who sets the
 * VS Code config can force any budget regardless of model size.
 */
function resolveCangjieContextTokenBudget(settings?: SystemPromptSettings): number {
	const fromConfig = vscode.workspace.getConfiguration(Package.name).get<number>("cangjieContextTokenBudget")
	if (typeof fromConfig === "number" && fromConfig > 0) {
		return Math.floor(fromConfig)
	}
	const explicit = settings?.cangjieContextTokenBudget
	if (typeof explicit === "number" && explicit > 0) {
		return Math.floor(explicit)
	}
	return DEFAULT_CANGJIE_CONTEXT_TOKEN_BUDGET
}

// Helper function to get prompt component, filtering out empty objects
export function getPromptComponent(
	customModePrompts: CustomModePrompts | undefined,
	mode: string,
): PromptComponent | undefined {
	const component = customModePrompts?.[mode]
	// Return undefined if component is empty
	if (isEmpty(component)) {
		return undefined
	}
	return component
}

export type SystemPromptParts = {
	staticPart: string
	dynamicPart: string
	fullPrompt: string
	cacheBreakpoints?: number[] // Character offsets where cache boundaries exist
	perToolHashes?: Record<string, string>
}

/**
 * Consolidated config object for system prompt generation.
 * Replaces the 13+ positional parameters on generatePrompt / SYSTEM_PROMPT / SYSTEM_PROMPT_PARTS.
 */
export interface SystemPromptConfig {
	context: vscode.ExtensionContext
	cwd: string
	supportsComputerUse: boolean
	mode: Mode
	mcpHub?: IMcpHubService
	diffStrategy?: DiffStrategy
	promptComponent?: PromptComponent
	customModeConfigs?: ModeConfig[]
	globalCustomInstructions?: string
	experiments?: Record<string, boolean>
	language?: string
	rooIgnoreInstructions?: string
	settings?: SystemPromptSettings
	todoList?: TodoItem[]
	modelId?: string
	skillsManager?: SkillsManager
}

async function generatePrompt(cfg: SystemPromptConfig): Promise<SystemPromptParts>
/** @deprecated Use the config-object overload. */
async function generatePrompt(
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mode: Mode,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	promptComponent?: PromptComponent,
	customModeConfigs?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<SystemPromptParts>
async function generatePrompt(
	contextOrCfg: vscode.ExtensionContext | SystemPromptConfig,
	cwd?: string,
	supportsComputerUse?: boolean,
	mode?: Mode,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	promptComponent?: PromptComponent,
	customModeConfigs?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<SystemPromptParts> {
	const cfg: SystemPromptConfig =
		"context" in contextOrCfg && "cwd" in contextOrCfg && "mode" in contextOrCfg
			? (contextOrCfg as SystemPromptConfig)
			: {
					context: contextOrCfg as vscode.ExtensionContext,
					cwd: cwd!,
					supportsComputerUse: supportsComputerUse!,
					mode: mode!,
					mcpHub,
					diffStrategy,
					promptComponent,
					customModeConfigs,
					globalCustomInstructions,
					experiments,
					language,
					rooIgnoreInstructions,
					settings,
					todoList,
					modelId,
					skillsManager,
				}
	return generatePromptImpl(cfg)
}

async function generatePromptImpl(cfg: SystemPromptConfig): Promise<SystemPromptParts> {
	const {
		context,
		cwd,
		supportsComputerUse: _supportsComputerUse,
		mode,
		mcpHub,
		diffStrategy: _diffStrategy,
		promptComponent,
		customModeConfigs,
		globalCustomInstructions,
		experiments: _experiments,
		language,
		rooIgnoreInstructions,
		settings,
		skillsManager,
		modelId: _modelId,
	} = cfg
	if (!context) {
		throw new Error("Extension context is required for generating system prompt")
	}

	// Get the full mode config to ensure we have the role definition (used for groups, etc.)
	const modeConfig = getModeBySlug(mode, customModeConfigs) || modes.find((m) => m.slug === mode) || modes[0]
	const { roleDefinition, baseInstructions } = getModeSelection(mode, promptComponent, customModeConfigs)
	let effectiveBaseInstructions = baseInstructions
	if (mode === "cangjie" && skillsManager) {
		const discovered = new Set(skillsManager.getSkillsForMode("cangjie").map((s) => s.name))
		effectiveBaseInstructions = filterCangjieSkillRoutingRows(baseInstructions, discovered)
	}

	// Check if MCP functionality should be included
	const hasMcpGroup = modeConfig!.groups.some((groupEntry) => getGroupName(groupEntry) === "mcp")
	const hasMcpServers = mcpHub && mcpHub.getServers().length > 0
	const shouldIncludeMcp = hasMcpGroup && hasMcpServers

	CodeIndexManager.getInstance(context, cwd)

	// Tool calling is native-only.
	const _effectiveProtocol = "native"
	const cangjieSkillTriggerText = settings?.lastUserMessageForCangjieHint

	const [modesSection, skillsSection] = await Promise.all([
		getModesSection(context),
		getSkillsSection(skillsManager, mode as string, cangjieSkillTriggerText),
	])

	const pruningEnabled = settings?.enableTurnAwarePromptPruning ?? true
	const isFollowupTurn = pruningEnabled && (settings?.turnIndex ?? 0) > 0

	let cangjieTokenBudget = resolveCangjieContextTokenBudget(settings)
	const trimCangjieBlockOnFollowup =
		isFollowupTurn &&
		(mode === "cangjie" ||
			((mode === "ask" || mode === "architect") &&
				detectCangjieRelevanceForAuxiliaryModes(cwd, settings?.lastUserMessageForCangjieHint)))
	if (trimCangjieBlockOnFollowup) {
		cangjieTokenBudget = Math.max(800, Math.floor(cangjieTokenBudget * 0.65))
	}
	const cangjieContextSection = await getCangjieContextSection(
		cwd,
		mode as string,
		context.extensionPath,
		cangjieTokenBudget,
		context.globalStorageUri.fsPath,
		settings?.lastUserMessageForCangjieHint,
		settings?.cangjieContextIntensity,
		settings?.cangjieRecentBuildRootCauses,
		settings?.cangjieRepairDirective,
	)
	const multiFileContextSection = cangjieContextSection ? "" : getMultiFileContextSection(cwd)

	// Tools catalog is not included in the system prompt.
	const toolsCatalog = ""

	const webSearchSection = settings?.enableWebSearch
		? `

====

WEB SEARCH

You have the web_search tool available for retrieving real-time information from the internet. Do NOT say "web search is unavailable" — you CAN search the web.

CRITICAL: NEVER use execute_command with curl, wget, httpie, Invoke-WebRequest, or any HTTP client to fetch web content. Use web_search instead.

EFFICIENCY RULES (IMPORTANT):
- Use AT MOST 1-2 searches per user question. One well-crafted search query is usually enough.
- Combine multiple aspects into a SINGLE search query instead of searching separately for each aspect.
  BAD: search "gold price" then search "gold price today" then search "gold price USD"
  GOOD: search "today gold price USD" (one query covers everything)
- Do NOT repeat or rephrase searches if the first search returned relevant results.
- If the first search gives a clear answer, STOP searching and respond immediately.
- Only do a second search if the first one truly failed to answer the question.

WHEN TO USE:
- When the user asks about recent events, current prices, latest versions, or time-sensitive topics
- When you lack knowledge about a specific project, product, or technology
- When the user explicitly asks you to search or look something up

WHEN NOT TO USE:
- When you already have reliable knowledge to answer the question
- When the question is about general programming concepts, syntax, or well-established patterns
- When previous search results in this conversation already contain the answer

HOW TO USE:
- Craft ONE specific, comprehensive search query that covers the user's full question
- Always synthesize results and cite source URLs
- Prefer web search results over training data when they conflict (search results are more recent)`
		: ""

	const reducedModesSection = isFollowupTurn ? "" : modesSection
	const reducedCapabilitiesSection = isFollowupTurn
		? ""
		: getCapabilitiesSection(cwd, shouldIncludeMcp ? mcpHub : undefined, settings?.taskId)

	// Build named sections for budget-aware trimming
	const roleDefinitionText = roleDefinition
	const formattingText = markdownFormattingSection()
	const toolUseText = `${getSharedToolUseSection()}${toolsCatalog}\n\n\t${getToolUseGuidelinesSection()}`
	const capabilitiesText = reducedCapabilitiesSection
	const webSearchText = webSearchSection
	const modesText = reducedModesSection
	const skillsText = skillsSection ? `\n${skillsSection}` : ""
	const cangjieText = cangjieContextSection ? `\n${cangjieContextSection}` : ""
	const multiFileText = multiFileContextSection ? `\n${multiFileContextSection}` : ""
	const outputEfficiencyText = getOutputEfficiencySection()
	const sessionMemoryText = settings?.sessionMemory
		? buildBudgetedSessionMemoryPrompt(settings.sessionMemory)
		: ""
	const memrlMemoryText = getMemrlMemorySection(
		settings?.memrlEpisodicHints ?? "",
		settings?.memrlLtmRules ?? "",
	)
	const rulesText = getRulesSection(cwd, settings)
	const systemInfoText = getSystemInfoSection(cwd)
	const objectiveText = getObjectiveSection()
	const customInstructionsText = await addCustomInstructions(effectiveBaseInstructions, globalCustomInstructions || "", cwd, mode, {
		language: language ?? "en",
		rooIgnoreInstructions,
		settings,
	})

	// Define section budgets with priorities for trimming
	const sectionEntries: { name: string; text: string; priority: number; required: boolean }[] = [
		{ name: "roleDefinition", text: roleDefinitionText, priority: 0, required: true },
		{ name: "formatting", text: formattingText, priority: 0, required: true },
		{ name: "toolDescriptions", text: toolUseText, priority: 0, required: true },
		{ name: "capabilitiesSection", text: capabilitiesText, priority: 3, required: false },
		{ name: "webSearchSection", text: webSearchText, priority: 2, required: false },
		{ name: "modesSection", text: modesText, priority: 3, required: false },
		{ name: "skillsSection", text: skillsText, priority: 2, required: false },
		{ name: "cangjieContext", text: cangjieText, priority: 1, required: false },
		{ name: "multiFileContext", text: multiFileText, priority: 1, required: false },
		{ name: "rulesSection", text: rulesText, priority: 4, required: false },
		{ name: "systemInfo", text: systemInfoText, priority: 0, required: true },
		{ name: "objective", text: objectiveText, priority: 0, required: true },
		{ name: "customInstructions", text: customInstructionsText, priority: 2, required: false },
		{ name: "outputEfficiency", text: outputEfficiencyText, priority: 1, required: false },
		{ name: "sessionMemory", text: sessionMemoryText, priority: 2, required: false },
		{ name: "memrlMemory", text: memrlMemoryText, priority: 2, required: false },
	]

	// Build SectionBudget array and apply trimming
	const budget = derivePromptTokenBudget(settings?.contextWindow)
	const maxTokens = budget?.systemPromptMaxTokens ?? Infinity

	const sectionBudgets: SectionBudget[] = sectionEntries.map((e) => ({
		name: e.name,
		priority: e.priority,
		estimatedTokens: estimatePromptTokens(e.text),
		required: e.required,
	}))

	const retainedSections = trimSectionsByBudget(sectionBudgets, maxTokens)

	// Helper to include section content only if retained
	const sec = (name: string): string => {
		const entry = sectionEntries.find((e) => e.name === name)
		if (!entry || !retainedSections.has(name)) return ""
		return entry.text
	}

	const staticPart = [sec("roleDefinition"), sec("formatting"), sec("toolDescriptions"), sec("outputEfficiency"), sec("capabilitiesSection") + sec("webSearchSection"), sec("modesSection")]
		.filter(Boolean)
		.join("\n\n")

	// Priority order: objective/system/customInstructions first so applySystemPromptBudget does not truncate mode workflow or task goal from the tail of a single blob.
	const skillsCangjieMulti = [sec("skillsSection"), sec("cangjieContext"), sec("multiFileContext")].filter(Boolean).join("")
	const dynamicSegments = [
		sec("objective"),
		sec("systemInfo"),
		sec("customInstructions"),
		skillsCangjieMulti,
		sec("rulesSection"),
		sec("sessionMemory"),
		sec("memrlMemory"),
	].filter((s) => s.length > 0)

	const renderedPrompt = renderPrompt({
		staticSections: [{ name: "static", text: staticPart, required: true }],
		dynamicSections: dynamicSegments.map((text, index) => ({ name: `dynamic-${index}`, text, required: true })),
		boundary: SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
		maxPromptTokens: budget?.systemPromptMaxTokens,
	})
	const perToolHashes: Record<string, string> = {
		toolDescriptions: crypto.createHash("sha256").update(toolUseText).digest("hex").slice(0, 16),
		capabilitiesSection: crypto.createHash("sha256").update(capabilitiesText).digest("hex").slice(0, 16),
		webSearchSection: crypto.createHash("sha256").update(webSearchText).digest("hex").slice(0, 16),
	}
	return {
		staticPart: renderedPrompt.staticPart,
		dynamicPart: renderedPrompt.dynamicPart,
		fullPrompt: renderedPrompt.fullPrompt,
		perToolHashes,
	}
}

/** Helper: resolve config from positional params (shared by SYSTEM_PROMPT_PARTS / SYSTEM_PROMPT). */
function positionalToConfig(
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	mode: Mode = defaultModeSlug,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): SystemPromptConfig {
	const promptComponent = getPromptComponent(customModePrompts, mode)
	const currentMode = getModeBySlug(mode, customModes) || modes.find((m) => m.slug === mode) || modes[0]
	return {
		context,
		cwd,
		supportsComputerUse,
		mode: currentMode!.slug,
		mcpHub,
		diffStrategy,
		promptComponent,
		customModeConfigs: customModes,
		globalCustomInstructions,
		experiments,
		language,
		rooIgnoreInstructions,
		settings,
		todoList,
		modelId,
		skillsManager,
	}
}

export async function SYSTEM_PROMPT_PARTS(cfg: SystemPromptConfig): Promise<SystemPromptParts>
/** @deprecated Use the config-object overload. */
export async function SYSTEM_PROMPT_PARTS(
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	mode?: Mode,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<SystemPromptParts>
export async function SYSTEM_PROMPT_PARTS(
	contextOrCfg: vscode.ExtensionContext | SystemPromptConfig,
	cwd?: string,
	supportsComputerUse?: boolean,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	mode?: Mode,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<SystemPromptParts> {
	if ("context" in contextOrCfg && "cwd" in contextOrCfg && "mode" in contextOrCfg) {
		return generatePrompt(contextOrCfg as SystemPromptConfig)
	}
	const cfg = positionalToConfig(
		contextOrCfg as vscode.ExtensionContext,
		cwd!, supportsComputerUse!, mcpHub, diffStrategy,
		mode, customModePrompts, customModes, globalCustomInstructions,
		experiments, language, rooIgnoreInstructions, settings,
		todoList, modelId, skillsManager,
	)
	return generatePrompt(cfg)
}

export async function SYSTEM_PROMPT(cfg: SystemPromptConfig): Promise<string>
/** @deprecated Use the config-object overload. */
export async function SYSTEM_PROMPT(
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	mode?: Mode,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<string>
export async function SYSTEM_PROMPT(
	contextOrCfg: vscode.ExtensionContext | SystemPromptConfig,
	cwd?: string,
	supportsComputerUse?: boolean,
	mcpHub?: IMcpHubService,
	diffStrategy?: DiffStrategy,
	mode?: Mode,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<string> {
	const parts = await SYSTEM_PROMPT_PARTS(
		contextOrCfg as UnsafeAny,
		cwd as UnsafeAny, supportsComputerUse as UnsafeAny, mcpHub, diffStrategy,
		mode, customModePrompts, customModes, globalCustomInstructions,
		experiments, language, rooIgnoreInstructions, settings,
		todoList, modelId, skillsManager,
	)
	return parts.fullPrompt
}
