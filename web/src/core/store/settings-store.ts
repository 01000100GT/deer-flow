// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { create } from "zustand";

import type { MCPServerMetadata, SimpleMCPServerMetadata } from "../mcp";

const SETTINGS_KEY = "DEER-SETTINGS";
const CURRENT_SETTINGS_VERSION = 1;

const DEFAULT_SETTINGS: SettingsState = {
  general: {
    autoAcceptedPlan: false,
    enableBackgroundInvestigation: false,
    maxPlanIterations: 1,
    maxStepNum: 3,
    maxSearchResults: 3,
    reportStyle: "academic",
  },
  mcp: {
    servers: [],
  },
};

export type SettingsState = {
  general: {
    autoAcceptedPlan: boolean;
    enableBackgroundInvestigation: boolean;
    maxPlanIterations: number;
    maxStepNum: number;
    maxSearchResults: number;
    reportStyle: "academic" | "popular_science" | "news" | "social_media";
  };
  mcp: {
    servers: MCPServerMetadata[];
  };
};

export const useSettingsStore = create<SettingsState>(() => ({
  ...DEFAULT_SETTINGS,
}));

export const useSettings = (key: keyof SettingsState) => {
  return useSettingsStore((state) => state[key]);
};

export const changeSettings = (settings: SettingsState) => {
  useSettingsStore.setState(settings);
};

export function loadSettings(): SettingsState {
  const json = localStorage.getItem(SETTINGS_KEY);
  if (!json) {
    return DEFAULT_SETTINGS;
  }
  try {
    const data = JSON.parse(json);
    if (data.version !== CURRENT_SETTINGS_VERSION) {
      console.warn(
        `Old settings version detected (found ${data.version}, expected ${CURRENT_SETTINGS_VERSION}). Discarding old settings.`,
      );
      return DEFAULT_SETTINGS;
    }
    return { ...DEFAULT_SETTINGS, ...data.settings };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: SettingsState) {
  try {
    const dataToStore = {
      version: CURRENT_SETTINGS_VERSION,
      settings: settings,
    };
    const json = JSON.stringify(dataToStore);
    localStorage.setItem(SETTINGS_KEY, json);
  } catch {
    // Do nothing
  }
}

useSettingsStore.subscribe((state) => {
  saveSettings(state);
});

export const getChatStreamSettings = () => {
  let mcpSettings:
    | {
        servers: Record<
          string,
          MCPServerMetadata & {
            enabled_tools: string[];
            add_to_agents: string[];
          }
        >;
      }
    | undefined = undefined;
  const { mcp, general } = useSettingsStore.getState();
  const mcpServers = mcp.servers.filter((server) => server.enabled);
  if (mcpServers.length > 0) {
    mcpSettings = {
      servers: mcpServers.reduce((acc, cur) => {
        const { transport, env } = cur;
        let server: SimpleMCPServerMetadata;
        if (transport === "stdio") {
          server = {
            name: cur.name,
            transport,
            env,
            command: cur.command,
            args: cur.args,
          };
        } else {
          server = {
            name: cur.name,
            transport,
            env,
            url: cur.url,
          };
        }
        return {
          ...acc,
          [cur.name]: {
            ...server,
            enabled_tools: cur.tools.map((tool) => tool.name),
            add_to_agents: ["researcher"],
          },
        };
      }, {}),
    };
  }
  return {
    ...general,
    mcpSettings,
  };
};

export function setReportStyle(value: "academic" | "popular_science" | "news" | "social_media") {
  useSettingsStore.setState((state) => ({
    general: {
      ...state.general,
      reportStyle: value,
    },
  }));
  saveSettings(useSettingsStore.getState());
}

export function setEnableBackgroundInvestigation(value: boolean) {
  useSettingsStore.setState((state) => ({
    general: {
      ...state.general,
      enableBackgroundInvestigation: value,
    },
  }));
  saveSettings(useSettingsStore.getState());
}

loadSettings();
