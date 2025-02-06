import { Plugin, IAgentRuntime, Memory, elizaLogger, generateText, ModelClass } from "@elizaos/core";

interface PluginConfig {
  apiKey: string;
  baseUrl?: string;
  runtime?: IAgentRuntime;
}

interface SizeData {
  prediction_id: string;
  measurement: number;
  age: number;
  social_connections: number;
  wallet_address: string;
}

interface MeasurementResponse {
  prediction_id: string;
  measurement: number;
  website_url: string;
  formatted_response: string;
}

export class SizeMeasurementPlugin implements Plugin {
  name = "SizeMeasurementPlugin";
  description = "Measures objects in images and provides fun responses";

  private apiKey?: string;
  private baseUrl: string;
  private generateText: (params: { context: string; modelClass: ModelClass }) => Promise<string>;

  constructor() {
    this.baseUrl = "https://sizematters.app";
  }

  init(params: { pluginConfig?: Record<string, any> }): void {
    elizaLogger.log("[SizeMeasurementPlugin] Initializing...");

    const config = params.pluginConfig as PluginConfig;
    this.apiKey = config?.apiKey;

    if (config?.baseUrl) {
      this.baseUrl = config.baseUrl;
    }

    if (!this.apiKey) {
      throw new Error("[SizeMeasurementPlugin] API key is required");
    }

    if (!config?.runtime) {
      throw new Error("[SizeMeasurementPlugin] Runtime is required");
    }

    this.generateText = async (params: { context: string; modelClass: ModelClass }) => {
      return generateText({
        runtime: config.runtime,
        context: params.context,
        modelClass: params.modelClass,
      });
    };

    elizaLogger.log("[SizeMeasurementPlugin] Initialized with config:", config);
  }

  private async generateFunnyResponse(measurement: number): Promise<string> {
    const context = `Generate a funny, playful one-liner about something that measures ${measurement}cm. Make it witty and teasing, but not mean. Include emojis.`;

    try {
      const funnyResponse = await this.generateText({
        context,
        modelClass: ModelClass.SMALL,
      });
      return funnyResponse.trim();
    } catch (error) {
      elizaLogger.error("[SizeMeasurementPlugin] Error generating funny response:", error);
      // Fallback responses if LLM fails
      const fallbackResponses = [
        `📏 ${measurement}cm? That's what she said! 😏`,
        `🔍 ${measurement}cm - Is that all? Just kidding! 😂`,
        `📐 ${measurement}cm - Size does matter, but so does confidence! 💪`
      ];
      return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    }
  }

  private async generateMeasurementResponse(data: SizeData): Promise<MeasurementResponse> {
    const measurement_url = data.prediction_id ? `${this.baseUrl}/photo/${data.prediction_id}` : this.baseUrl;
    const funnyResponse = await this.generateFunnyResponse(data.measurement);

    return {
      prediction_id: data.prediction_id,
      measurement: data.measurement,
      website_url: this.baseUrl,
      formatted_response: `${funnyResponse}\n\nSize: ${data.measurement}cm\nSubmit to win $SIZE: ${measurement_url}`
    };
  }

  async measureImage(imageUrl: string): Promise<MeasurementResponse | string> {
    try {
      elizaLogger.log("[SizeMeasurementPlugin] Starting measurement process for image:", imageUrl);

      if (!this.apiKey) {
        throw new Error("API key not configured");
      }

      // Send the image URL directly to the measurement endpoint
      const measureResponse = await fetch(`${this.baseUrl}/api/measure`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_url: imageUrl
        }),
      });

      elizaLogger.log("[SizeMeasurementPlugin] Measure response:", measureResponse);

      if (!measureResponse.ok) {
        throw new Error(`Measurement API request failed: ${measureResponse.statusText}`);
      }

      const data = await measureResponse.json();
      elizaLogger.log("[SizeMeasurementPlugin] Received measurement data:", data);

      // Transform API response
      const sizeData: SizeData = {
        prediction_id: data.prediction_id,
        measurement: parseFloat(data.measurement),
        age: data.age,
        social_connections: data.social_connections,
        wallet_address: data.wallet_address,
      };

      return this.generateMeasurementResponse(sizeData);
    } catch (error) {
      elizaLogger.error("[SizeMeasurementPlugin] Error:", error);
      return `Whoopsie! My size-o-meter is having a moment! 🎪 ${
        error instanceof Error ? error.message : "It's not you, it's me!"
      }`;
    }
  }

  async handleMessage(runtime: IAgentRuntime, message: Memory): Promise<MeasurementResponse | string | null> {
    try {
      const content = message.content as { text?: string; image?: { url: string; type: string } };

      // Check if message contains both text asking about size and an image
      if (!content.text?.toLowerCase().includes("size") || !content.image?.url) {
        return null;
      }

      elizaLogger.log("[SizeMeasurementPlugin] Processing message with image");
      return await this.measureImage(content.image.url);
    } catch (error) {
      elizaLogger.error("[SizeMeasurementPlugin] Message handling error:", error);
      return null;
    }
  }

  cleanup(): void {
    elizaLogger.log("[SizeMeasurementPlugin] Cleaning up...");
  }
}