/**
 * Core HTTP Client Abstraction
 * Professional service layer handling base URL resolution, request timeouts,
 * error normalization, and environment overrides.
 */

export class ApiError extends Error {
  public readonly status: number;
  public readonly detail: any;

  constructor(message: string, status: number, detail: any = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

export class HttpClient {
  private baseUrl: string;
  private defaultTimeout: number;

  constructor(baseUrl?: string, defaultTimeout = 10000) {
    // Resolve base URL from environment or default relative proxy path
    this.baseUrl =
      baseUrl ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      '/api/v1';
    this.defaultTimeout = defaultTimeout;
  }

  /**
   * Returns fully-qualified endpoint URL
   */
  private buildUrl(endpoint: string): string {
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    // Remove trailing slash from base if present
    const cleanBase = this.baseUrl.replace(/\/+$/, '');
    return `${cleanBase}${cleanEndpoint}`;
  }

  /**
   * Executes HTTP request with timeout & standardized error parsing
   */
  public async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const { timeoutMs = this.defaultTimeout, headers, ...customConfig } = options;
    const url = this.buildUrl(endpoint);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const config: RequestInit = {
      ...customConfig,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...headers,
      },
      signal: controller.signal,
    };

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData: any = null;
        try {
          errorData = await response.json();
        } catch {
          errorData = { message: response.statusText };
        }

        const message =
          errorData?.detail
            ? typeof errorData.detail === 'string'
              ? errorData.detail
              : JSON.stringify(errorData.detail)
            : errorData?.message || `HTTP ${response.status} Request Failed`;

        throw new ApiError(message, response.status, errorData);
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);

      if (err.name === 'AbortError') {
        throw new ApiError(`Request timeout after ${timeoutMs}ms`, 408);
      }

      if (err instanceof ApiError) {
        throw err;
      }

      throw new ApiError(
        err.message || 'Network error: Failed to connect to server',
        0
      );
    }
  }

  public get<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  public post<T>(
    endpoint: string,
    body: any,
    options?: RequestOptions
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body),
    });
  }
}

// Export default singleton instance
export class HttpClientInstance extends HttpClient {}
export const defaultHttpClient = new HttpClient();
