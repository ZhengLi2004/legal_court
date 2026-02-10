import type { CompatTransport, TransportRequest } from "./transport";

export class HttpTransportError extends Error {
  readonly status: number;
  readonly path: string;

  constructor(message: string, status: number, path: string) {
    super(message);
    this.status = status;
    this.path = path;
  }
}

export class HttpTransport implements CompatTransport {
  readonly kind = "http" as const;
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async request<T = unknown>(req: TransportRequest): Promise<T> {
    const url = `${this.baseUrl}${req.path}`;

    const response = await fetch(url, {
      method: req.method,
      headers: {
        "Content-Type": "application/json",
      },
      body: req.body === undefined ? undefined : JSON.stringify(req.body),
    });

    if (!response.ok) {
      const text = await response.text();

      throw new HttpTransportError(
        text || `HTTP ${response.status}`,
        response.status,
        req.path,
      );
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return (await response.json()) as T;
  }
}
