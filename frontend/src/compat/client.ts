import { HttpTransport } from "./httpTransport";
import type { CompatTransport, TransportRequest } from "./transport";

export interface AdapterOptions {
  baseUrl?: string;
}

export interface RequestPayload {
  method: "GET" | "POST";
  path: string;
  body?: unknown;
}

export class CompatClient {
  readonly baseUrl: string;
  private readonly httpTransport: CompatTransport;
  private activeTransport: CompatTransport;

  constructor(options: AdapterOptions = {}) {
    this.baseUrl = options.baseUrl ?? "/api";
    this.httpTransport = new HttpTransport(this.baseUrl);
    this.activeTransport = this.httpTransport;
  }

  get transportKind(): "http" {
    return this.activeTransport.kind;
  }

  async request(payload: RequestPayload): Promise<unknown> {
    const req: TransportRequest = {
      method: payload.method,
      path: payload.path,
      body: payload.body,
    };

    return this.activeTransport.request(req);
  }
}

export function withQuery(
  path: string,
  query: Record<string, string | number | undefined>,
): string {
  const params = new URLSearchParams();

  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null) {
      return;
    }

    params.set(key, String(value));
  });

  const suffix = params.toString();
  return suffix ? `${path}?${suffix}` : path;
}
