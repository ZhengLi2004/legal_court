import { HttpTransport, HttpTransportError } from "./httpTransport";
import type { CompatTransport, TransportRequest } from "./transport";

export interface AdapterOptions {
  baseUrl?: string;
}

export interface RequestCandidate {
  method: "GET" | "POST";
  path: string;
  body?: unknown;
}

function isNotFound(err: unknown): boolean {
  return (
    err instanceof HttpTransportError &&
    (err.status === 404 || err.status === 405)
  );
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

  async request(candidate: RequestCandidate): Promise<unknown> {
    const req: TransportRequest = {
      method: candidate.method,
      path: candidate.path,
      body: candidate.body,
    };

    return this.activeTransport.request(req);
  }

  async callWithCandidates(candidates: RequestCandidate[]): Promise<unknown> {
    let lastError: unknown;

    for (const candidate of candidates) {
      try {
        return await this.request(candidate);
      } catch (err) {
        if (isNotFound(err)) {
          lastError = err;
          continue;
        }

        lastError = err;
      }
    }

    throw lastError ?? new Error("No compatible endpoint found");
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
