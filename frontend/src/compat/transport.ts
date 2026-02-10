export interface TransportRequest {
  method: "GET" | "POST";
  path: string;
  body?: unknown;
}

export interface CompatTransport {
  readonly kind: "mock" | "http";
  request<T = unknown>(req: TransportRequest): Promise<T>;
}
