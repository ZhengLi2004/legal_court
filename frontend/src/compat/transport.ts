export interface TransportRequest {
  method: "GET" | "POST";
  path: string;
  body?: unknown;
}

export interface CompatTransport {
  readonly kind: "http";
  request<T = unknown>(req: TransportRequest): Promise<T>;
}
