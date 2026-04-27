/**
 * Phase 4 wallet onboarding helper for the PRSM JS SDK.
 *
 * Implements the client-side half of the SIWE + identity-binding flow
 * exposed by ``prsm.interface.api.wallet_api`` (POST /siwe/nonce, POST
 * /siwe/verify, POST /wallet/bind). Coinbase Wallet SDK,
 * @walletconnect/web3wallet, and embedded-wallet vendors (Privy in
 * Phase 4 Task 4) all implement EIP-1193 — the same `request({method,
 * params})` shape — so this module accepts any such provider as a peer
 * dependency rather than baking a specific SDK in.
 *
 * Design notes:
 *   - No transitive wallet-SDK dependency. Callers wire their own
 *     provider instance (e.g. `new CoinbaseWalletSDK(...).makeWeb3Provider()`).
 *   - Pure transport: this module does not generate signing keys, store
 *     credentials, or run UI flows.
 *   - Errors surface as ``WalletAuthError`` subclasses with stable
 *     ``code`` strings (matches the backend's stable error codes).
 */

import { keccak_256 } from 'js-sha3';

export type FetchFn = typeof fetch;

/** Minimal EIP-1193 surface this module relies on. Coinbase Wallet
 *  SDK, WalletConnect, MetaMask, Privy, etc. all implement it. */
export interface EthereumProvider {
  request(args: { method: string; params?: unknown[] | object }): Promise<unknown>;
}

// ────────────────────────────────────────────────────────────────────
// Wire-format types — mirror the Python Pydantic models
// ────────────────────────────────────────────────────────────────────

export interface NonceResponse {
  nonce: string;
  domain: string;
  chain_id: number;
  expires_at_unix: number;
}

export interface SiweVerifyResponse {
  address: string;
  node_id_hex: string;
  is_new_user: boolean;
  binding_message: string;
  binding_issued_at: string;
}

export interface WalletBindRequestBody {
  wallet_address: string;
  node_id_hex: string;
  signature: string;
  issued_at: string;
}

export interface WalletBindResponse {
  wallet_address: string;
  node_id_hex: string;
  bound_at_unix: number;
  signing_message_hash: string;
}

export interface BalanceResponse {
  wallet_address: string;
  node_id_hex: string;
  ftns: string;
  usd: string | null;
  formatted: string;
  mode: 'usd' | 'ftns';
}

// ────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────

/** Base error for wallet-onboarding failures. ``code`` matches the
 *  backend's stable error codes from ``wallet_api.SIWE_ERROR_CODES``
 *  and ``BINDING_ERROR_CODES`` (e.g. "siwe_signature_invalid",
 *  "binding_conflict"). */
export class WalletAuthError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
    this.name = 'WalletAuthError';
  }
}

export class WalletAuthTransportError extends WalletAuthError {
  constructor(message: string, status: number) {
    super('transport_error', message, status);
    this.name = 'WalletAuthTransportError';
  }
}

// ────────────────────────────────────────────────────────────────────
// SIWE message construction (EIP-4361)
// ────────────────────────────────────────────────────────────────────

export interface BuildSiweOptions {
  address: string;
  nonce: string;
  domain: string;
  chainId: number;
  uri: string;
  version?: string;
  statement?: string;
  issuedAt?: string;
}

/** Convert an Ethereum address to EIP-55 checksum form.
 *
 *  EIP-1193 providers vary: MetaMask + Coinbase Wallet typically
 *  return checksummed addresses, but WalletConnect v2 and some
 *  embedded-wallet vendors (e.g. Privy) return lowercase. The
 *  Python `siwe` library REJECTS non-checksum addresses with
 *  ``ValidationError`` (round-1 review HIGH-2 finding). Normalize
 *  here so the produced SIWE message always parses backend-side
 *  regardless of provider quirks.
 *
 *  Implementation: lowercase the hex, keccak256 the ASCII bytes,
 *  uppercase nibbles whose corresponding hash nibble is ≥ 8. Uses
 *  ``@noble/hashes`` for the underlying primitive — small (~12 KB),
 *  audited, zero-dep, and the de-facto standard for Web3 stacks. */
export function toChecksumAddress(address: string): string {
  if (typeof address !== 'string') {
    throw new Error('address must be a string');
  }
  const stripped = address.toLowerCase().replace(/^0x/, '');
  if (stripped.length !== 40 || !/^[0-9a-f]{40}$/.test(stripped)) {
    throw new Error(`invalid Ethereum address: ${address}`);
  }
  // keccak256 of the ASCII-encoded lowercase hex (no 0x prefix).
  // js-sha3's keccak_256 returns a hex string when called with a string
  // (treats input as utf-8 bytes — same as ASCII for hex chars).
  const hashHex = keccak_256(stripped);
  let out = '0x';
  for (let i = 0; i < stripped.length; i++) {
    const c = stripped[i];
    if (/[0-9]/.test(c)) {
      out += c;
    } else {
      // Uppercase if hash nibble at the same position is ≥ 8
      out += parseInt(hashHex[i], 16) >= 8 ? c.toUpperCase() : c;
    }
  }
  return out;
}

/** Build an EIP-4361 SIWE message string. Whitespace and field order
 *  are fixed — the backend re-parses with the `siwe` Python library
 *  which is strict about both.
 *
 *  Layout invariants (round-1 review HIGH-1 finding):
 *    - Address is EIP-55 checksummed unconditionally (lowercase
 *      addresses fail backend validation).
 *    - Statement section's bracketing blank lines are ALWAYS emitted,
 *      even when statement is omitted — the EIP-4361 grammar requires
 *      two blank lines between address and URI in the no-statement
 *      form (one blank, one empty-statement-line, then URI). */
export function buildSiweMessage(opts: BuildSiweOptions): string {
  const version = opts.version ?? '1';
  const issuedAt = opts.issuedAt ?? new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  const checksumAddress = toChecksumAddress(opts.address);
  const lines = [
    `${opts.domain} wants you to sign in with your Ethereum account:`,
    checksumAddress,
    '',
  ];
  if (opts.statement) {
    lines.push(opts.statement);
  }
  // ALWAYS emit the second blank line — present whether statement is
  // included or omitted. Without it, `siwe.SiweMessage.from_message`
  // raises ValueError on the no-statement layout.
  lines.push('');
  lines.push(`URI: ${opts.uri}`);
  lines.push(`Version: ${version}`);
  lines.push(`Chain ID: ${opts.chainId}`);
  lines.push(`Nonce: ${opts.nonce}`);
  lines.push(`Issued At: ${issuedAt}`);
  return lines.join('\n');
}

// ────────────────────────────────────────────────────────────────────
// WalletAuth client
// ────────────────────────────────────────────────────────────────────

export interface WalletAuthOptions {
  /** Base URL of the PRSM API (e.g. https://api.prsm-network.com). */
  baseUrl: string;
  /** Fetch implementation. Defaults to globalThis.fetch; pass a custom
   *  one for environments without a global fetch (older Node, jest)
   *  or to inject auth headers. */
  fetchFn?: FetchFn;
}

export interface ConnectCoinbaseWalletOptions {
  /** EIP-4361 domain. Must match the server's expected_domain. */
  domain: string;
  /** Origin URI of the login page. */
  uri: string;
  /** Optional human-readable statement embedded in the SIWE message. */
  statement?: string;
  /** SIWE chain_id; defaults to the server-issued chain_id from the nonce response. */
  chainId?: number;
  /** Override the EIP-4361 version (default '1'). */
  version?: string;
  /** Override issued-at timestamp (ISO-8601 UTC ending with Z). */
  issuedAt?: string;
}

export interface ConnectCoinbaseWalletResult {
  address: string;
  nodeIdHex: string;
  isNewUser: boolean;
  binding: WalletBindResponse;
}

export class WalletAuth {
  private readonly baseUrl: string;
  private readonly fetchFn: FetchFn;

  constructor(options: WalletAuthOptions) {
    if (!options.baseUrl) {
      throw new Error('WalletAuth requires baseUrl');
    }
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    const f =
      options.fetchFn ??
      (typeof globalThis !== 'undefined' ? globalThis.fetch : undefined);
    if (!f) {
      throw new Error(
        'No fetch implementation available. Pass options.fetchFn explicitly.',
      );
    }
    // Bind to globalThis so `this` inside fetch is correct in browsers.
    this.fetchFn = f.bind(globalThis as unknown as object);
  }

  // ─── Raw HTTP wrappers ─────────────────────────────────────────────

  async requestNonce(chainId?: number): Promise<NonceResponse> {
    return this.post<NonceResponse>('/api/v1/auth/wallet/siwe/nonce', {
      ...(chainId !== undefined ? { chain_id: chainId } : {}),
    });
  }

  async verifySiwe(message: string, signature: string): Promise<SiweVerifyResponse> {
    return this.post<SiweVerifyResponse>('/api/v1/auth/wallet/siwe/verify', {
      message,
      signature,
    });
  }

  async bindWallet(req: WalletBindRequestBody): Promise<WalletBindResponse> {
    return this.post<WalletBindResponse>('/api/v1/auth/wallet/bind', req);
  }

  async getBinding(walletAddress: string): Promise<WalletBindResponse | null> {
    const url = new URL(`${this.baseUrl}/api/v1/auth/wallet/binding`);
    url.searchParams.set('wallet_address', walletAddress);
    const res = await this.fetchFn(url.toString());
    if (res.ok) {
      const body = (await res.json()) as WalletBindResponse | null;
      return body;
    }
    throw await this.toError(res);
  }

  async getBalance(
    walletAddress: string,
    mode: 'usd' | 'ftns' = 'usd',
  ): Promise<BalanceResponse> {
    const url = new URL(`${this.baseUrl}/api/v1/auth/wallet/balance`);
    url.searchParams.set('wallet_address', walletAddress);
    url.searchParams.set('mode', mode);
    const res = await this.fetchFn(url.toString());
    if (!res.ok) {
      throw await this.toError(res);
    }
    return (await res.json()) as BalanceResponse;
  }

  // ─── Composed flow ─────────────────────────────────────────────────

  /**
   * Full Coinbase-Wallet-style onboarding flow against an EIP-1193
   * provider:
   *   1. eth_requestAccounts → wallet address
   *   2. POST /siwe/nonce → server-issued nonce + domain/chain_id
   *   3. Build SIWE message client-side
   *   4. personal_sign → SIWE signature
   *   5. POST /siwe/verify → node_id + canonical binding message
   *   6. personal_sign → binding signature
   *   7. POST /wallet/bind → durable wallet ↔ node_id binding
   *
   * Works against any EIP-1193 provider — Coinbase Wallet SDK,
   * MetaMask, WalletConnect, Privy embedded provider, etc. The
   * function name reflects the Phase 4 Task 3 spec ("Coinbase Wallet
   * SDK primary"), but the body is provider-agnostic.
   */
  async connectCoinbaseWallet(
    provider: EthereumProvider,
    opts: ConnectCoinbaseWalletOptions,
  ): Promise<ConnectCoinbaseWalletResult> {
    if (!provider || typeof provider.request !== 'function') {
      throw new Error(
        'connectCoinbaseWallet requires an EIP-1193 provider with a request() method',
      );
    }

    // Step 1 — wallet address
    const accounts = (await provider.request({
      method: 'eth_requestAccounts',
    })) as string[];
    if (!Array.isArray(accounts) || accounts.length === 0) {
      throw new WalletAuthError(
        'no_accounts',
        'eth_requestAccounts returned no addresses',
        0,
      );
    }
    const address = accounts[0];

    // Step 2 — nonce
    const nonceResp = await this.requestNonce(opts.chainId);
    const chainId = opts.chainId ?? nonceResp.chain_id;

    // Step 3 — SIWE message
    const siweMessage = buildSiweMessage({
      address,
      nonce: nonceResp.nonce,
      domain: opts.domain,
      chainId,
      uri: opts.uri,
      version: opts.version,
      statement: opts.statement,
      issuedAt: opts.issuedAt,
    });

    // Step 4 — SIWE signature
    const siweSignature = (await provider.request({
      method: 'personal_sign',
      params: [siweMessage, address],
    })) as string;

    // Step 5 — verify + receive binding-attestation message
    const verified = await this.verifySiwe(siweMessage, siweSignature);

    // Step 6 — binding signature
    const bindingSignature = (await provider.request({
      method: 'personal_sign',
      params: [verified.binding_message, address],
    })) as string;

    // Step 7 — bind
    const binding = await this.bindWallet({
      wallet_address: verified.address,
      node_id_hex: verified.node_id_hex,
      signature: bindingSignature,
      issued_at: verified.binding_issued_at,
    });

    return {
      address: verified.address,
      nodeIdHex: verified.node_id_hex,
      isNewUser: verified.is_new_user,
      binding,
    };
  }

  // ─── Internals ─────────────────────────────────────────────────────

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await this.fetchFn(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw await this.toError(res);
    }
    return (await res.json()) as T;
  }

  private async toError(res: Response): Promise<WalletAuthError> {
    let bodyText = '';
    try {
      bodyText = await res.text();
    } catch {
      bodyText = '';
    }
    let parsed: { detail?: { error?: string; message?: string } } = {};
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      // Non-JSON body — fall through.
    }
    const code = parsed.detail?.error ?? 'http_error';
    const message =
      parsed.detail?.message ?? bodyText ?? `HTTP ${res.status}`;
    if (code === 'http_error' && res.status >= 500) {
      return new WalletAuthTransportError(message, res.status);
    }
    return new WalletAuthError(code, message, res.status);
  }
}
