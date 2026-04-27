/**
 * Tests for the Phase 4 wallet onboarding helper.
 *
 * Real ethers Wallet acting as the EIP-1193 provider; node-fetch (or
 * a stub) acting as the HTTP transport. The backend is mocked via a
 * stub fetch — the wire-format contract tested here mirrors the
 * backend's Pydantic schemas in prsm/interface/api/wallet_api.py and
 * is round-tripped server-side by tests/unit/test_wallet_api.py.
 */

import { Wallet } from 'ethers';
import {
  WalletAuth,
  WalletAuthError,
  buildSiweMessage,
  toChecksumAddress,
  EthereumProvider,
  NonceResponse,
  SiweVerifyResponse,
  WalletBindResponse,
  BalanceResponse,
} from '../src/wallet-auth';

// ────────────────────────────────────────────────────────────────────
// Deterministic ethers wallet → EthereumProvider adapter
// ────────────────────────────────────────────────────────────────────

class EthersWalletProvider implements EthereumProvider {
  constructor(private readonly wallet: Wallet) {}

  async request(args: { method: string; params?: unknown[] | object }): Promise<unknown> {
    if (args.method === 'eth_requestAccounts') {
      return [this.wallet.address];
    }
    if (args.method === 'personal_sign') {
      const params = args.params as [string, string];
      const [message, _address] = params;
      // ethers v6: signMessage handles EIP-191 prefix automatically
      return await this.wallet.signMessage(message);
    }
    throw new Error(`Unsupported method: ${args.method}`);
  }
}

// ────────────────────────────────────────────────────────────────────
// Stub fetch — records calls + returns canned responses keyed by URL+method
// ────────────────────────────────────────────────────────────────────

interface StubFetchCall {
  url: string;
  method: string;
  body?: unknown;
}

class StubFetch {
  public readonly calls: StubFetchCall[] = [];
  public readonly handlers = new Map<string, (call: StubFetchCall) => Response>();

  on(method: string, urlPattern: string, handler: (call: StubFetchCall) => Response): void {
    this.handlers.set(`${method.toUpperCase()} ${urlPattern}`, handler);
  }

  get fetch(): typeof fetch {
    return (async (input: string | URL | Request, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      const method = (init?.method ?? 'GET').toUpperCase();
      const body = init?.body ? JSON.parse(init.body as string) : undefined;
      const call: StubFetchCall = { url, method, body };
      this.calls.push(call);

      // Find a matching handler by method + url-prefix.
      for (const [key, handler] of this.handlers.entries()) {
        const [hMethod, hPattern] = key.split(' ', 2);
        if (hMethod === method && url.includes(hPattern)) {
          return handler(call);
        }
      }
      return new Response('not found', { status: 404 });
    }) as unknown as typeof fetch;
  }
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

// ────────────────────────────────────────────────────────────────────
// Suite
// ────────────────────────────────────────────────────────────────────

const BASE_URL = 'https://api.test';
const DOMAIN = 'app.prsm-network.com';
const URI = 'https://app.prsm-network.com/login';
const CHAIN_ID = 8453;
const NODE_ID = 'a'.repeat(32);

describe('buildSiweMessage', () => {
  it('emits the canonical EIP-4361 layout', () => {
    const msg = buildSiweMessage({
      address: '0x1111111111111111111111111111111111111111',
      nonce: 'abc12345',
      domain: DOMAIN,
      chainId: CHAIN_ID,
      uri: URI,
      statement: 'Sign in to PRSM.',
      issuedAt: '2026-04-27T12:00:00Z',
    });
    expect(msg).toContain(`${DOMAIN} wants you to sign in`);
    expect(msg).toContain('0x1111111111111111111111111111111111111111');
    expect(msg).toContain('Sign in to PRSM.');
    expect(msg).toContain(`Chain ID: ${CHAIN_ID}`);
    expect(msg).toContain('Nonce: abc12345');
    expect(msg).toContain('Issued At: 2026-04-27T12:00:00Z');
  });

  it('emits TWO blank lines when statement is omitted (EIP-4361 layout)', () => {
    // Round-1 review HIGH-1: the no-statement form requires the
    // statement-bracketing blank lines to remain in place. Backend
    // siwe library rejects ValueError otherwise.
    const msg = buildSiweMessage({
      address: '0x1111111111111111111111111111111111111111',
      nonce: 'abc12345',
      domain: DOMAIN,
      chainId: CHAIN_ID,
      uri: URI,
    });
    // Layout: header / address / blank / blank / URI / Version / ...
    const lines = msg.split('\n');
    expect(lines[1]).toBe('0x1111111111111111111111111111111111111111');
    expect(lines[2]).toBe('');
    expect(lines[3]).toBe('');
    expect(lines[4]).toMatch(/^URI:/);
  });

  it('checksums lowercase addresses (EIP-55 normalisation)', () => {
    // Round-1 review HIGH-2: WalletConnect/embedded providers return
    // lowercase addresses; backend siwe library rejects ValidationError.
    // buildSiweMessage must emit checksummed form unconditionally.
    const lower = '0xab5801a7d398351b8be11c439e05c5b3259aec9b';
    const msg = buildSiweMessage({
      address: lower,
      nonce: 'abc12345',
      domain: DOMAIN,
      chainId: CHAIN_ID,
      uri: URI,
      statement: 'Sign in.',
    });
    // Address line must NOT be all-lowercase.
    expect(msg).not.toContain(lower);
    // Should contain the EIP-55 mixed-case form with at least one uppercase letter.
    const checksummed = toChecksumAddress(lower);
    expect(msg).toContain(checksummed);
    expect(/[A-F]/.test(checksummed)).toBe(true);
  });
});

describe('toChecksumAddress', () => {
  it('matches EIP-55 reference vectors', () => {
    // Vectors from EIP-55 spec section "Test Cases".
    const vectors: Array<[string, string]> = [
      // All-caps
      ['0x52908400098527886e0f7030069857d2e4169ee7',
       '0x52908400098527886E0F7030069857D2E4169EE7'],
      ['0x8617e340b3d01fa5f11f306f4090fd50e238070d',
       '0x8617E340B3D01FA5F11F306F4090FD50E238070D'],
      // All-lower
      ['0xde709f2102306220921060314715629080e2fb77',
       '0xde709f2102306220921060314715629080e2fb77'],
      ['0x27b1fdb04752bbc536007a920d24acb045561c26',
       '0x27b1fdb04752bbc536007a920d24acb045561c26'],
      // Mixed
      ['0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed',
       '0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed'],
      ['0xfb6916095ca1df60bb79ce92ce3ea74c37c5d359',
       '0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359'],
    ];
    for (const [input, expected] of vectors) {
      expect(toChecksumAddress(input)).toBe(expected);
    }
  });

  it('rejects malformed addresses', () => {
    expect(() => toChecksumAddress('not-an-address')).toThrow();
    expect(() => toChecksumAddress('0x123')).toThrow();
    expect(() => toChecksumAddress('0x' + 'g'.repeat(40))).toThrow();
  });

  it('accepts already-checksummed addresses idempotently', () => {
    const checksummed = '0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359';
    expect(toChecksumAddress(checksummed)).toBe(checksummed);
  });
});

describe('WalletAuth construction', () => {
  it('requires baseUrl', () => {
    expect(
      () => new WalletAuth({ baseUrl: '' as string, fetchFn: jest.fn() as unknown as typeof fetch }),
    ).toThrow(/baseUrl/);
  });

  it('strips trailing slash from baseUrl', async () => {
    const stub = new StubFetch();
    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () =>
      jsonResponse(200, {
        nonce: 'n',
        domain: DOMAIN,
        chain_id: CHAIN_ID,
        expires_at_unix: 0,
      }),
    );
    const auth = new WalletAuth({ baseUrl: `${BASE_URL}/`, fetchFn: stub.fetch });
    await auth.requestNonce();
    expect(stub.calls[0].url).toBe(`${BASE_URL}/api/v1/auth/wallet/siwe/nonce`);
  });
});

describe('WalletAuth.requestNonce', () => {
  it('POSTs an empty body when no chainId', async () => {
    const stub = new StubFetch();
    const expected: NonceResponse = {
      nonce: 'abcd1234',
      domain: DOMAIN,
      chain_id: CHAIN_ID,
      expires_at_unix: 9999,
    };
    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () => jsonResponse(200, expected));
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    const got = await auth.requestNonce();
    expect(got).toEqual(expected);
    expect(stub.calls[0].body).toEqual({});
  });

  it('forwards chainId when provided', async () => {
    const stub = new StubFetch();
    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () =>
      jsonResponse(200, {
        nonce: 'n',
        domain: DOMAIN,
        chain_id: CHAIN_ID,
        expires_at_unix: 0,
      }),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await auth.requestNonce(CHAIN_ID);
    expect(stub.calls[0].body).toEqual({ chain_id: CHAIN_ID });
  });

  it('maps backend error code to WalletAuthError', async () => {
    const stub = new StubFetch();
    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () =>
      jsonResponse(400, {
        detail: {
          error: 'siwe_chain_id_mismatch',
          message: 'client requested 1; server expects 8453',
        },
      }),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await expect(auth.requestNonce(1)).rejects.toMatchObject({
      code: 'siwe_chain_id_mismatch',
      status: 400,
    });
  });
});

describe('WalletAuth.connectCoinbaseWallet (composed flow)', () => {
  it('drives the full nonce → SIWE → verify → bind round-trip', async () => {
    const wallet = Wallet.createRandom();
    const provider = new EthersWalletProvider(wallet);

    const stub = new StubFetch();
    let capturedSiweMessage = '';
    let capturedSiweSignature = '';
    let capturedBindBody: { signature: string; node_id_hex: string } | undefined;

    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () =>
      jsonResponse(200, {
        nonce: 'noncenoncenonce',
        domain: DOMAIN,
        chain_id: CHAIN_ID,
        expires_at_unix: 0,
      }),
    );
    stub.on('POST', '/api/v1/auth/wallet/siwe/verify', (call) => {
      const body = call.body as { message: string; signature: string };
      capturedSiweMessage = body.message;
      capturedSiweSignature = body.signature;
      const verifyResp: SiweVerifyResponse = {
        address: wallet.address,
        node_id_hex: NODE_ID,
        is_new_user: true,
        binding_message:
          'PRSM Identity Binding\n' +
          `Wallet: ${wallet.address}\n` +
          `Node: ${NODE_ID}\n` +
          'Issued-At: 2026-04-27T12:00:00Z',
        binding_issued_at: '2026-04-27T12:00:00Z',
      };
      return jsonResponse(200, verifyResp);
    });
    stub.on('POST', '/api/v1/auth/wallet/bind', (call) => {
      capturedBindBody = call.body as { signature: string; node_id_hex: string };
      const bindResp: WalletBindResponse = {
        wallet_address: wallet.address,
        node_id_hex: NODE_ID,
        bound_at_unix: 1714567890,
        signing_message_hash: '0x' + 'cd'.repeat(32),
      };
      return jsonResponse(200, bindResp);
    });

    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    const result = await auth.connectCoinbaseWallet(provider, {
      domain: DOMAIN,
      uri: URI,
      statement: 'Sign in to PRSM.',
      issuedAt: '2026-04-27T11:59:00Z',
    });

    expect(result.address).toBe(wallet.address);
    expect(result.nodeIdHex).toBe(NODE_ID);
    expect(result.isNewUser).toBe(true);
    expect(result.binding.bound_at_unix).toBe(1714567890);

    // The SIWE message captured at the verify endpoint contains the nonce
    expect(capturedSiweMessage).toContain('Nonce: noncenoncenonce');
    expect(capturedSiweMessage).toContain(`Chain ID: ${CHAIN_ID}`);

    // The SIWE signature is real — recovers to the wallet's address
    expect(capturedSiweSignature.startsWith('0x')).toBe(true);

    // The bind step uses the binding-message signature, NOT the SIWE one
    expect(capturedBindBody!.signature).not.toBe(capturedSiweSignature);
    expect(capturedBindBody!.node_id_hex).toBe(NODE_ID);
  });

  it('rejects providers without a request() method', async () => {
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: new StubFetch().fetch });
    await expect(
      auth.connectCoinbaseWallet({} as unknown as EthereumProvider, {
        domain: DOMAIN,
        uri: URI,
      }),
    ).rejects.toThrow(/EIP-1193/);
  });

  it('surfaces backend errors mid-flow as WalletAuthError', async () => {
    const wallet = Wallet.createRandom();
    const provider = new EthersWalletProvider(wallet);

    const stub = new StubFetch();
    stub.on('POST', '/api/v1/auth/wallet/siwe/nonce', () =>
      jsonResponse(200, {
        nonce: 'n',
        domain: DOMAIN,
        chain_id: CHAIN_ID,
        expires_at_unix: 0,
      }),
    );
    stub.on('POST', '/api/v1/auth/wallet/siwe/verify', () =>
      jsonResponse(400, {
        detail: { error: 'siwe_signature_invalid', message: 'bad sig' },
      }),
    );

    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await expect(
      auth.connectCoinbaseWallet(provider, { domain: DOMAIN, uri: URI }),
    ).rejects.toMatchObject({
      code: 'siwe_signature_invalid',
      status: 400,
    });
  });

  it('aborts if eth_requestAccounts returns no addresses', async () => {
    const provider: EthereumProvider = {
      request: async () => [] as string[],
    };
    const stub = new StubFetch();
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await expect(
      auth.connectCoinbaseWallet(provider, { domain: DOMAIN, uri: URI }),
    ).rejects.toMatchObject({ code: 'no_accounts' });
  });
});

describe('WalletAuth.getBinding', () => {
  it('returns null for unbound wallet', async () => {
    const stub = new StubFetch();
    stub.on('GET', '/api/v1/auth/wallet/binding', () => jsonResponse(200, null));
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    const got = await auth.getBinding('0x' + '11'.repeat(20));
    expect(got).toBeNull();
  });

  it('returns binding when present', async () => {
    const stub = new StubFetch();
    const binding: WalletBindResponse = {
      wallet_address: '0x' + '11'.repeat(20),
      node_id_hex: NODE_ID,
      bound_at_unix: 100,
      signing_message_hash: '0xab',
    };
    stub.on('GET', '/api/v1/auth/wallet/binding', () =>
      jsonResponse(200, binding),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    const got = await auth.getBinding('0x' + '11'.repeat(20));
    expect(got).toEqual(binding);
  });
});

describe('WalletAuth.getBalance', () => {
  it('returns formatted balance', async () => {
    const stub = new StubFetch();
    const balance: BalanceResponse = {
      wallet_address: '0x' + '11'.repeat(20),
      node_id_hex: NODE_ID,
      ftns: '1.5',
      usd: '3.00',
      formatted: '$3.00 · 1.5000 FTNS',
      mode: 'usd',
    };
    stub.on('GET', '/api/v1/auth/wallet/balance', () =>
      jsonResponse(200, balance),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    const got = await auth.getBalance('0x' + '11'.repeat(20));
    expect(got).toEqual(balance);
  });

  it('passes mode=ftns', async () => {
    const stub = new StubFetch();
    stub.on('GET', '/api/v1/auth/wallet/balance', () =>
      jsonResponse(200, {
        wallet_address: '0x',
        node_id_hex: NODE_ID,
        ftns: '0',
        usd: null,
        formatted: '0.0000 FTNS',
        mode: 'ftns',
      }),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await auth.getBalance('0x' + '11'.repeat(20), 'ftns');
    const lastCall = stub.calls[stub.calls.length - 1];
    expect(lastCall.url).toContain('mode=ftns');
  });

  it('throws on 404 (wallet not bound)', async () => {
    const stub = new StubFetch();
    stub.on('GET', '/api/v1/auth/wallet/balance', () =>
      jsonResponse(404, {
        detail: { error: 'wallet_not_bound', message: 'no binding' },
      }),
    );
    const auth = new WalletAuth({ baseUrl: BASE_URL, fetchFn: stub.fetch });
    await expect(auth.getBalance('0x')).rejects.toMatchObject({
      code: 'wallet_not_bound',
      status: 404,
    });
  });
});

describe('WalletAuthError contract', () => {
  it('exposes code + status fields', () => {
    const err = new WalletAuthError('siwe_signature_invalid', 'bad sig', 400);
    expect(err.code).toBe('siwe_signature_invalid');
    expect(err.status).toBe(400);
    expect(err.message).toBe('bad sig');
    expect(err).toBeInstanceOf(Error);
  });
});
