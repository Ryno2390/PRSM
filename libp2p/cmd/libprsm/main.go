package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"encoding/json"
	"log"
	"strings"
	"unsafe"

	"github.com/Ryno2390/PRSM/libp2p/internal"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// PrsmStart creates a libp2p host with the given Ed25519 key and listen port,
// registers it, and returns a handle. ed25519Key must point to 64 raw bytes.
//
//export PrsmStart
func PrsmStart(ed25519Key *C.char, listenPort C.int, bootstrapAddrs *C.char, udsPath *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmStart: recovered from panic: %v", r)
		}
	}()

	keyBytes := C.GoBytes(unsafe.Pointer(ed25519Key), 64)

	// Build NAT options and create host.
	natOpts := internal.NATOptions(true, true)
	p2pHost, err := internal.CreateHost(keyBytes, int(listenPort), natOpts...)
	if err != nil {
		log.Printf("PrsmStart: failed to create host: %v", err)
		return C.int(0)
	}

	ctx, cancel := context.WithCancel(context.Background())

	var udsWriter *internal.UDSWriter
	udsPathStr := C.GoString(udsPath)
	if udsPathStr != "" {
		udsWriter, err = internal.NewUDSWriter(udsPathStr)
		if err != nil {
			log.Printf("PrsmStart: failed to create UDS writer at %q: %v", udsPathStr, err)
			// Non-fatal: continue without data plane.
			udsWriter = nil
		} else {
			go udsWriter.AcceptConnection()
		}
	}

	pubsubMgr, err := internal.NewPubSubManager(ctx, p2pHost, udsWriter)
	if err != nil {
		log.Printf("PrsmStart: failed to create GossipSub: %v", err)
		p2pHost.Close()
		if udsWriter != nil {
			udsWriter.Close()
		}
		cancel()
		return C.int(0)
	}

	streamMgr := internal.NewStreamManager(ctx, p2pHost, udsWriter)

	// Parse bootstrap addresses and initialise DHT (non-fatal on failure).
	bootstrapList := []string{}
	if bs := C.GoString(bootstrapAddrs); bs != "" {
		for _, a := range strings.Split(bs, ",") {
			a = strings.TrimSpace(a)
			if a != "" {
				bootstrapList = append(bootstrapList, a)
			}
		}
	}

	var dhtMgr *internal.DHTManager
	dhtMgr, err = internal.NewDHTManager(ctx, p2pHost, "auto", bootstrapList)
	if err != nil {
		log.Printf("PrsmStart: DHT initialisation failed (non-fatal): %v", err)
		dhtMgr = nil
	}

	host := &internal.Host{
		ListenPort: int(listenPort),
		PeerID:     p2pHost.ID().String(),
		Ctx:        ctx,
		Cancel:     cancel,
		P2PHost:    p2pHost,
		UDS:        udsWriter,
		PubSub:     pubsubMgr,
		Streams:    streamMgr,
		DHT:        dhtMgr,
	}

	handle := internal.Register(host)
	return C.int(handle)
}

// PrsmStop shuts down the libp2p host associated with handle and removes it
// from the registry. Returns 0 on success, -1 if the handle is not found.
//
//export PrsmStop
func PrsmStop(handle C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmStop: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		return C.int(-1)
	}

	if h.DHT != nil {
		if err := h.DHT.Close(); err != nil {
			log.Printf("PrsmStop: error closing DHT: %v", err)
		}
	}
	if h.UDS != nil {
		h.UDS.Close()
	}
	h.Cancel()
	if err := h.P2PHost.Close(); err != nil {
		log.Printf("PrsmStop: error closing host: %v", err)
	}
	internal.Remove(int(handle))
	return C.int(0)
}

// PrsmPeerCount returns the number of connected peers for the given handle.
// Returns -1 if the handle is not found.
//
//export PrsmPeerCount
func PrsmPeerCount(handle C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmPeerCount: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		return C.int(-1)
	}
	return C.int(len(h.P2PHost.Network().Peers()))
}

// PrsmConnect dials the peer at the given multiaddr string and returns the
// peer's ID string on success, or nil on failure. The returned string must be
// freed with PrsmFree.
//
//export PrsmConnect
func PrsmConnect(handle C.int, maddr *C.char) *C.char {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmConnect: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmConnect: handle %d not found", int(handle))
		return nil
	}

	maddrStr := C.GoString(maddr)
	ma, err := multiaddr.NewMultiaddr(maddrStr)
	if err != nil {
		log.Printf("PrsmConnect: invalid multiaddr %q: %v", maddrStr, err)
		return nil
	}

	peerInfo, err := peer.AddrInfoFromP2pAddr(ma)
	if err != nil {
		log.Printf("PrsmConnect: failed to extract peer info from %q: %v", maddrStr, err)
		return nil
	}

	if err := h.P2PHost.Connect(h.Ctx, *peerInfo); err != nil {
		log.Printf("PrsmConnect: failed to connect to %s: %v", peerInfo.ID, err)
		return nil
	}

	return C.CString(peerInfo.ID.String())
}

// PrsmFree releases memory allocated by C.CString.
//
//export PrsmFree
func PrsmFree(ptr *C.char) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmFree: recovered from panic: %v", r)
		}
	}()

	C.free(unsafe.Pointer(ptr))
}

// PrsmPublish publishes data to the named GossipSub topic.
// Returns 0 on success, -1 on error.
//
//export PrsmPublish
func PrsmPublish(handle C.int, topic *C.char, data *C.char, dataLen C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmPublish: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmPublish: handle %d not found", int(handle))
		return C.int(-1)
	}
	if h.PubSub == nil {
		log.Printf("PrsmPublish: PubSub not initialised on handle %d", int(handle))
		return C.int(-1)
	}

	if err := h.PubSub.Publish(C.GoString(topic), C.GoBytes(unsafe.Pointer(data), dataLen)); err != nil {
		log.Printf("PrsmPublish: %v", err)
		return C.int(-1)
	}
	return C.int(0)
}

// PrsmSubscribe subscribes to the named GossipSub topic.
// Returns 0 on success, -1 on error.
//
//export PrsmSubscribe
func PrsmSubscribe(handle C.int, topic *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmSubscribe: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmSubscribe: handle %d not found", int(handle))
		return C.int(-1)
	}
	if h.PubSub == nil {
		log.Printf("PrsmSubscribe: PubSub not initialised on handle %d", int(handle))
		return C.int(-1)
	}

	if err := h.PubSub.Subscribe(C.GoString(topic)); err != nil {
		log.Printf("PrsmSubscribe: %v", err)
		return C.int(-1)
	}
	return C.int(0)
}

// PrsmUnsubscribe unsubscribes from the named GossipSub topic.
// Always returns 0.
//
//export PrsmUnsubscribe
func PrsmUnsubscribe(handle C.int, topic *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmUnsubscribe: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmUnsubscribe: handle %d not found", int(handle))
		return C.int(0)
	}
	if h.PubSub != nil {
		h.PubSub.Unsubscribe(C.GoString(topic))
	}
	return C.int(0)
}

// PrsmSend sends data directly to the named peer using the /prsm/direct/1.0.0 protocol.
// proto is reserved for future use (pass empty string or a protocol hint).
// Returns 0 on success, -1 on error.
//
//export PrsmSend
func PrsmSend(handle C.int, peerIDStr *C.char, proto *C.char, data *C.char, dataLen C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmSend: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmSend: handle %d not found", int(handle))
		return C.int(-1)
	}
	if h.Streams == nil {
		log.Printf("PrsmSend: Streams not initialised on handle %d", int(handle))
		return C.int(-1)
	}

	pid, err := peer.Decode(C.GoString(peerIDStr))
	if err != nil {
		log.Printf("PrsmSend: invalid peer ID %q: %v", C.GoString(peerIDStr), err)
		return C.int(-1)
	}

	if err := h.Streams.Send(pid, C.GoBytes(unsafe.Pointer(data), dataLen)); err != nil {
		log.Printf("PrsmSend: %v", err)
		return C.int(-1)
	}
	return C.int(0)
}

// PrsmDHTProvide announces to the DHT that this node provides the content
// identified by key. Returns 0 on success, -1 on error.
//
//export PrsmDHTProvide
func PrsmDHTProvide(handle C.int, key *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmDHTProvide: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmDHTProvide: handle %d not found", int(handle))
		return C.int(-1)
	}
	if h.DHT == nil {
		log.Printf("PrsmDHTProvide: DHT not initialised on handle %d", int(handle))
		return C.int(-1)
	}
	if err := h.DHT.Provide(C.GoString(key)); err != nil {
		log.Printf("PrsmDHTProvide: %v", err)
		return C.int(-1)
	}
	return C.int(0)
}

// peerAddrInfoJSON is the JSON representation used by PrsmDHTFindProviders and PrsmPeerList.
type peerAddrInfoJSON struct {
	PeerID string   `json:"peer_id"`
	Addrs  []string `json:"addrs"`
}

// PrsmDHTFindProviders searches the DHT for up to limit peers that provide key.
// Returns a JSON array of {"peer_id":"…","addrs":["…"]} objects as a C string.
// The caller must free the result with PrsmFree. Returns nil on error.
//
//export PrsmDHTFindProviders
func PrsmDHTFindProviders(handle C.int, key *C.char, limit C.int) *C.char {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmDHTFindProviders: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmDHTFindProviders: handle %d not found", int(handle))
		return nil
	}
	if h.DHT == nil {
		log.Printf("PrsmDHTFindProviders: DHT not initialised on handle %d", int(handle))
		return nil
	}

	providers, err := h.DHT.FindProviders(C.GoString(key), int(limit))
	if err != nil {
		log.Printf("PrsmDHTFindProviders: %v", err)
		return nil
	}

	items := make([]peerAddrInfoJSON, 0, len(providers))
	for _, pi := range providers {
		addrs := make([]string, 0, len(pi.Addrs))
		for _, a := range pi.Addrs {
			addrs = append(addrs, a.String())
		}
		items = append(items, peerAddrInfoJSON{PeerID: pi.ID.String(), Addrs: addrs})
	}

	b, err := json.Marshal(items)
	if err != nil {
		log.Printf("PrsmDHTFindProviders: JSON marshal error: %v", err)
		return nil
	}
	return C.CString(string(b))
}

// PrsmGetNATStatus returns "public" if the host has at least one non-localhost
// address, or "private" otherwise. The caller must free the result with PrsmFree.
//
//export PrsmGetNATStatus
func PrsmGetNATStatus(handle C.int) *C.char {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmGetNATStatus: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmGetNATStatus: handle %d not found", int(handle))
		return C.CString("unknown")
	}

	for _, addr := range h.P2PHost.Addrs() {
		addrStr := addr.String()
		// Heuristic: if the address is not loopback it is considered public.
		if !strings.HasPrefix(addrStr, "/ip4/127.") &&
			!strings.HasPrefix(addrStr, "/ip6/::1") {
			return C.CString("public")
		}
	}
	return C.CString("private")
}

// PrsmPeerList returns a JSON array of {"peer_id":"…","addrs":["…"]} objects
// for all currently connected peers. The caller must free the result with PrsmFree.
// Returns nil on error or if the handle is not found.
//
//export PrsmPeerList
func PrsmPeerList(handle C.int) *C.char {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmPeerList: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmPeerList: handle %d not found", int(handle))
		return nil
	}

	peers := h.P2PHost.Network().Peers()
	items := make([]peerAddrInfoJSON, 0, len(peers))
	for _, pid := range peers {
		conns := h.P2PHost.Network().ConnsToPeer(pid)
		addrs := make([]string, 0, len(conns))
		for _, c := range conns {
			addrs = append(addrs, c.RemoteMultiaddr().String())
		}
		items = append(items, peerAddrInfoJSON{PeerID: pid.String(), Addrs: addrs})
	}

	b, err := json.Marshal(items)
	if err != nil {
		log.Printf("PrsmPeerList: JSON marshal error: %v", err)
		return nil
	}
	return C.CString(string(b))
}

// main is required for c-shared build mode.
func main() {}
