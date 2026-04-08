package internal

import (
	libp2p "github.com/libp2p/go-libp2p"
)

// NATOptions returns a slice of libp2p options that enable NAT traversal.
// enableRelay adds relay dialing/listening support.
// enableNAT adds AutoNATv2 and hole-punching.
// NATPortMap (UPnP/NAT-PMP) is always included.
func NATOptions(enableRelay, enableNAT bool) []libp2p.Option {
	opts := []libp2p.Option{
		libp2p.NATPortMap(),
	}
	if enableNAT {
		opts = append(opts,
			libp2p.EnableAutoNATv2(),
			libp2p.EnableHolePunching(),
		)
	}
	if enableRelay {
		opts = append(opts, libp2p.EnableRelay())
	}
	return opts
}
