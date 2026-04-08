package internal

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	libp2phost "github.com/libp2p/go-libp2p/core/host"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
)

// PubSubManager wraps a GossipSub instance and manages topic subscriptions,
// routing incoming messages to the UDS data plane.
type PubSubManager struct {
	ps     *pubsub.PubSub
	topics map[string]*pubsub.Topic
	subs   map[string]*pubsub.Subscription
	uds    *UDSWriter
	host   libp2phost.Host
	mu     sync.RWMutex
	ctx    context.Context
}

// NewPubSubManager creates a GossipSub instance attached to the given host.
func NewPubSubManager(ctx context.Context, host libp2phost.Host, uds *UDSWriter) (*PubSubManager, error) {
	ps, err := pubsub.NewGossipSub(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("failed to create GossipSub: %w", err)
	}

	return &PubSubManager{
		ps:     ps,
		topics: make(map[string]*pubsub.Topic),
		subs:   make(map[string]*pubsub.Subscription),
		uds:    uds,
		host:   host,
		ctx:    ctx,
	}, nil
}

// Subscribe joins a topic and starts routing incoming messages to the UDS writer.
// Idempotent: if already subscribed, returns nil immediately.
func (m *PubSubManager) Subscribe(topicName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, alreadySubscribed := m.subs[topicName]; alreadySubscribed {
		return nil
	}

	topic, err := m.ps.Join(topicName)
	if err != nil {
		return fmt.Errorf("failed to join topic %q: %w", topicName, err)
	}

	sub, err := topic.Subscribe()
	if err != nil {
		_ = topic.Close()
		return fmt.Errorf("failed to subscribe to topic %q: %w", topicName, err)
	}

	m.topics[topicName] = topic
	m.subs[topicName] = sub

	go m.readLoop(topicName, sub)
	return nil
}

// readLoop reads messages from a subscription and forwards them to the UDS writer.
func (m *PubSubManager) readLoop(topicName string, sub *pubsub.Subscription) {
	for {
		msg, err := sub.Next(m.ctx)
		if err != nil {
			// Context cancelled or subscription cancelled — exit silently.
			return
		}

		// Filter out messages from self.
		if msg.ReceivedFrom == m.host.ID() {
			continue
		}

		if m.uds == nil {
			continue
		}

		udsMsg := UDSMessage{
			MsgType:     "gossip",
			TopicOrPeer: topicName,
			Data:        string(msg.Data),
			SenderID:    msg.ReceivedFrom.String(),
		}
		if err := m.uds.Write(udsMsg); err != nil {
			log.Printf("PubSubManager: failed to write gossip msg to UDS: %v", err)
		}
	}
}

// Unsubscribe cancels the subscription, closes the topic, and removes them from the maps.
func (m *PubSubManager) Unsubscribe(topicName string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if sub, ok := m.subs[topicName]; ok {
		sub.Cancel()
		delete(m.subs, topicName)
	}

	if topic, ok := m.topics[topicName]; ok {
		_ = topic.Close()
		delete(m.topics, topicName)
	}
}

// Publish sends data to the named topic, joining it first if not already joined.
func (m *PubSubManager) Publish(topicName string, data []byte) error {
	m.mu.Lock()
	topic, ok := m.topics[topicName]
	if !ok {
		var err error
		topic, err = m.ps.Join(topicName)
		if err != nil {
			m.mu.Unlock()
			return fmt.Errorf("failed to join topic %q for publish: %w", topicName, err)
		}
		m.topics[topicName] = topic
	}
	m.mu.Unlock()

	return topic.Publish(m.ctx, data)
}

// TopicPeers returns the number of peers known in the named topic.
func (m *PubSubManager) TopicPeers(topicName string) int {
	m.mu.RLock()
	topic, ok := m.topics[topicName]
	m.mu.RUnlock()
	if !ok {
		return 0
	}
	return len(topic.ListPeers())
}

// Stats returns a JSON map of topic name → peer count.
func (m *PubSubManager) Stats() string {
	m.mu.RLock()
	counts := make(map[string]int, len(m.topics))
	for name, topic := range m.topics {
		counts[name] = len(topic.ListPeers())
	}
	m.mu.RUnlock()

	out, err := json.Marshal(counts)
	if err != nil {
		return "{}"
	}
	return string(out)
}
