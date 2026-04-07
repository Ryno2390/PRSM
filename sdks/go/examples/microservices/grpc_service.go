// PRSM Go SDK - gRPC Microservice Example
// This example demonstrates how to build a production-ready gRPC microservice
// powered by PRSM for AI inference in distributed systems.

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/PRSM-AI/prsm-go-sdk/client"
	"github.com/PRSM-AI/prsm-go-sdk/compute"
)

// AIServiceServer implements the gRPC AI service
type AIServiceServer struct {
	prsmClient *client.Client
	UnimplementedAIServiceServer
}

// InferenceRequest represents a request for AI inference
type InferenceRequest struct {
	Prompt      string            `json:"prompt"`
	Model       string            `json:"model,omitempty"`
	MaxTokens   int32             `json:"max_tokens,omitempty"`
	Temperature float32           `json:"temperature,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// InferenceResponse represents the response from AI inference
type InferenceResponse struct {
	Content   string                 `json:"content"`
	Model     string                 `json:"model"`
	Usage     *TokenUsage            `json:"usage"`
	Cost      float64                `json:"cost"`
	RequestId string                 `json:"request_id"`
	Timestamp *timestamppb.Timestamp `json:"timestamp"`
}

// TokenUsage represents token consumption
type TokenUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// BatchInferenceRequest represents a batch inference request
type BatchInferenceRequest struct {
	Requests []*InferenceRequest `json:"requests"`
}

// BatchInferenceResponse represents a batch inference response
type BatchInferenceResponse struct {
	Responses []*InferenceResponse `json:"responses"`
	TotalCost float64              `json:"total_cost"`
	RequestId string               `json:"request_id"`
}

// NewAIServiceServer creates a new AI service server
func NewAIServiceServer(apiKey string) (*AIServiceServer, error) {
	prsmClient := client.New(apiKey)

	return &AIServiceServer{
		prsmClient: prsmClient,
	}, nil
}

// Infer performs single AI inference
func (s *AIServiceServer) Infer(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	// Validate request
	if req.Prompt == "" {
		return nil, status.Error(codes.InvalidArgument, "prompt is required")
	}

	if len(req.Prompt) > 100000 {
		return nil, status.Error(codes.InvalidArgument, "prompt too long (max 100,000 characters)")
	}

	// Extract request metadata
	requestID := extractRequestID(ctx)
	
	log.Printf("[%s] Processing inference request: model=%s, prompt_length=%d", 
		requestID, req.Model, len(req.Prompt))

	// Set default values
	model := req.Model
	if model == "" {
		model = "nwtn"
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 500
	}

	temperature := req.Temperature
	if temperature == 0 {
		temperature = 0.7
	}

	// Execute compute job via PRSM SDK
	jobReq := compute.JobRequest{
		Prompt:      req.Prompt,
		Model:       model,
		MaxTokens:   int(maxTokens),
		Temperature: float64(temperature),
	}

	// Submit job
	jobResp, err := s.prsmClient.Compute.SubmitJob(ctx, jobReq)
	if err != nil {
		log.Printf("[%s] Job submission failed: %v", requestID, err)
		return nil, status.Error(codes.Internal, "job submission failed")
	}

	// Wait for completion
	result, err := s.prsmClient.Compute.WaitForCompletion(ctx, jobResp.JobID, 30*time.Second)
	if err != nil {
		log.Printf("[%s] Inference failed: %v", requestID, err)
		return nil, status.Error(codes.Internal, "inference failed")
	}

	log.Printf("[%s] Inference completed: tokens=%d, cost=%.4f FTNS", 
		requestID, result.TokenUsage["total"], result.FTNSCost)

	// Build response
	response := &InferenceResponse{
		Content:   result.Content,
		Model:     result.Model,
		Cost:      result.FTNSCost,
		RequestId: requestID,
		Timestamp: timestamppb.Now(),
		Usage: &TokenUsage{
			PromptTokens:     int32(result.TokenUsage["prompt"]),
			CompletionTokens: int32(result.TokenUsage["completion"]),
			TotalTokens:      int32(result.TokenUsage["total"]),
		},
	}

	return response, nil
}

// extractRequestID extracts request ID from context metadata
func extractRequestID(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return fmt.Sprintf("req-%d", time.Now().UnixNano())
	}
	
	if ids := md.Get("x-request-id"); len(ids) > 0 {
		return ids[0]
	}
	
	return fmt.Sprintf("req-%d", time.Now().UnixNano())
}

func main() {
	// Get API key from environment
	apiKey := os.Getenv("PRSM_API_KEY")
	if apiKey == "" {
		log.Fatal("PRSM_API_KEY environment variable is required")
	}

	// Create gRPC server
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	// Create AI service server
	aiServer, err := NewAIServiceServer(apiKey)
	if err != nil {
		log.Fatalf("Failed to create AI server: %v", err)
	}

	// Create gRPC server with interceptors
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor),
		grpc.StreamInterceptor(streamLoggingInterceptor),
	)

	// Register service
	RegisterAIServiceServer(grpcServer, aiServer)

	// Start server in goroutine
	go func() {
		log.Printf("Starting gRPC server on %s", lis.Addr())
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down gRPC server...")
	grpcServer.GracefulStop()
}

// loggingInterceptor logs all unary requests
func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	
	resp, err := handler(ctx, req)
	
	log.Printf("[%s] %s %v %s", 
		info.FullMethod,
		time.Since(start),
		status.Code(err),
		getClientIP(ctx),
	)
	
	return resp, err
}

// streamLoggingInterceptor logs all streaming requests
func streamLoggingInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	
	err := handler(srv, ss)
	
	log.Printf("[%s] %v %s", 
		info.FullMethod,
		time.Since(start),
		status.Code(err),
	)
	
	return err
}

// getClientIP extracts client IP from context
func getClientIP(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "unknown"
	}
	
	if ips := md.Get("x-forwarded-for"); len(ips) > 0 {
		return ips[0]
	}
	
	return "unknown"
}
