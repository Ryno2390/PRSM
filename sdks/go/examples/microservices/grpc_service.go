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

	"github.com/PRSM-AI/prsm-go-sdk/pkg/prsm"
)

// AIServiceServer implements the gRPC AI service
type AIServiceServer struct {
	prsmClient *prsm.Client
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

// TokenUsage represents token usage statistics
type TokenUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// StreamInferenceResponse represents a streaming response chunk
type StreamInferenceResponse struct {
	Content   string `json:"content"`
	Model     string `json:"model"`
	RequestId string `json:"request_id"`
	Done      bool   `json:"done"`
}

// BatchInferenceRequest represents a batch inference request
type BatchInferenceRequest struct {
	Requests []*InferenceRequest `json:"requests"`
	Model    string              `json:"model,omitempty"`
}

// BatchInferenceResponse represents a batch inference response
type BatchInferenceResponse struct {
	Responses []*InferenceResponse `json:"responses"`
	TotalCost float64              `json:"total_cost"`
	RequestId string               `json:"request_id"`
}

// NewAIServiceServer creates a new AI service server
func NewAIServiceServer(apiKey string) (*AIServiceServer, error) {
	client, err := prsm.NewClient(&prsm.Config{
		APIKey:     apiKey,
		BaseURL:    os.Getenv("PRSM_BASE_URL"),
		Timeout:    60 * time.Second,
		MaxRetries: 3,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create PRSM client: %w", err)
	}

	return &AIServiceServer{
		prsmClient: client,
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
		model = "gpt-4"
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 500
	}

	temperature := req.Temperature
	if temperature == 0 {
		temperature = 0.7
	}

	// Prepare PRSM request
	prsmReq := &prsm.InferenceRequest{
		Model:       model,
		Prompt:      req.Prompt,
		MaxTokens:   int(maxTokens),
		Temperature: float64(temperature),
	}

	// Execute inference with timeout
	inferCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	result, err := s.prsmClient.Infer(inferCtx, prsmReq)
	if err != nil {
		log.Printf("[%s] Inference failed: %v", requestID, err)
		
		// Convert PRSM errors to gRPC status codes
		switch {
		case prsm.IsBudgetExceededError(err):
			return nil, status.Error(codes.ResourceExhausted, "budget exceeded")
		case prsm.IsRateLimitError(err):
			return nil, status.Error(codes.ResourceExhausted, "rate limit exceeded")
		case prsm.IsAuthenticationError(err):
			return nil, status.Error(codes.Unauthenticated, "authentication failed")
		case prsm.IsValidationError(err):
			return nil, status.Error(codes.InvalidArgument, err.Error())
		default:
			return nil, status.Error(codes.Internal, "inference failed")
		}
	}

	log.Printf("[%s] Inference completed: tokens=%d, cost=$%.4f", 
		requestID, result.Usage.TotalTokens, result.Cost)

	// Build response
	response := &InferenceResponse{
		Content:   result.Content,
		Model:     result.Model,
		Cost:      result.Cost,
		RequestId: requestID,
		Timestamp: timestamppb.Now(),
		Usage: &TokenUsage{
			PromptTokens:     int32(result.Usage.PromptTokens),
			CompletionTokens: int32(result.Usage.CompletionTokens),
			TotalTokens:      int32(result.Usage.TotalTokens),
		},
	}

	return response, nil
}

// StreamInfer performs streaming AI inference
func (s *AIServiceServer) StreamInfer(req *InferenceRequest, stream AIService_StreamInferServer) error {
	// Validate request
	if req.Prompt == "" {
		return status.Error(codes.InvalidArgument, "prompt is required")
	}

	ctx := stream.Context()
	requestID := extractRequestID(ctx)
	
	log.Printf("[%s] Processing streaming inference request: model=%s", requestID, req.Model)

	// Set defaults
	model := req.Model
	if model == "" {
		model = "gpt-4"
	}

	// Prepare PRSM streaming request
	prsmReq := &prsm.StreamInferenceRequest{
		Model:       model,
		Prompt:      req.Prompt,
		MaxTokens:   int(req.MaxTokens),
		Temperature: float64(req.Temperature),
	}

	// Start streaming
	streamCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	resultChan, errChan := s.prsmClient.StreamInfer(streamCtx, prsmReq)

	// Process streaming chunks
	for {
		select {
		case chunk := <-resultChan:
			if chunk == nil {
				// Stream completed
				return stream.Send(&StreamInferenceResponse{
					RequestId: requestID,
					Done:      true,
				})
			}

			// Send chunk to client
			if err := stream.Send(&StreamInferenceResponse{
				Content:   chunk.Content,
				Model:     chunk.Model,
				RequestId: requestID,
				Done:      false,
			}); err != nil {
				log.Printf("[%s] Failed to send stream chunk: %v", requestID, err)
				return err
			}

		case err := <-errChan:
			if err != nil {
				log.Printf("[%s] Streaming error: %v", requestID, err)
				return status.Error(codes.Internal, "streaming failed")
			}

		case <-ctx.Done():
			log.Printf("[%s] Stream cancelled by client", requestID)
			return status.Error(codes.Cancelled, "stream cancelled")
		}
	}
}

// BatchInfer performs batch AI inference
func (s *AIServiceServer) BatchInfer(ctx context.Context, req *BatchInferenceRequest) (*BatchInferenceResponse, error) {
	if len(req.Requests) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one request is required")
	}

	if len(req.Requests) > 10 {
		return nil, status.Error(codes.InvalidArgument, "maximum 10 requests per batch")
	}

	requestID := extractRequestID(ctx)
	log.Printf("[%s] Processing batch inference: %d requests", requestID, len(req.Requests))

	// Process requests concurrently
	type result struct {
		response *InferenceResponse
		index    int
		err      error
	}

	resultChan := make(chan result, len(req.Requests))

	// Launch concurrent inferences
	for i, inferReq := range req.Requests {
		go func(index int, request *InferenceRequest) {
			resp, err := s.Infer(ctx, request)
			resultChan <- result{
				response: resp,
				index:    index,
				err:      err,
			}
		}(i, inferReq)
	}

	// Collect results
	responses := make([]*InferenceResponse, len(req.Requests))
	totalCost := 0.0
	var firstError error

	for i := 0; i < len(req.Requests); i++ {
		res := <-resultChan
		
		if res.err != nil && firstError == nil {
			firstError = res.err
		}
		
		if res.response != nil {
			responses[res.index] = res.response
			totalCost += res.response.Cost
		}
	}

	// Return error if any inference failed
	if firstError != nil {
		return nil, firstError
	}

	log.Printf("[%s] Batch inference completed: total_cost=$%.4f", requestID, totalCost)

	return &BatchInferenceResponse{
		Responses: responses,
		TotalCost: totalCost,
		RequestId: requestID,
	}, nil
}

// HealthCheck provides service health status
func (s *AIServiceServer) HealthCheck(ctx context.Context, req *HealthCheckRequest) (*HealthCheckResponse, error) {
	// Check PRSM client health
	if err := s.prsmClient.HealthCheck(ctx); err != nil {
		return &HealthCheckResponse{
			Status:  "unhealthy",
			Message: fmt.Sprintf("PRSM client unhealthy: %v", err),
		}, nil
	}

	return &HealthCheckResponse{
		Status:    "healthy",
		Message:   "All systems operational",
		Timestamp: timestamppb.Now(),
	}, nil
}

// Helper functions

func extractRequestID(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if values := md.Get("x-request-id"); len(values) > 0 {
			return values[0]
		}
	}
	
	// Generate new request ID if not provided
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

// Middleware for logging and monitoring
func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	requestID := extractRequestID(ctx)
	
	log.Printf("[%s] %s started", requestID, info.FullMethod)
	
	resp, err := handler(ctx, req)
	
	duration := time.Since(start)
	status := "success"
	if err != nil {
		status = "error"
	}
	
	log.Printf("[%s] %s completed: status=%s, duration=%v", 
		requestID, info.FullMethod, status, duration)
	
	return resp, err
}

// Stream logging interceptor
func streamLoggingInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	ctx := ss.Context()
	requestID := extractRequestID(ctx)
	
	log.Printf("[%s] %s stream started", requestID, info.FullMethod)
	
	err := handler(srv, ss)
	
	duration := time.Since(start)
	status := "success"
	if err != nil {
		status = "error"
	}
	
	log.Printf("[%s] %s stream completed: status=%s, duration=%v", 
		requestID, info.FullMethod, status, duration)
	
	return err
}

func main() {
	// Get configuration from environment
	apiKey := os.Getenv("PRSM_API_KEY")
	if apiKey == "" {
		log.Fatal("PRSM_API_KEY environment variable is required")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	// Create AI service server
	server, err := NewAIServiceServer(apiKey)
	if err != nil {
		log.Fatalf("Failed to create AI service server: %v", err)
	}

	// Create gRPC server with middleware
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor),
		grpc.StreamInterceptor(streamLoggingInterceptor),
	)

	// Register service
	RegisterAIServiceServer(grpcServer, server)

	// Setup listener
	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", port, err)
	}

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Received shutdown signal, stopping server...")
		grpcServer.GracefulStop()
	}()

	// Start server
	log.Printf("🚀 PRSM AI gRPC service starting on port %s", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}