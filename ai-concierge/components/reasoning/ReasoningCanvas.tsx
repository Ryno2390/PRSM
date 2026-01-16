import React, { useMemo } from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  Background, 
  Controls, 
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';

// Define the shape of the raw trace step based on NeuroSymbolicOrchestrator output
export interface TraceStep {
  t: number; // timestamp
  a: string; // action
  c: string; // content
  m: any;    // metadata
  p?: string;// provenance
  v?: string;// version
  s?: number;// surprise
}

interface ReasoningCanvasProps {
  trace: TraceStep[];
}

const ReasoningCanvas: React.FC<ReasoningCanvasProps> = ({ trace }) => {
  const { nodes, edges } = useMemo(() => {
    if (!trace || trace.length === 0) return { nodes: [], edges: [] };

    const nodes: Node[] = [];
    const edges: Edge[] = [];
    
    trace.forEach((step, index) => {
      // Determine color based on action/type
      let borderColor = 'border-gray-300';
      let bgColor = 'bg-white';
      
      if (step.a.includes('SECURITY')) {
        borderColor = 'border-red-500';
        bgColor = 'bg-red-50';
      } else if (step.a.includes('VERIFICATION')) {
        borderColor = 'border-green-500';
        bgColor = 'bg-green-50';
      } else if (step.a.includes('STRATEGY')) {
        borderColor = 'border-purple-500';
        bgColor = 'bg-purple-50';
      }

      // Create Node
      nodes.push({
        id: `step-${index}`,
        position: { x: 250, y: index * 180 + 50 },
        data: { 
          label: (
            <div className={`p-3 border-l-4 rounded shadow-sm w-[250px] text-left ${borderColor} ${bgColor}`}>
              <div className="flex justify-between items-center mb-1">
                <span className="font-bold text-xs uppercase text-gray-700">{step.a}</span>
                <span className="text-[10px] text-gray-400">{step.t.toFixed(2)}s</span>
              </div>
              <div className="text-xs text-gray-600 line-clamp-3 mb-2" title={step.c}>
                {step.c}
              </div>
              {step.s !== undefined && (
                 <div className="flex items-center mt-1 pt-1 border-t border-gray-100">
                   <span className="text-[10px] font-medium text-blue-600">
                     Surprise: {(step.s * 100).toFixed(0)}%
                   </span>
                 </div>
              )}
            </div>
          ) 
        },
        type: 'default',
        style: { 
          background: 'transparent', 
          border: 'none', 
          width: 250 
        }
      });

      // Create Edge to previous node
      if (index > 0) {
        edges.push({
          id: `edge-${index-1}-${index}`,
          source: `step-${index-1}`,
          target: `step-${index}`,
          type: 'smoothstep',
          markerEnd: { type: MarkerType.ArrowClosed },
          animated: true,
          style: { stroke: '#9ca3af' },
          label: index === trace.length - 1 ? 'conclusion' : 'next_step',
          labelStyle: { fill: '#9ca3af', fontSize: 10 },
        });
      }
    });

    return { nodes, edges };
  }, [trace]);

  return (
    <div className="h-[600px] w-full border rounded-lg bg-slate-50 overflow-hidden relative">
      <div className="absolute top-2 left-2 z-10 bg-white/80 backdrop-blur px-2 py-1 rounded text-xs font-mono text-gray-500">
        Trace Visualization
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        attributionPosition="bottom-right"
        defaultEdgeOptions={{ type: 'smoothstep' }}
      >
        <Background color="#e2e8f0" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default ReasoningCanvas;
