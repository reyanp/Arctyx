/**
 * Workflow Panel Component
 * 
 * Handles the complete data generation workflow:
 * 1. Label data (optional)
 * 2. Create training config
 * 3. Train model
 * 4. Generate synthetic data
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { 
  createLabels,
  createTrainingConfig,
  trainModel,
  generateData,
  convertToParquet,
  runAgentGenerate,
  type LabelingFunction,
  type CreateConfigRequest,
  type TrainModelRequest,
  type GenerateDataRequest
} from "@/lib/api";
import { useDatasetState } from "@/hooks/use-dataset-state";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { 
  Sparkles, 
  Check, 
  Loader2, 
  AlertCircle,
  Tag,
  Settings,
  Brain,
  Database,
  ChevronRight
} from "lucide-react";

type WorkflowStep = 'idle' | 'labeling' | 'config' | 'training' | 'generating' | 'complete' | 'error';

export function WorkflowPanel() {
  const navigate = useNavigate();
  const { path: datasetPath } = useDatasetState();
  
  // Agent Mode state
  const [agentMode, setAgentMode] = useState(false);
  const [agentGoal, setAgentGoal] = useState('');
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentOutput, setAgentOutput] = useState<string | null>(null);
  
  // Workflow state
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('idle');
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // User configuration
  const [skipLabeling, setSkipLabeling] = useState(false); // Changed to false - always label/convert
  const [numSamples, setNumSamples] = useState(1000);
  const [epochs, setEpochs] = useState(50);
  const [modelType, setModelType] = useState<'tabular_cvae' | 'mixed_data_cvae' | 'tabular_vae_gmm' | 'tabular_ctgan'>('tabular_cvae');
  const [labelCondition, setLabelCondition] = useState<number>(1.0);
  
  // Labeling configuration (if needed)
  const [labelingFunctionsCode, setLabelingFunctionsCode] = useState(`def lf_high_capital_gain(x):
    """Label high income based on capital gains"""
    return 1 if x.get('capital-gain', 0) > 5000 else 0

def lf_educated(x):
    """Label high income based on education"""
    return 1 if x.get('education-num', 0) >= 13 else 0

def lf_professional(x):
    """Label high income based on occupation"""
    occupation = str(x.get('occupation', ''))
    return 1 if 'Prof-specialty' in occupation or 'Exec-managerial' in occupation else 0`);

  // Paths for workflow outputs
  const [labeledDataPath, setLabeledDataPath] = useState<string | null>(null);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [preprocessorPath, setPreprocessorPath] = useState<string | null>(null);
  const [syntheticDataPath, setSyntheticDataPath] = useState<string | null>(null);

  // Mutations
  const labelingMutation = useMutation({
    mutationFn: createLabels,
    onSuccess: (data) => {
      setLabeledDataPath(data.output_path);
      setProgress(25);
    },
  });

  const configMutation = useMutation({
    mutationFn: createTrainingConfig,
    onSuccess: (data) => {
      setConfigPath(data.config_path);
      setProgress(skipLabeling ? 25 : 50);
    },
  });

  const trainingMutation = useMutation({
    mutationFn: trainModel,
    onSuccess: (data) => {
      setModelPath(data.model_path);
      setPreprocessorPath(data.preprocessor_path);
      setProgress(skipLabeling ? 50 : 75);
    },
  });

  const generationMutation = useMutation({
    mutationFn: generateData,
    onSuccess: (data) => {
      setSyntheticDataPath(data.output_path);
      setProgress(100);
      setCurrentStep('complete');
      
      // Create generation record
      const generationRecord = {
        id: Date.now().toString(),
        syntheticDataPath: data.output_path,
        originalDataPath: datasetPath,
        numSamples: data.num_generated,
        modelPath,
        configPath,
        timestamp: Date.now(),
        skippedLabeling: skipLabeling, // Track if labeling was skipped
      };
      
      // Store as latest generation
      sessionStorage.setItem('latestGeneration', JSON.stringify(generationRecord));
      
      // Add to generation history
      const historyKey = 'generationHistory';
      const storedHistory = sessionStorage.getItem(historyKey);
      let history = [];
      
      if (storedHistory) {
        try {
          history = JSON.parse(storedHistory);
        } catch (error) {
          console.error('Failed to parse generation history:', error);
        }
      }
      
      // Add new generation to the beginning of the array
      history.unshift(generationRecord);
      
      // Keep only the last 10 generations
      if (history.length > 10) {
        history = history.slice(0, 10);
      }
      
      // Save updated history
      sessionStorage.setItem(historyKey, JSON.stringify(history));
    },
  });

  // Run complete workflow
  const runWorkflow = async () => {
    if (!datasetPath) {
      setErrorMessage('No dataset selected');
      return;
    }

    try {
      setErrorMessage(null);
      setProgress(0);

      let dataPathForTraining = datasetPath;

      if (skipLabeling) {
        // Skip labeling entirely - just convert format if needed
        // Training requires Parquet format, so check if we need to convert
        if (datasetPath.endsWith('.csv')) {
          setCurrentStep('labeling'); // Reuse labeling step for format conversion
          setProgress(10);
          
          const conversionResult = await convertToParquet({
            csv_path: datasetPath,
            output_path: `output_data/converted_${Date.now()}.parquet`,
          });
          
          dataPathForTraining = conversionResult.parquet_path;
          setProgress(25);
        } else {
          // Already Parquet, use as-is
          dataPathForTraining = datasetPath;
          setProgress(25);
        }
      } else {
        // Step 1: Labeling with custom functions
        setCurrentStep('labeling');
        setProgress(0);
        
        // Parse user-provided labeling functions
        const labelingFunctions = parseLabelingFunctions(labelingFunctionsCode);
        
        if (labelingFunctions.length === 0) {
          throw new Error('No valid labeling functions provided');
        }

        const labelingResult = await labelingMutation.mutateAsync({
          data_path: datasetPath,
          output_path: `output_data/labeled_${Date.now()}.parquet`,
          labeling_functions: labelingFunctions,
        });
        
        dataPathForTraining = labelingResult.output_path;
        setProgress(25);
      }

      // Step 2: Create training config
      setCurrentStep('config');
      const configRequest: CreateConfigRequest = {
        data_path: dataPathForTraining,
        output_dir: 'output_data/models/',
        model_type: modelType,
        training_params: {
          epochs: epochs,
          batch_size: 128,
          learning_rate: 0.001,
        },
      };
      
      const configResult = await configMutation.mutateAsync(configRequest);

      // Step 3: Train model
      setCurrentStep('training');
      const trainingRequest: TrainModelRequest = {
        config_path: configResult.config_path,
      };
      
      const trainingResult = await trainingMutation.mutateAsync(trainingRequest);

      // Step 4: Generate synthetic data
      setCurrentStep('generating');
      const generationRequest: GenerateDataRequest = {
        model_path: trainingResult.model_path,
        config_path: trainingResult.config_path,
        label: labelCondition,
        num_to_generate: numSamples,
        output_path: `output_data/synthetic_${Date.now()}.parquet`,
        output_format: 'parquet',
      };
      
      await generationMutation.mutateAsync(generationRequest);

    } catch (error) {
      setCurrentStep('error');
      setErrorMessage(error instanceof Error ? error.message : 'An unknown error occurred');
    }
  };

  // Parse labeling functions from code
  const parseLabelingFunctions = (code: string): LabelingFunction[] => {
    const functions: LabelingFunction[] = [];
    const functionRegex = /def\s+(\w+)\s*\([^)]*\):/g;
    let match;
    
    while ((match = functionRegex.exec(code)) !== null) {
      const functionName = match[1];
      const startIndex = match.index;
      let endIndex = code.indexOf('\ndef ', startIndex + 1);
      if (endIndex === -1) endIndex = code.length;
      
      const functionCode = code.substring(startIndex, endIndex).trim();
      functions.push({
        name: functionName,
        code: functionCode,
      });
    }
    
    return functions;
  };

  // Run agent workflow
  const runAgent = async () => {
    if (!agentGoal.trim()) {
      setErrorMessage('Please describe your goal');
      return;
    }

    try {
      setAgentRunning(true);
      setErrorMessage(null);
      setAgentOutput(null);

      const response = await runAgentGenerate({
        input_message: agentGoal,
        dataset_path: datasetPath || undefined,
      });

      if (response.error) {
        throw new Error(response.error);
      }

      setAgentOutput(response.output);
      
      // Store results if synthetic data was generated
      if (response.file_paths.synthetic_output_path && response.steps_completed.includes('generation')) {
        const generationRecord = {
          id: Date.now().toString(),
          syntheticDataPath: response.file_paths.synthetic_output_path,
          originalDataPath: datasetPath || '',
          numSamples: 0, // Unknown from agent
          modelPath: response.file_paths.model_path,
          configPath: response.file_paths.config_path,
          timestamp: Date.now(),
          skippedLabeling: !response.steps_completed.includes('labeling'),
          agentGenerated: true,
        };
        
        sessionStorage.setItem('latestGeneration', JSON.stringify(generationRecord));
        
        // Add to history
        const historyKey = 'generationHistory';
        const storedHistory = sessionStorage.getItem(historyKey);
        let history = [];
        
        if (storedHistory) {
          try {
            history = JSON.parse(storedHistory);
          } catch (error) {
            console.error('Failed to parse generation history:', error);
          }
        }
        
        history.unshift(generationRecord);
        
        if (history.length > 10) {
          history = history.slice(0, 10);
        }
        
        sessionStorage.setItem(historyKey, JSON.stringify(history));
        
        // Navigate to export after success
        setTimeout(() => {
          navigate('/export');
        }, 2000);
      }
      
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Agent workflow failed');
    } finally {
      setAgentRunning(false);
    }
  };

  const isRunning = currentStep !== 'idle' && currentStep !== 'complete' && currentStep !== 'error';
  const isComplete = currentStep === 'complete';
  const hasError = currentStep === 'error';

  return (
    <div className="relative">
      {/* Agent Mode Toggle Button */}
      <div className="absolute -top-10 right-0 z-10">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAgentMode(!agentMode)}
          className="rounded-full px-4 border-teal-500 text-teal-600 hover:bg-teal-50 hover:text-teal-700"
        >
          {agentMode ? 'Manual Mode' : 'Agent Mode'}
        </Button>
      </div>

      <CardModern>
        <CardModernHeader>
          <CardModernTitle>
            {agentMode ? 'Agent Mode' : 'Data Generation Workflow'}
          </CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          {agentMode ? (
            // AGENT MODE UI
            <div className="space-y-6 py-4">
              {!agentRunning && !agentOutput && (
                <>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Describe your goal and let Arctyx generate the workflow.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="agent-goal" className="text-sm font-medium">
                      Your Goal
                    </Label>
                    <Textarea
                      id="agent-goal"
                      value={agentGoal}
                      onChange={(e) => setAgentGoal(e.target.value)}
                      placeholder="Example: Generate 1000 synthetic samples of high-income individuals using weak supervision labeling..."
                      className="min-h-[200px] resize-none"
                    />
                  </div>

                  <div className="flex flex-col gap-3 pt-4">
                    <Button 
                      size="lg" 
                      className="w-full bg-teal-600 hover:bg-teal-700"
                      disabled={!agentGoal.trim() || !datasetPath}
                      onClick={runAgent}
                    >
                      <Sparkles className="w-4 h-4 mr-2" />
                      Run Agent
                    </Button>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setAgentMode(false)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Back to Manual Mode
                    </Button>
                  </div>

                  {!datasetPath && (
                    <Alert>
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        Please upload a dataset first to use Agent Mode.
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              )}

              {/* Agent Running State */}
              {agentRunning && (
                <div className="flex flex-col items-center justify-center py-8 space-y-4">
                  <img
                    src="/jensen-disc.png"
                    alt="Processing"
                    className="w-32 h-32 animate-slowspin"
                  />
                  <div className="text-center space-y-1">
                    <p className="font-medium text-sm">
                      🤖 Agent is orchestrating your workflow...
                    </p>
                    <p className="text-xs text-muted-foreground">
                      This may take a few minutes
                    </p>
                  </div>
                </div>
              )}

              {/* Agent Output */}
              {agentOutput && (
                <>
                  <Alert className="border-green-500/50 bg-green-500/10">
                    <Check className="h-4 w-4 text-green-500" />
                    <AlertDescription className="text-green-700 dark:text-green-400">
                      <div className="space-y-2">
                        <p className="font-medium">Agent workflow completed!</p>
                        <p className="text-xs whitespace-pre-wrap">{agentOutput}</p>
                      </div>
                    </AlertDescription>
                  </Alert>
                  
                  <div className="flex flex-col gap-3">
                    <Button 
                      size="lg" 
                      onClick={() => navigate('/export')}
                      className="w-full"
                    >
                      View Results
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                    
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setAgentOutput(null);
                        setAgentGoal('');
                      }}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Run Another Task
                    </Button>
                  </div>
                </>
              )}

              {/* Error State */}
              {errorMessage && !agentRunning && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{errorMessage}</AlertDescription>
                </Alert>
              )}
            </div>
          ) : (
            // NORMAL WORKFLOW UI
            <div className="space-y-6">
          {/* Workflow Progress */}
          {isRunning && (
            <div className="flex flex-col items-center justify-center py-8 space-y-4">
              {/* Spinning Jensen Disc */}
              <img
                src="/jensen-disc.png"
                alt="Processing"
                className="w-32 h-32 animate-slowspin"
              />
              <div className="text-center space-y-1">
                <p className="font-medium text-sm">
                  {currentStep === 'labeling' && (skipLabeling ? '📋  Converting data format...' : '🏷️  Labeling data...')}
                  {currentStep === 'config' && '⚙️  Creating configuration...'}
                  {currentStep === 'training' && '🧠  Training model...'}
                  {currentStep === 'generating' && '✨  Generating synthetic data...'}
                </p>
                <p className="text-xs text-muted-foreground">{progress}%</p>
              </div>
            </div>
          )}

          {/* Success State */}
          {isComplete && syntheticDataPath && (
            <Alert className="border-green-500/50 bg-green-500/10">
              <Check className="h-4 w-4 text-green-500" />
              <AlertDescription className="text-green-700 dark:text-green-400">
                <div className="space-y-2">
                  <p className="font-medium">Workflow completed successfully!</p>
                  <div className="text-xs space-y-1">
                    <p>Generated: {numSamples} synthetic samples</p>
                    <p className="font-mono truncate">{syntheticDataPath}</p>
                  </div>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Error State */}
          {hasError && errorMessage && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{errorMessage}</AlertDescription>
            </Alert>
          )}

          {/* Configuration Section */}
          {!isRunning && !isComplete && (
            <>
              {/* Labeling Option */}
              <div className="space-y-4 p-4 border rounded-lg bg-muted/20">
                <div className="flex items-center gap-2">
                  <Tag className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">Step 1: Data Labeling</h4>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="skip-labeling" 
                    checked={skipLabeling}
                    onCheckedChange={(checked) => setSkipLabeling(checked as boolean)}
                  />
                  <Label htmlFor="skip-labeling" className="text-sm cursor-pointer">
                    Skip labeling (use existing data columns as conditions)
                  </Label>
                </div>
                <p className="text-xs text-muted-foreground pl-6">
                  {skipLabeling 
                    ? "The model will use one of your existing columns as the condition variable. No Snorkel labeling will be performed."
                    : "Define custom labeling functions to add weak supervision labels"}
                </p>

                {!skipLabeling && (
                  <div className="space-y-2">
                    <Label htmlFor="labeling-functions" className="text-xs">
                      Labeling Functions (Python code)
                    </Label>
                    <Textarea
                      id="labeling-functions"
                      value={labelingFunctionsCode}
                      onChange={(e) => setLabelingFunctionsCode(e.target.value)}
                      placeholder="def lf_example(x):&#10;    return 1 if condition else 0"
                      className="font-mono text-xs h-32"
                    />
                    <p className="text-xs text-muted-foreground">
                      Define Python functions that return 1 (positive) or 0 (negative/abstain)
                    </p>
                  </div>
                )}
              </div>

              <Separator />

              {/* Training Configuration */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Settings className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">Step 2 & 3: Model Training</h4>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="model-type" className="text-sm">Model Type</Label>
                    <Select value={modelType} onValueChange={(v: any) => setModelType(v)}>
                      <SelectTrigger id="model-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tabular_cvae">Tabular CVAE (Recommended)</SelectItem>
                        <SelectItem value="mixed_data_cvae">Mixed Data CVAE</SelectItem>
                        <SelectItem value="tabular_vae_gmm">VAE + GMM</SelectItem>
                        <SelectItem value="tabular_ctgan">CTGAN</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="epochs" className="text-sm">Training Epochs</Label>
                    <Input
                      id="epochs"
                      type="number"
                      min="3"
                      max="1000"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                    />
                  </div>
                </div>
              </div>

              <Separator />

              {/* Generation Configuration */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">Step 4: Data Generation</h4>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="num-samples" className="text-sm">Number of Samples</Label>
                    <Input
                      id="num-samples"
                      type="number"
                      min="100"
                      max="100000"
                      value={numSamples}
                      onChange={(e) => setNumSamples(parseInt(e.target.value) || 1000)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="label-condition" className="text-sm">Label Condition</Label>
                    <Select value={labelCondition.toFixed(1)} onValueChange={(v) => setLabelCondition(parseFloat(v))}>
                      <SelectTrigger id="label-condition">
                        <SelectValue placeholder="Select label condition" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1.0">Positive Class (1.0)</SelectItem>
                        <SelectItem value="0.0">Negative Class (0.0)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            {!isComplete && (
              <Button 
                className="flex-1" 
                size="lg" 
                onClick={runWorkflow}
                disabled={isRunning || !datasetPath}
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Start Workflow
                  </>
                )}
              </Button>
            )}
            
            {isComplete && (
              <Button 
                className="flex-1" 
                size="lg" 
                onClick={() => navigate('/export')}
              >
                Export Data
                <ChevronRight className="w-4 h-4 ml-2" />
              </Button>
            )}
          </div>
        </div>
          )}
        </CardModernContent>
      </CardModern>
    </div>
  );
}

