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
import { Progress } from "@/components/ui/progress";
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
      // Store results in sessionStorage for Results page
      sessionStorage.setItem('latestGeneration', JSON.stringify({
        syntheticDataPath: data.output_path,
        originalDataPath: datasetPath,
        numSamples: data.num_generated, // Fixed: backend returns 'num_generated'
        modelPath,
        configPath,
      }));
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

      // Step 1: Labeling/Conversion
      // Note: Training config requires Parquet format, so we always need to label or convert
      setCurrentStep('labeling');
      
      let labelingFunctions: LabelingFunction[];
      
      if (skipLabeling) {
        // Create 3 simple abstain functions (Snorkel requires at least 3)
        // These don't actually label - they just convert CSV to Parquet format
        labelingFunctions = [
          {
            name: 'lf_abstain_1',
            code: `def lf_abstain_1(x):
    """Abstain function 1 - no labeling"""
    return -1`
          },
          {
            name: 'lf_abstain_2',
            code: `def lf_abstain_2(x):
    """Abstain function 2 - no labeling"""
    return -1`
          },
          {
            name: 'lf_abstain_3',
            code: `def lf_abstain_3(x):
    """Abstain function 3 - no labeling"""
    return -1`
          }
        ];
      } else {
        // Parse user-provided labeling functions
        labelingFunctions = parseLabelingFunctions(labelingFunctionsCode);
        
        if (labelingFunctions.length === 0) {
          throw new Error('No valid labeling functions provided');
        }
      }

      const labelingResult = await labelingMutation.mutateAsync({
        data_path: datasetPath,
        output_path: `output_data/labeled_${Date.now()}.parquet`,
        labeling_functions: labelingFunctions,
      });
      
      const dataPathForTraining = labelingResult.output_path;

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

  const isRunning = currentStep !== 'idle' && currentStep !== 'complete' && currentStep !== 'error';
  const isComplete = currentStep === 'complete';
  const hasError = currentStep === 'error';

  return (
    <CardModern>
      <CardModernHeader>
        <CardModernTitle>Data Generation Workflow</CardModernTitle>
      </CardModernHeader>
      <CardModernContent>
        <div className="space-y-6">
          {/* Workflow Progress */}
          {isRunning && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">
              {currentStep === 'labeling' && (skipLabeling ? '📋  Converting data format...' : '🏷️  Labeling data...')}
              {currentStep === 'config' && '⚙️  Creating configuration...'}
              {currentStep === 'training' && '🧠  Training model...'}
              {currentStep === 'generating' && '✨  Generating synthetic data...'}
                </span>
                <span className="text-muted-foreground">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
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
                    Skip custom labeling (use existing labels or convert format only)
                  </Label>
                </div>
                <p className="text-xs text-muted-foreground pl-6">
                  {skipLabeling 
                    ? "Data will be converted to required format without custom labels"
                    : "Define custom labeling functions below"}
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
                      min="10"
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
                    <Select value={labelCondition.toString()} onValueChange={(v) => setLabelCondition(parseFloat(v))}>
                      <SelectTrigger id="label-condition">
                        <SelectValue />
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
                onClick={() => navigate('/results')}
              >
                View Results
                <ChevronRight className="w-4 h-4 ml-2" />
              </Button>
            )}
          </div>
        </div>
      </CardModernContent>
    </CardModern>
  );
}

