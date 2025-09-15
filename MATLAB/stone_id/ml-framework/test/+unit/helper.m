classdef helper < matlab.unittest.TestCase
    %HELPER Summary of this class goes here
    %
    % Copyright 2021 The MathWorks Inc.
    
    methods ( TestClassSetup )
        function initialize( testCase )

        end
    end
    
    methods ( Test )

        function testValidatePipeline( testCase )
            
            parameters = validatePipelineParameters( ...
                "basePipelineClassification" );
            
            testCase.verifyNotEmpty( parameters )
            testCase.verifyEqual( parameters, "Normalization" )
            
            
        end %function
        
        function testValidatePipelineInPackage( testCase )
            
            %Static methods in package
            parameters = validatePipelineParameters( ...
                "pck.classification.basePipelineClassification" );
            
            testCase.verifyEqual( parameters, ["Normalization" "Selection"] )
            
            %Function in package
            parameters = validatePipelineParameters( ...
                "pck.basePipelineRegression" );
            
            testCase.verifyEqual( parameters, ["Normalization" "Engineering" "Selection"])
            
            %No options/parameters in arguments
            parameters = validatePipelineParameters("pck.classification.basePipelineNoOptions");
            testCase.verifyEqual( parameters, "")
            

        end %function
        
        
        function testValidateLearnerInPackage( testCase )
            
            types = [...
                "Classification"    "fitc"
                "Regression"        "fitr"
                "SemiSupervised"    "fits"
                "Cluster"           "clst"
                "PdM"               "fitdegradation"];
            
            
            for iType = 1:height(types)
                
                mthd = types(iType, 1);
                pck  = types(iType, 2);
                
                learners = experiment.enum.Learner.(mthd).Values;
                
                for iLearner = learners(:)'
                    
                    functionname = pck +"."+ iLearner;
                    parameters = validatePipelineParameters( functionname );
                    
                    testCase.verifyNotEmpty( parameters )
                    
                end %for iLearner
                
            end %for iType
            
            %Test for right answers
            parameters = validatePipelineParameters( "clst.spectral" );
            testCase.verifyEqual( parameters, ...
                ["FeatureNames" "Clusters" "Distance"] )
            
           parameters = validatePipelineParameters( "clst.som" );
            testCase.verifyEqual( parameters, ...
                ["FeatureNames" "Clusters" "CoverSteps" "InitNeighbor" "TopologyFcn" "Distance"] )
            
            parameters = validatePipelineParameters( "fitdegradation.expDegradation" );
            testCase.verifyEqual( parameters(:), ...
                ["HealthIndicatorName"
                "DataVariable"
                "LifeTimeVariable"
                "Theta"
                "ThetaVariance"
                "Beta"
                "BetaVariance"
                "Rho"
                "Phi"
                "NoiseVariance"
                "SlopeDetectionLevel"
                "UseParallel"
                "LifeTimeUnit"
                ])
            
            parameters = validatePipelineParameters( "fitc.auto" );
            testCase.verifyEqual( parameters(:), ...
                [ ...
                "PredictorNames"
                "ResponseName"
                "Include"
                "CrossValidation"
                "KFold"
                "Holdout"
                "Seed"
                "Cost"
                "Learners"
                "OptimizeHyperparameters"
                "HyperparameterOptimizationOptions"
                ])
   
            
        end %function
        
    end %methods
end %classdef

