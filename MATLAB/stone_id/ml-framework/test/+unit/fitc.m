classdef fitc < matlab.unittest.TestCase
    %FITC Unit tests for classification models 
    %

    %Notes: 
    %   -Review fitc.auto cross validation 

    % Copyright 2021 The MathWorks Inc.
    
    properties
       data
       hyperopts 
       optimizable
       metrics = {'modelType','errorOnResub','errorOnCV','precisionOnTrain',...
           'recallOnTrain','f1ScoreOnTrain','AUCOnTrain'};
    end

    
    properties (TestParameter)
        automl = { "auto" } %autointernal
        model  = { "tree", "discr", "nb", "knn", "svm", "ensemble", "ecoc", "linear", "kernel", "net", "treebagger"}
        CV = {"off", "KFold", "Leaveout", "Holdout"}
        hyperparameters = {"auto"}
    end
    

    methods ( TestClassSetup )
        
        function initialize( testCase )
            
            %Load Data
            load carsmall %#ok<LOAD>
            Origin = categorical(cellstr(Origin)); %#ok<NODEF>
            
            Origin = mergecats(Origin,{'France','Japan','Germany', ...
                'Sweden','Italy','England'},'NotUSA');
            
            this = table(Acceleration,Displacement,Horsepower, ...
                Model_Year,MPG,Weight,Origin);

            testCase.data = basePipelineClassification( this );
            
            testCase.hyperopts = struct(...
                "MaxObjectiveEvaluations",1, ...
                "Verbose", 0, ...
                "ShowPlots", 0, ...
                "UseParallel", false);
            
            testCase.optimizable = fitc.optimizablemodelslist();
            
        end %function 
        
    end %methods
    

    methods ( Test )
        function auto( testCase, automl )
            
            %Data preparation parameters
            [features, response, istraining] = i_getProperties(testCase.data);
        
            [mdl, info] = fitc.(automl)(...
                testCase.data,...
                "PredictorNames", features,...
                "ResponseName", response,...
                "Include", istraining, ...
                "OptimizeHyperparameters", "auto", ...
                "HyperparameterOptimizationOptions", testCase.hyperopts);
            
            testCase.verifyNotEmpty( info )
            testCase.verifyClass( info, 'table' )      
            testCase.verifyEqual(info.Properties.VariableNames,testCase.metrics)
            testCase.verifySize(info.precisionOnTrain, [1 numel(mdl.ClassNames)+1]) 
    
        end %function
            
        
        function models( testCase, model, CV, hyperparameters )
            
            %Data preparation parameters
            [features, response, istraining] = i_getProperties(testCase.data);
            
            [mdl, info] = fitc.(model)(...
                testCase.data,...
                "PredictorNames", features,...
                "ResponseName", response,...
                "Include", istraining, ...
                "CrossValidation",CV, ...
                "OptimizeHyperparameters", hyperparameters, ...
                "HyperparameterOptimizationOptions", testCase.hyperopts);
            
            testCase.verifyNotEmpty( info )
            testCase.verifyClass( info, 'table' )      
            testCase.verifyEqual(info.Properties.VariableNames,testCase.metrics)
            testCase.verifySize(info.precisionOnTrain, [1 numel(mdl.ClassNames)+1]) 
                    
        end %function

    end %methods
    

    methods ( Test )
       
        function tree( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.tree( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );

            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters 
            params = fitc.hyperparameters(testCase.data, "tree", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationTree" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function discr( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.discr( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "discr", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationDiscriminant" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
             
        end %function
        
        
        function nb( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.nb( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "nb", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationNaiveBayes" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function knn( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.knn( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "knn", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationKNN" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )

        end %function
        
        
        function svm( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
              %Train
            [mdl, info] = fitc.svm( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "svm", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationSVM" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )

        end %function
        
        
        function ensemble( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.ensemble( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "ensemble", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "classreg.learning.classif.ClassificationEnsemble" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function ecoc( testCase )
               
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.ecoc( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "ecoc", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationECOC" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
            
        end %function
        
        
        function linear( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.linear( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "linear", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationLinear" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function kernel( testCase )
                
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.kernel( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "kernel", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "ClassificationKernel" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function 
        
        
        function nnet( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.nnet( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining );
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation 
            metadata = join(info, infoT); %#ok<NASGU>
            
            testCase.verifyError(@()fitc.hyperparameters(testCase.data, "nnet"), "fitc:modelValidation")  
            testCase.verifyClass( mdl, "classificationNeuralNetwork" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function net( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.net( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining);
            
            %Predict
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
            
            %Evaluation
            metadata = join(info, infoT); %#ok<NASGU>
            
            %Hyperparameters
            params = fitc.hyperparameters(testCase.data, "net", ...
                "PredictorNames", features,...
                "ResponseName", response); %#ok<NASGU>

            testCase.verifyClass( mdl, "ClassificationNeuralNetwork" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
        
        
        function treebagger( testCase )
            
            [features, response, istraining] = i_getProperties( testCase.data );
            
            %Train
            [mdl, info] = fitc.treebagger( testCase.data, ...
                "PredictorNames", features,...
                "ResponseName", response, ...
                "Include", istraining);
 
            %Predict 
            [predictions, infoT] = fitc.predictandupdate( testCase.data(~istraining,:), mdl, ...
                "ResponseName", response);
             
            %Evaluation 
            metadata = join(info,infoT); %#ok<NASGU>
           
            %Hyperparameters 
            params = fitc.hyperparameters(testCase.data, "treebagger", ...
                "PredictorNames", features,...
                "ResponseName", response);
            
            testCase.verifyClass( mdl, "TreeBagger" )
            testCase.verifyClass( info, "table" )
            testCase.verifyClass( infoT, "table" )
            testCase.verifyEqual( info.modelType, infoT.modelType )
            testCase.verifyEqual( class( testCase.data.(response) ), ...
                class( predictions.("Prediction1") ) )
            
        end %function
  
    end %methods
    
end %classdef 

%Local Helper 
function [features, response, istraining] = i_getProperties(data)
    custom = data.Properties.CustomProperties;
    features = custom.Features;
    response = custom.Response;
    istraining = custom.TrainingObservations;
end %function

